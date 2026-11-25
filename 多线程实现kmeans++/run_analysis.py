# -*- coding: utf-8 -*-
"""
run_analysis.py —— Kaggle「商场客户细分」数据集完整分析
=======================================================

流程：EDA → 多线程 K-means++ 选 k（肘部法 + 轮廓系数）→ 最终聚类 →
      簇画像（客户分群解读）→ 分配步多线程基准（加速比/并行效率）

输出：figures/fig_eda.png        探索性分析图
      figures/fig_elbow.png     肘部法 + 轮廓系数选 k
      figures/fig_clusters.png  最终聚类结果 + 客户画像
      figures/fig_bench.png     多线程加速比
      tables/*.csv              各项数值表（供 MD 报告引用）
      results.json              关键数字汇总
"""
from __future__ import annotations

import json
import os
import sys

# 基准公平性：先钉死 BLAS 线程，再 import numpy
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from kmeans_parallel import kmeans_fit, silhouette_score, benchmark_assignment  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

DATA = os.path.join(BASE, "data", "Mall_Customers.csv")
FIGDIR = os.path.join(BASE, "figures")
TABDIR = os.path.join(BASE, "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

res = {}

# ==========================================================================
# 1. 读数据 + EDA
# ==========================================================================
df = pd.read_csv(DATA)
df.columns = ["CustomerID", "Gender", "Age", "Income", "Spending"]  # 简化列名
n, d_raw = df.shape
res["dataset"] = dict(n_rows=int(n), n_cols=int(d_raw),
                      columns=list(df.columns),
                      missing=int(df.isna().sum().sum()),
                      duplicated_id=int(df["CustomerID"].duplicated().sum()))
print(df.describe().round(2).to_string(), "\n")
df.describe().round(2).to_csv(os.path.join(TABDIR, "describe.csv"), encoding="utf-8-sig")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
ax = axes[0, 0]
cnt = df["Gender"].value_counts()
ax.bar(cnt.index, cnt.values, color=["#e8747c", "#5b8ff9"])
for i, v in enumerate(cnt.values):
    ax.text(i, v + 2, f"{v} ({v/len(df):.0%})", ha="center")
ax.set_title("性别分布"); ax.set_ylabel("人数")

for ax, col, color in [(axes[0, 1], "Age", "#61c0bf"),
                       (axes[0, 2], "Income", "#9d4edd"),
                       (axes[1, 0], "Spending", "#f4a261")]:
    ax.hist(df[col], bins=20, color=color, edgecolor="white")
    ax.axvline(df[col].mean(), ls="--", c="k", lw=1,
               label=f"均值 {df[col].mean():.1f}")
    ax.set_title(f"{col} 分布"); ax.legend()

ax = axes[1, 1]
for g, sub in df.groupby("Gender"):
    ax.scatter(sub["Income"], sub["Spending"], s=14, alpha=0.6, label=g)
ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score")
ax.set_title("收入 × 消费分（按性别着色）"); ax.legend()

ax = axes[1, 2]
ax.scatter(df["Age"], df["Spending"], s=14, alpha=0.6, c="#2a9d8f")
ax.set_xlabel("Age"); ax.set_ylabel("Spending Score")
ax.set_title("年龄 × 消费分")
fig.suptitle("图 1  Mall_Customers.csv 探索性分析（n=200）", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_eda.png"), bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# 2. 选 k：肘部法（WCSS）+ 轮廓系数（3 个种子取平均抑制随机性）
# ==========================================================================
X2 = df[["Income", "Spending"]].to_numpy(dtype=np.float64)
ks = np.arange(1, 10)
seeds = (0, 1, 2)
wcss = [float(np.mean([kmeans_fit(X2, int(k), n_threads=4, seed=s)[2]
                       for s in seeds])) for k in ks]
sil = {int(k): float(np.mean([silhouette_score(X2, kmeans_fit(X2, int(k),
        n_threads=4, seed=s)[1]) for s in seeds])) for k in range(2, 10)}
k_best = int(max(sil, key=sil.get))
res["model_selection"] = dict(ks=list(map(int, ks)),
                              wcss=[round(w, 1) for w in wcss],
                              silhouette={str(k): round(v, 4) for k, v in sil.items()},
                              k_best=k_best)
print(f"肘部法 + 轮廓系数 → 最佳 k = {k_best}  "
      f"(silhouette={sil[k_best]:.4f})\n")
pd.DataFrame({"k": ks, "WCSS": np.round(wcss, 1),
              "Silhouette": [sil.get(int(k), np.nan) for k in ks]}
             ).to_csv(os.path.join(TABDIR, "model_selection.csv"),
                      index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(ks, wcss, "o-", c="#e63946")
ax.axvline(k_best, ls="--", c="gray")
ax.annotate(f"肘部 k={k_best}", (k_best, wcss[k_best - 1]),
            xytext=(k_best + 0.6, wcss[k_best - 1] + 12000),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("簇数 k"); ax.set_ylabel("WCSS（簇内平方和）")
ax.set_title("肘部法"); ax.grid(alpha=0.3)
ax = axes[1]
ks2 = sorted(sil)
ax.plot(ks2, [sil[k] for k in ks2], "o-", c="#457b9d")
ax.axvline(k_best, ls="--", c="gray")
ax.set_xlabel("簇数 k"); ax.set_ylabel("平均轮廓系数")
ax.set_title(f"轮廓系数（最优 k={k_best}）"); ax.grid(alpha=0.3)
fig.suptitle("图 2  簇数选择", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_elbow.png"), bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# 3. 最终聚类（多线程 K-means++）+ 客户画像
# ==========================================================================
C, labels, inertia, n_iter, hist = kmeans_fit(X2, k_best, n_threads=8, seed=42)
res["final_fit"] = dict(k=k_best, inertia=round(inertia, 1),
                        n_iter=n_iter, threads=8,
                        silhouette=sil[k_best],
                        first_wcss=round(hist[0], 1))
print(f"最终聚类: k={k_best}, 迭代 {n_iter} 轮收敛, WCSS={inertia:.1f}\n")

prof = (df.assign(cluster=labels)
          .groupby("cluster")
          .agg(n=("CustomerID", "count"),
               income_mean=("Income", "mean"),
               spend_mean=("Spending", "mean"),
               age_mean=("Age", "mean"),
               male_pct=("Gender", lambda g: (g == "Male").mean()))
          .round(2)
          .sort_values(["income_mean", "spend_mean"], ascending=[False, False]))

def persona(r):
    hi_i, hi_s = r.income_mean >= X2[:, 0].mean(), r.spend_mean >= X2[:, 1].mean()
    if hi_i and hi_s:
        return "① 高收入高消费 · 目标VIP"
    if hi_i and not hi_s:
        return "② 高收入低消费 · 谨慎富裕型"
    if not hi_i and hi_s:
        return "③ 低收入高消费 · 冲动消费型"
    if not hi_i and not hi_s and r.income_mean < 45:
        return "④ 低收入低消费 · 低预算型"
    return "⑤ 中等收入中等消费 · 大众型"

prof["persona"] = prof.apply(persona, axis=1)
prof.to_csv(os.path.join(TABDIR, "cluster_profile.csv"), encoding="utf-8-sig")
print(prof[["persona", "n", "income_mean", "spend_mean", "age_mean",
            "male_pct"]].to_string(), "\n")
res["cluster_profile"] = prof.reset_index().to_dict("records")

cmap = plt.cm.tab10(np.linspace(0, 1, 10))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
for c in sorted(np.unique(labels)):
    sub = df[labels == c]
    name = prof.loc[c, "persona"]
    ax.scatter(sub["Income"], sub["Spending"], s=22, alpha=0.75,
               color=cmap[c], label=f"{name}  (n={len(sub)})")
ax.scatter(C[:, 0], C[:, 1], marker="*", s=320, c="black",
           edgecolors="white", linewidths=1.2, label="质心", zorder=5)
ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score (1-100)")
ax.set_title(f"K-means++ 聚类结果（k={k_best}，多线程分配步）")
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.72))
ax = axes[1]
w = 0.38
idx = np.arange(len(prof))
ax.bar(idx - w / 2, prof["income_mean"], w, label="平均收入 (k$)", color="#9d4edd")
ax.bar(idx + w / 2, prof["spend_mean"], w, label="平均消费分", color="#f4a261")
ax.set_xticks(idx)
ax.set_xticklabels([f"簇{c}\nn={int(r.n)}" for c, r in prof.iterrows()], fontsize=9)
ax.set_title("各簇画像：收入 vs 消费分"); ax.legend()
fig.suptitle("图 3  聚类结果与客户画像", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_clusters.png"), bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# 4. 分配步多线程基准（加速比 / 并行效率）
# ==========================================================================
threads_list = (1, 2, 4, 8, 16)
rows = benchmark_assignment(n_list=(100_000, 400_000, 1_000_000),
                            k=8, d=2, threads_list=threads_list, reps=5)
bench = pd.DataFrame(rows)
bench.to_csv(os.path.join(TABDIR, "benchmark.csv"), index=False, encoding="utf-8-sig")
base_t = bench[bench.threads == 1].set_index("n")["ms"]
bench["speedup"] = bench.apply(lambda r: base_t[r.n] / r.ms, axis=1)
bench["efficiency"] = bench["speedup"] / bench["threads"]
res["benchmark_base_ms"] = {str(int(k)): round(float(v), 2) for k, v in base_t.items()}
res["speedup_16t_1m"] = round(float(bench.query("n == 1000000 and threads == 16")["speedup"].iloc[0]), 2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
for nn, sub in bench.groupby("n"):
    ax.plot(sub.threads, sub.ms, "o-", label=f"n={nn:,}")
ax.set_xlabel("线程数"); ax.set_ylabel("分配步用时 (ms)")
ax.set_title("分配步耗时 vs 线程数"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax = axes[1]
for nn, sub in bench.groupby("n"):
    ax.plot(sub.threads, sub.speedup, "o-", label=f"n={nn:,}")
lim = threads_list[-1]
ax.plot([1, lim], [1, lim], "k--", lw=1, label="理想加速比 y=x")
ax.set_xlabel("线程数"); ax.set_ylabel("加速比 S = T1 / Tp")
ax.set_title(f"加速比（16 逻辑核；n=1M 时 16 线程达 {res['speedup_16t_1m']}×）")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.suptitle("图 4  多线程分配步基准测试（k=8, d=2, 5 次取最小）", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_bench.png"), bbox_inches="tight")
plt.close(fig)

with open(os.path.join(BASE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)
print("\n全部图表与表格已生成 → figures/ tables/ results.json")

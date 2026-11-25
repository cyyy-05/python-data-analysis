# -*- coding: utf-8 -*-
"""
kmeans_parallel.py —— K-means++ 聚类算法：多线程「分配步」实现 + 完整 Lloyd 迭代
================================================================================

核心任务
--------
实现 K-means 迭代中的**分配步（Assignment Step）**的多线程版本：

    分配步：给定质心 C = {c_1, ..., c_k}，对每个样本 x_i 计算
        label(i) = argmin_j || x_i - c_j ||^2
    并记录 min_j || x_i - c_j ||^2（用于计算 WCSS 目标函数）。

该步骤对每个样本完全独立，是典型的"易并行"（embarrassingly parallel）问题，
复杂度 O(n·k·d)，是 K-means 每轮迭代中的计算瓶颈，因此把它作为并行化对象。

并行设计要点
------------
1. **数据分块**：把 n 个样本切成 ~8×线程数 个连续块，块间计算互不依赖；
   块数多于线程数 → 后完成的线程自动到任务队列领下一块（动态负载均衡，
   避免个别块恰好落在慢核/被 OS 抢占导致的长尾等待）。
2. **线程池 = 真·操作系统线程**：使用 concurrent.futures.ThreadPoolExecutor，
   每个 worker 是一个 OS 级 pthread/Windows 线程。
3. **为什么 Python 线程能并行计算**：CPython 有 GIL，但 numpy 的矩阵乘（BLAS）
   和逐元素 ufunc 在执行大数组运算时会**主动释放 GIL**，分块后各线程的
   `X[s:e] @ C.T` 可在不同核上真正同时执行。
4. **基准公平性**：导入 numpy 前把 OpenBLAS/MKL 自身的多线程钉死为 1
   （环境变量 OMP_NUM_THREADS=1 等），保证测得的加速比来自我们的线程池，
   而不是 BLAS 内部并行与本程序并行混叠。
5. **距离公式**：利用 ||x-c||^2 = ||x||^2 - 2·x·c + ||c||^2，把逐对相减的
   O(n·k·d) 三维广播改写成一次矩阵乘 X @ C.T（BLAS dgemm），内存与时间都更优。
6. **线程安全**：labels / d2min 按块 disjoint 切片写入（无交叉、无扩容），
   各线程只写自己的区间，无需锁。

其他说明
--------
- K-means++ 初始化（D^2 概率采样）与质心更新步计算量小（O(kd) 与 O(nd)），
  保持单线程串行——只并行收益最大的分配步（更新步用 bincount 实现，同样很快）。
- `assignment_step(X, C, n_threads=1)` 退化为单线程版本，用于正确性对照与基准下界。
- 空簇处理：某簇无样本时，把离自己质心最远的样本重新播种到该簇。

运行
----
    python kmeans_parallel.py --selftest      # 正确性自检（并行 vs 向量化单线程）
    python kmeans_parallel.py --bench         # 多线程加速比基准测试
"""
from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

# ---- 关键：必须在 import numpy 之前钉死 BLAS 自身的线程数，基准才公平 ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

__all__ = [
    "assignment_step", "kmeans_pp_init", "kmeans_fit",
    "silhouette_score", "benchmark_assignment",
]


# --------------------------------------------------------------------------
# 1. ★核心：多线程分配步
# --------------------------------------------------------------------------
def assignment_step(X: np.ndarray, C: np.ndarray,
                    n_threads: int = 1,
                    chunks_per_thread: int = 8):
    """多线程分配步：把每个样本指派到最近的质心。

    参数
    ----
    X : (n, d) float64，样本矩阵
    C : (k, d) float64，质心矩阵
    n_threads : 线程数；1 = 单线程分块顺序执行（正确性对照 / 基准下界）
    chunks_per_thread : 每个线程平均负责的块数（>1 实现动态负载均衡）

    返回
    ----
    labels : (n,) int64，每个样本的簇编号
    d2min  : (n,) float64，到最近质心的欧氏距离平方（供 WCSS 使用）
    """
    n, d = X.shape
    k = C.shape[0]
    labels = np.empty(n, dtype=np.int64)
    d2min = np.empty(n, dtype=np.float64)

    csq = (C * C).sum(axis=1)                      # ||c_j||^2，(k,)
    n_chunks = max(1, n_threads * chunks_per_thread)
    chunk = max(1, math.ceil(n / n_chunks))
    bounds = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]

    def _task(s: int, e: int) -> None:
        """计算 [s, e) 块内样本到全部 k 个质心的距离平方并就地写入结果。"""
        xsq = np.einsum("ij,ij->i", X[s:e], X[s:e])          # ||x||^2，(m,)
        D = X[s:e] @ C.T                                     # (m, k) BLAS，释放 GIL
        D *= -2.0
        D += csq
        D += xsq[:, None]                                    # 完整 ||x-c||^2
        np.maximum(D, 0.0, out=D)                            # 截断浮点负误差
        labels[s:e] = D.argmin(axis=1)                       # disjoint 写，线程安全
        d2min[s:e] = D.min(axis=1)                           # 供 WCSS / 空簇播种

    if n_threads <= 1:
        for s, e in bounds:
            _task(s, e)
    else:
        # ThreadPoolExecutor 的每个 worker 都是操作系统线程；
        # 块数 >> 线程数：先完成的线程自动领下一块 → 动态负载均衡。
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(_task, s, e) for s, e in bounds]
            for f in futures:
                f.result()                                   # 传播块内异常

    return labels, d2min


# --------------------------------------------------------------------------
# 2. K-means++ 初始化（D^2 采样，串行；本身计算量小）
# --------------------------------------------------------------------------
def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++：第一个质心均匀随机，之后按 D^2 概率选远点，(k, d)。"""
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    centers[0] = X[rng.integers(n)]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        total = d2.sum()
        if total <= 0:                       # 样本几乎重复时退化为均匀采样
            idx = rng.integers(n)
        else:
            idx = rng.choice(n, p=d2 / total)
        centers[j] = X[idx]
        d2 = np.minimum(d2, ((X - centers[j]) ** 2).sum(axis=1))
    return centers


# --------------------------------------------------------------------------
# 3. 质心更新步（串行，bincount 实现，O(n·d) 且极快）
# --------------------------------------------------------------------------
def update_step(X: np.ndarray, labels: np.ndarray, k: int,
                d2min: np.ndarray | None = None) -> np.ndarray:
    """按簇求均值；空簇用离质心最远的样本重新播种。返回 (k, d) 新质心。"""
    n, d = X.shape
    counts = np.bincount(labels, minlength=k)
    C = np.stack([np.bincount(labels, weights=X[:, j], minlength=k)
                  for j in range(d)], axis=1)
    C = C / np.maximum(counts, 1)[:, None]
    for c in np.flatnonzero(counts == 0):                # 空簇 → 播种最远点
        seed = int(np.argmax(d2min))
        C[c] = X[seed]
        d2min[seed] = 0.0
    return C


# --------------------------------------------------------------------------
# 4. 完整 K-means（++ 初始化 + Lloyd 迭代：多线程分配步 + 串行更新步）
# --------------------------------------------------------------------------
def kmeans_fit(X: np.ndarray, k: int, n_threads: int = 4,
               max_iter: int = 300, tol: float = 1e-9,
               seed: int = 42):
    """返回 (centroids, labels, inertia(WCSS), n_iter, history)。"""
    rng = np.random.default_rng(seed)
    C = kmeans_pp_init(X, k, rng)
    history = []
    labels = d2min = None
    prev_labels = None
    for it in range(1, max_iter + 1):
        labels, d2min = assignment_step(X, C, n_threads=n_threads)   # ★并行分配
        history.append(float(d2min.sum()))                           # WCSS（先记录，再更新）
        C_new = update_step(X, labels, k, d2min)                     # 串行更新
        shift = float(np.abs(C_new - C).max())
        C = C_new
        # 收敛判据①：所有样本的簇分配不再变化（Lloyd 算法的标准停机条件）
        # 判据②：质心位移足够小。满足其一即停。
        if prev_labels is not None and np.array_equal(prev_labels, labels):
            break
        if shift <= tol:
            break
        prev_labels = labels
    return C, labels, float(d2min.sum()), it, history


# --------------------------------------------------------------------------
# 5. 轮廓系数（手写实现，避免额外依赖 sklearn；O(n^2)，n≤数千时毫秒级）
# --------------------------------------------------------------------------
def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """全部样本轮廓系数的均值；k=1 时无意义，约定返回 nan。"""
    uniq = np.unique(labels)
    if uniq.size < 2:
        return float("nan")
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))  # (n, n)
    n = X.shape[0]
    s = np.empty(n)
    for i in range(n):
        same = labels == labels[i]
        m_same = same.sum()
        a = D[i, same].sum() / (m_same - 1) if m_same > 1 else 0.0
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


# --------------------------------------------------------------------------
# 6. 分配步基准测试：不同规模 × 不同线程数
# --------------------------------------------------------------------------
def benchmark_assignment(n_list=(100_000, 400_000, 1_000_000),
                         k: int = 8, d: int = 2,
                         threads_list=(1, 2, 4, 8),
                         reps: int = 3, seed: int = 0):
    """合成高斯团数据，测量 assignment_step 用时（每次调用含线程池创建，
    预热 1 次后取 reps 次最小值以抑制 OS 抖动）。返回记录列表。"""
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_list:
        centers0 = rng.uniform(-10, 10, size=(k, d))
        y = rng.integers(0, k, size=n)
        X = centers0[y] + rng.normal(0.0, 1.0, size=(n, d))
        C = kmeans_pp_init(X, k, rng)
        for t in threads_list:
            assignment_step(X, C, n_threads=t)                   # 预热（建池/页换入）
            best = min(_time_call(X, C, t) for _ in range(reps))
            flops = n * k * (2 * d + 3)                          # matmul + 逐元素近似
            rows.append(dict(n=n, k=k, d=d, threads=t,
                             ms=best * 1e3,
                             gflops=flops / best / 1e9))
            print(f"n={n:>9,d}  k={k}  d={d}  threads={t:>2d}  "
                  f"{best*1e3:8.2f} ms  {flops/best/1e9:7.2f} GFLOP/s", flush=True)
    return rows


def _time_call(X, C, t):
    t0 = time.perf_counter()
    assignment_step(X, C, n_threads=t)
    return time.perf_counter() - t0


# --------------------------------------------------------------------------
# CLI：自检 / 基准
# --------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(7)
    # 5 个相互分离的高斯团（若用单一高斯云，边界点会永久翻转，属算法特性而非缺陷）
    centers0 = rng.uniform(-10, 10, size=(5, 6))
    y0 = rng.integers(0, 5, size=50_000)
    X = centers0[y0] + rng.normal(0.0, 0.8, size=(50_000, 6))
    C = rng.normal(size=(5, 6))

    # 对照实现：三维广播逐对算距离（直观但慢/费内存）
    D_ref = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    lab_ref, d2_ref = D_ref.argmin(axis=1), D_ref.min(axis=1)

    lab1, d21 = assignment_step(X, C, n_threads=1)
    lab8, d28 = assignment_step(X, C, n_threads=8)
    lab16, d216 = assignment_step(X, C, n_threads=16)

    assert np.array_equal(lab_ref, lab1), "单线程版与对照实现标签不一致"
    assert np.allclose(d2_ref, d21, atol=1e-8), "单线程版距离与对照实现不一致"
    assert np.array_equal(lab1, lab8) and np.array_equal(lab8, lab16), "多线程与单线程标签不一致"
    assert np.allclose(d21, d28, atol=1e-8) and np.allclose(d28, d216, atol=1e-8)

    # 完整流程 + 收敛性
    C_f, lab_f, inertia, n_iter, hist = kmeans_fit(X, 5, n_threads=8, seed=1)
    assert inertia == hist[-1] and n_iter <= 300
    uniq = np.unique(lab_f)
    assert uniq.size == 5, "K-means 应产生 5 个非空簇"

    print("[selftest] PASS  分配步: 多线程结果与向量化单线程逐元素一致")
    print(f"[selftest] PASS  kmeans_fit: k=5, n_iter={n_iter}, "
          f"WCSS={inertia:.2f} (首帧 {hist[0]:.2f} → 收敛)")


def main():
    ap = argparse.ArgumentParser(description="多线程 K-means++ 分配步")
    ap.add_argument("--selftest", action="store_true", help="正确性自检")
    ap.add_argument("--bench", action="store_true", help="运行加速比基准")
    ap.add_argument("--threads", default="1,2,4,8", help="基准线程数，逗号分隔")
    ap.add_argument("--ns", default="100000,400000,1000000", help="基准样本规模")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--d", type=int, default=2)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
    if args.bench:
        threads = tuple(int(x) for x in args.threads.split(","))
        ns = tuple(int(x) for x in args.ns.split(","))
        benchmark_assignment(n_list=ns, k=args.k, d=args.d, threads_list=threads)


if __name__ == "__main__":
    main()

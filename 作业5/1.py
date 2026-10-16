"""
SVM‑Spambase垃圾邮件分类
使用本地 spambase.data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 解决matplotlib中文方框乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1.读取本地数据集 =====================
col_names = [f"feat_{i}" for i in range(57)] + ["label"]
df = pd.read_csv("spambase.data", header=None, names=col_names)

print("数据集形状：", df.shape)
print("\n标签分布：")
print(df["label"].value_counts())

X = df.drop("label", axis=1)
y = df["label"]

# ===================== 2.数据集划分 & 标准化 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== 3.线性SVM Baseline =====================
print("="*50)
print("【1】线性SVM(LinearSVC)")
lin_svc = LinearSVC(C=1, max_iter=10000, random_state=42)
lin_svc.fit(X_train_scaled, y_train)

y_pred_lin = lin_svc.predict(X_test_scaled)
acc_lin = accuracy_score(y_test, y_pred_lin)
print(f"测试集准确率：{acc_lin:.4f}")
print(classification_report(y_test, y_pred_lin))

# ===================== 4.RBF核SVM + 网格搜索 n_jobs=1适配IDLE =====================
print("="*50)
print("【2】RBF核SVM，GridSearchCV搜索超参数C、γ")
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10]
}

rbf_svc = SVC(kernel="rbf", random_state=42)
grid = GridSearchCV(estimator=rbf_svc, param_grid=param_grid, cv=5, n_jobs=1, verbose=1)
grid.fit(X_train_scaled, y_train)

print("\n网格搜索最优参数：", grid.best_params_)
print(f"5折交叉验证最优得分：{grid.best_score_:.4f}")

best_rbf = grid.best_estimator_
y_pred_rbf = best_rbf.predict(X_test_scaled)
acc_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"RBF最优模型测试集准确率：{acc_rbf:.4f}")
print(classification_report(y_test, y_pred_rbf))

# ===================== 5.混淆矩阵打印 =====================
print("="*50)
print("线性SVM混淆矩阵：")
print(confusion_matrix(y_test, y_pred_lin))
print("RBF‑SVM混淆矩阵：")
print(confusion_matrix(y_test, y_pred_rbf))

# ===================== 6.结果对比可视化 =====================
models = ["Linear SVM", "RBF SVM(best params)"]
accs = [acc_lin, acc_rbf]

plt.figure(figsize=(6, 4))
bars = plt.bar(models, accs, color=["#2563eb", "#0d9488"])
plt.ylim((0.85, 0.96))
plt.ylabel("测试集准确率")
plt.title("SVM Spambase 模型准确率对比")
for bar, a in zip(bars, accs):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f"{a:.4f}", ha="center")
plt.savefig("svm_acc_compare.png", dpi=150, bbox_inches="tight")
plt.show()

# ===================== 7.输出支持向量信息(RBF) =====================
print("="*50)
print(f"RBF SVM支持向量数量：{best_rbf.n_support_}")
print(f"正常邮件类支持向量:{best_rbf.n_support_[0]},垃圾邮件类支持向量:{best_rbf.n_support_[1]}")

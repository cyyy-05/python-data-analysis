import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 解决matplotlib中文乱码
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# 加载Boston数据集
Boston = load_data('Boston')
X = Boston.drop('medv', axis=1)
y = Boston['medv']

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Z-score标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10折交叉验证训练模型
ridge = RidgeCV(alphas=np.logspace(-3, 6, 100), cv=10, scoring='neg_mean_squared_error')
ridge.fit(X_train_scaled, y_train)

lasso = LassoCV(alphas=np.logspace(-3, 3, 100), cv=10, random_state=42)
lasso.fit(X_train_scaled, y_train)

elastic = ElasticNetCV(alphas=np.logspace(-3,3,100), l1_ratio=[0.1,0.3,0.5,0.7,0.9], cv=10, random_state=42)
elastic.fit(X_train_scaled, y_train)

# 计算RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

ridge_pred = ridge.predict(X_test_scaled)
lasso_pred = lasso.predict(X_test_scaled)
elastic_pred = elastic.predict(X_test_scaled)

ridge_rmse = rmse(y_test, ridge_pred)
lasso_rmse = rmse(y_test, lasso_pred)
elastic_rmse = rmse(y_test, elastic_pred)

ridge_nonzero = np.sum(ridge.coef_ != 0)
lasso_nonzero = np.sum(lasso.coef_ != 0)
elastic_nonzero = np.sum(elastic.coef_ != 0)

print(f"Ridge: 测试RMSE={ridge_rmse:.2f}, 非零系数数量={ridge_nonzero}")
print(f"Lasso: 测试RMSE={lasso_rmse:.2f}, 非零系数数量={lasso_nonzero}")
print(f"ElasticNet: 测试RMSE={elastic_rmse:.2f}, 非零系数数量={elastic_nonzero}")

# 绘制系数路径图
fig, axes = plt.subplots(1,3, figsize=(18,5))

alphas_ridge = np.logspace(-3,6,100)
coefs_ridge = []
for a in alphas_ridge:
    m = Ridge(alpha=a).fit(X_train_scaled, y_train)
    coefs_ridge.append(m.coef_)
axes[0].plot(alphas_ridge, coefs_ridge)
axes[0].set_xscale("log")
axes[0].set_title("Ridge 系数路径")
axes[0].set_xlabel(r"$\lambda$")
axes[0].set_ylabel("系数")

alphas_lasso = np.logspace(-3,3,100)
coefs_lasso = []
for a in alphas_lasso:
    m = Lasso(alpha=a).fit(X_train_scaled, y_train)
    coefs_lasso.append(m.coef_)
axes[1].plot(alphas_lasso, coefs_lasso)
axes[1].set_xscale("log")
axes[1].set_title("Lasso 系数路径")
axes[1].set_xlabel(r"$\lambda$")

alphas_elastic = np.logspace(-3,3,100)
coefs_elastic = []
for a in alphas_elastic:
    m = ElasticNet(alpha=a, l1_ratio=elastic.l1_ratio_).fit(X_train_scaled, y_train)
    coefs_elastic.append(m.coef_)
axes[2].plot(alphas_elastic, coefs_elastic)
axes[2].set_xscale("log")
axes[2].set_title("ElasticNet 系数路径")
axes[2].set_xlabel(r"$\lambda$")

plt.tight_layout()
plt.show()

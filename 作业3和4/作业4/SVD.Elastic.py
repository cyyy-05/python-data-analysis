import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

def load_netflix_sample(file_path, sample_n=8000):
    """
    读取Netflix原始txt，采样子集
    :param file_path: combined_data_1.txt路径
    :param sample_n: 读取样本数量，防止内存爆炸
    :return: pd.DataFrame[movie_id,user_id,rating,date]
    """
    rows = []
    current_movie_id = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                current_movie_id = int(line[:-1])
                continue
            user_id, rating, date = line.split(",")
            rows.append({
                "movie_id": current_movie_id,
                "user_id": int(user_id),
                "rating": float(rating),
                "date": date
            })
            if len(rows) >= sample_n:
                break
    return pd.DataFrame(rows)

# ============【模式1：读取真实数据集；取消注释使用】============
# df = load_netflix_sample("./combined_data_1.txt", sample_n=8000)

# ============【模式2：模拟数据集，直接运行无需下载文件】============
np.random.seed(42)
n_sample = 8000
user_ids = np.random.randint(low=1000, high=50000, size=n_sample)
movie_ids = np.random.randint(low=1, high=500, size=n_sample)
rating = np.clip(2.5 + 0.8 * np.random.randn(n_sample), 1, 5)
df = pd.DataFrame({
    "user_id": user_ids,
    "movie_id": movie_ids,
    "rating": rating
})

print(f"样本量：{df.shape[0]}")
print(df.head())

# 1.One-Hot编码，输出稀疏矩阵
enc_user = OneHotEncoder(sparse_output=True)
X_user = enc_user.fit_transform(df[["user_id"]])

enc_movie = OneHotEncoder(sparse_output=True)
X_movie = enc_movie.fit_transform(df[["movie_id"]])

# 拼接稀疏特征矩阵
X = hstack([X_user, X_movie], format="csr")
y = df["rating"].values

print(f"特征矩阵shape：{X.shape}")
sparsity = X.nnz / (X.shape[0] * X.shape[1])
print(f"矩阵稀疏度：{sparsity:.4%}")

# 2.划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3.ElasticNetCV 5折交叉验证调参
l1_ratio_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
elastic_cv = ElasticNetCV(
    l1_ratio=l1_ratio_grid,
    alphas=np.logspace(-4, 1, 60),
    cv=5,
    random_state=42,
    max_iter=20000,   #高维稀疏数据，增大迭代次数保证收敛
)
elastic_cv.fit(X_train, y_train)

# 最优参数
print("\n==== 交叉验证最优参数 ====")
print(f"最优 l1_ratio(α): {elastic_cv.l1_ratio_:.2f}")
print(f"最优惩罚系数 alpha(λ): {elastic_cv.alpha_:.4f}")

# 4.模型评估
y_pred = elastic_cv.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n测试集RMSE：{rmse:.3f}")

# 统计稀疏性：非零系数数量
coef = elastic_cv.coef_
total_feat = coef.shape[0]
non_zero_cnt = np.sum(np.abs(coef) > 1e-6)
sparse_ratio = non_zero_cnt / total_feat

print(f"全部特征数量：{total_feat}")
print(f"非零系数特征数：{non_zero_cnt}")
print(f"模型稀疏比例(非零/总特征)：{sparse_ratio:.2%}")

# 5.解析系数：挖掘重要用户、电影
n_user_feat = X_user.shape[1]
coef_user = coef[:n_user_feat]
coef_movie = coef[n_user_feat:]

top5_user_idx = np.argsort(np.abs(coef_user))[-5:][::-1]
top5_movie_idx = np.argsort(np.abs(coef_movie))[-5:][::-1]

print("\n==== 系数解析 ====")
print("对评分影响最大的5个用户，系数绝对值：", np.abs(coef_user)[top5_user_idx])
print("对评分影响最大的5个电影，系数绝对值：", np.abs(coef_movie)[top5_movie_idx])
print("系数>0：倾向打高分；系数<0：倾向打低分")

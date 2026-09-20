# 第三题：Carseats 多元线性回归分析

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


# ==========================================
# 1. 加载 Carseats 数据集
# ==========================================

url = "https://raw.githubusercontent.com/selva86/datasets/master/Carseats.csv"

df = pd.read_csv(url)

print("Carseats 数据集前5行：")
print(df.head())


# ==========================================
# 2. 建立多元线性回归模型
# ==========================================
# 因变量：Sales
# 自变量：Price、Income、Advertising、ShelveLoc
#
# ShelveLoc 是分类变量
# 将 Bad 设置为基准组

model = smf.ols(
    "Sales ~ Price + Income + Advertising + "
    "C(ShelveLoc, Treatment(reference='Bad'))",
    data=df
).fit()


# ==========================================
# 3. 输出模型拟合报告
# ==========================================

print("\n================ 回归模型拟合报告 ================\n")

print(model.summary())


# ==========================================
# 4. 指出 ShelveLoc 的基准组
# ==========================================

print("\n================ ShelveLoc 基准组 ================\n")

print("ShelveLoc 的基准组是：Bad")


# ==========================================
# 5. 解释 ShelveLoc[Good] 系数
# ==========================================

good = "C(ShelveLoc, Treatment(reference='Bad'))[T.Good]"

good_coef = model.params[good]
good_pvalue = model.pvalues[good]

print("\n================ ShelveLoc[Good] ================\n")

print("ShelveLoc[Good] 系数：", round(good_coef, 4))
print("P值：", round(good_pvalue, 4))

print("\n商业含义：")

if good_coef > 0:
    print(
        f"在 Price、Income 和 Advertising 保持不变的情况下，"
        f"与 ShelveLoc 为 Bad 的商店相比，"
        f"ShelveLoc 为 Good 的商店，其 Sales 平均高 "
        f"{good_coef:.4f} 个单位。"
    )
else:
    print(
        f"在 Price、Income 和 Advertising 保持不变的情况下，"
        f"与 ShelveLoc 为 Bad 的商店相比，"
        f"ShelveLoc 为 Good 的商店，其 Sales 平均低 "
        f"{abs(good_coef):.4f} 个单位。"
    )


# ==========================================
# 6. 计算 VIF
# ==========================================

print("\n================ VIF 多重共线性检验 ================\n")

formula = (
    "Sales ~ Price + Income + Advertising + "
    "C(ShelveLoc, Treatment(reference='Bad'))"
)

y, X = dmatrices(
    formula,
    data=df,
    return_type="dataframe"
)

vif = pd.DataFrame()

vif["Variable"] = X.columns

vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print(vif.to_string(index=False))


# ==========================================
# 7. 判断是否存在多重共线性
# ==========================================

print("\n================ 多重共线性判断 ================\n")

for i in range(len(vif)):

    variable = vif.loc[i, "Variable"]
    value = vif.loc[i, "VIF"]

    if variable == "Intercept":
        continue

    if value < 5:
        result = "多重共线性较弱"
    elif value < 10:
        result = "存在一定程度的多重共线性"
    else:
        result = "存在较严重的多重共线性"

    print(
        f"{variable}: VIF = {value:.4f} → {result}"
    )


# ==========================================
# 8. 作业结论
# ==========================================

print("\n================ 第三题结论 ================\n")

print("1. 已建立 Sales 关于 Price、Income、Advertising 和 ShelveLoc 的多元线性回归模型。")
print("2. ShelveLoc 的基准组为 Bad。")
print(
    f"3. ShelveLoc[Good] 的回归系数为 {good_coef:.4f}，"
    "表示在其他变量保持不变时，Good 货架位置相对于 Bad 货架位置对 Sales 的平均影响。"
)

max_vif = vif.loc[vif["Variable"] != "Intercept", "VIF"].max()

print(f"4. 非截距变量中的最大 VIF 为 {max_vif:.4f}。")

if max_vif < 5:
    print("   综合判断：不存在明显的多重共线性问题。")
elif max_vif < 10:
    print("   综合判断：存在一定程度的多重共线性。")
else:
    print("   综合判断：存在较严重的多重共线性问题。")

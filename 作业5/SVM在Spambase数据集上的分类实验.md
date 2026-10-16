# SVM在Spambase数据集上的分类实验

## 1. 实验说明

本实验使用支持向量机（SVM）算法对 Spambase 垃圾邮件数据集进行分类。

Spambase 数据集主要用于判断一封邮件是否为垃圾邮件。数据集包含 57 个特征和 1 个分类标签，其中：

- `0`：正常邮件
- `1`：垃圾邮件

本实验使用 Python 和 Scikit-learn 完成数据预处理、模型训练和分类效果评价。

## 2. 数据处理

首先读取 `spambase.data` 数据集，将最后一列作为分类标签，其余 57 列作为特征。

然后按照 **7:3** 的比例划分训练集和测试集，并使用 `StandardScaler` 对特征进行标准化。

标准化的原因是 SVM 对特征尺度比较敏感，如果不同特征的取值范围差异较大，可能会影响模型的分类效果。

代码中的主要处理过程如下：

```
col_names = [f"feat_{i}" for i in range(57)] + ["label"]
df = pd.read_csv("spambase.data", header=None, names=col_names)

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 3. SVM分类模型

本实验使用了两种 SVM 模型进行比较。

### 3.1 Linear SVM

首先使用线性 SVM 建立基础模型：

```
lin_svc = LinearSVC(C=1, max_iter=10000, random_state=42)
lin_svc.fit(X_train_scaled, y_train)

y_pred_lin = lin_svc.predict(X_test_scaled)
acc_lin = accuracy_score(y_test, y_pred_lin)
```

Linear SVM 可以作为基础模型，用来观察 Spambase 数据集在线性分类情况下的效果。

### 3.2 RBF SVM

在 Linear SVM 的基础上，进一步使用 RBF 核函数：

```
rbf_svc = SVC(kernel="rbf", random_state=42)
```

RBF 核可以建立非线性的分类边界，因此可以处理更加复杂的分类关系。

RBF SVM 中比较重要的参数是 `C` 和 `gamma`。本实验使用 `GridSearchCV` 对这两个参数进行搜索：

```
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10]
}

grid = GridSearchCV(estimator=rbf_svc, param_grid=param_grid, cv=5, n_jobs=1, verbose=1)
grid.fit(X_train_scaled, y_train)
```

通过 5 折交叉验证选择最优参数，然后使用最优模型对测试集进行预测。

## 4. 实验结果

本实验对 Linear SVM 和 RBF SVM 的分类效果进行了比较。

| 模型       | 测试集准确率 |
| ---------- | ------------ |
| Linear SVM | 92.83%       |
| RBF SVM    | 92.76%       |

RBF SVM 通过网格搜索得到的最优参数为：

```
C = 10
gamma = 0.01
```

5 折交叉验证的最佳得分约为 **93.76%**。

模型准确率对比如下：

![](D:\python数据分析\作业5\svm_acc_compare.png)

## 5. 分类结果分析

从实验结果可以看出，两种 SVM 模型在 Spambase 数据集上的分类准确率都超过了 92%，说明 SVM 对该数据集具有较好的分类能力。

Linear SVM 的测试集准确率为 **92.83%**，略高于 RBF SVM 的 **92.76%**。这说明对于本次实验使用的 Spambase 数据集，线性的分类模型已经能够取得较好的效果，RBF 核增加的非线性能力并没有在测试集上带来明显的准确率提升。

另外，本实验还使用 `classification_report` 和 `confusion_matrix` 对模型进行了进一步评价。除了准确率之外，Precision、Recall 和 F1-score 可以从不同角度反映模型对正常邮件和垃圾邮件的识别效果。

## 6. 实验结论

通过本次实验，我使用 SVM 完成了 Spambase 垃圾邮件数据集的二分类任务。

实验过程中首先对数据进行了训练集、测试集划分和标准化，然后分别建立 Linear SVM 和 RBF SVM 模型，并使用 GridSearchCV 对 RBF SVM 的参数进行优化。

实验结果表明，两种模型都具有较好的分类效果，其中本次实验的 Linear SVM 测试集准确率略高于 RBF SVM。因此，对于当前数据集，Linear SVM 已经能够取得较好的分类效果。

通过本次实验，我进一步理解了 SVM 的分类思想、特征标准化、核函数以及超参数 `C` 和 `gamma` 对分类模型的影响。
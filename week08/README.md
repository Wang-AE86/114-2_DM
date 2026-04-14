# 第 8 週｜KNN、SVM 與決策樹

## 週次資訊

| 項目 | 說明 |
|------|------|
| 對應教科書 | Ch7 K 最近鄰、Ch8 支持向量機、Ch9 決策樹 |
| 日期 | 4/15（三）09:10–12:00 |
| Colab 程式 | [Ch7 KNN](https://colab.research.google.com/drive/1r6TRIRFWD5UmP8KTMWZaeo8W66b1LMPz)、[Ch8 SVM](https://colab.research.google.com/drive/1ZXKuUxTmRIaQcd6OVRtcNlKw3E0fB041)、[Ch9 Decision Tree](https://colab.research.google.com/drive/1TBS0721z22BuJbkuKQ9nwtaZh1G-jH8n) |
| 作業資料集 | `data/Ship_Performance_Dataset.csv`（船舶營運績效資料，全學期共用） |

---

# 教學內容

## 教學主題 1：K 最近鄰 KNN（Ch7，約 40 分鐘）

### 核心觀念

- KNN 是**懶惰學習**（Lazy Learning）：不建模，預測時才找最近的 K 個鄰居投票
- 距離計算：歐幾里得距離，所以**標準化非常重要**（不同量級的特徵會主宰距離）
- K 值選擇：K 太小 → 過擬合（對雜訊敏感）、K 太大 → 欠擬合（決策邊界太平滑）
- PCA 降維：高維資料中 KNN 效果差（維度詛咒），PCA 壓縮維度後改善

### 課堂範例：乳癌資料 KNN 分類 + 標準化影響

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# === 載入資料 ===
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 比較：有無標準化 ===
# 不標準化
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, knn_raw.predict(X_test))

# 有標準化（Pipeline）
pipe_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
pipe_knn.fit(X_train, y_train)
acc_scaled = accuracy_score(y_test, pipe_knn.predict(X_test))

print(f'不標準化 KNN 準確率：{acc_raw:.4f}')
print(f'有標準化 KNN 準確率：{acc_scaled:.4f}')
print(f'差異：{acc_scaled - acc_raw:+.4f}')

# === K 值選擇：測試 K=1~20 ===
k_range = range(1, 21)
scores = []
for k in k_range:
    pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])
    pipe.fit(X_train, y_train)
    scores.append(pipe.score(X_test, y_test))

plt.figure(figsize=(8, 4))
plt.plot(k_range, scores, 'bo-')
plt.xlabel('K 值')
plt.ylabel('準確率')
plt.title('KNN：K 值 vs 準確率')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.show()

best_k = list(k_range)[np.argmax(scores)]
print(f'\n最佳 K 值：{best_k}，準確率：{max(scores):.4f}')
```

### PCA 降維示範

```python
from sklearn.decomposition import PCA

# === 原始 30 維 vs PCA 降到 2 維 ===
pipe_full = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])
pipe_full.fit(X_train, y_train)

pipe_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
pipe_pca.fit(X_train, y_train)

print(f'原始 30 維 KNN：{pipe_full.score(X_test, y_test):.4f}')
print(f'PCA 2 維 KNN： {pipe_pca.score(X_test, y_test):.4f}')

# === PCA 2D 視覺化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlBu', alpha=0.6)
plt.xlabel(f'PC1（解釋 {pca.explained_variance_ratio_[0]:.1%} 變異）')
plt.ylabel(f'PC2（解釋 {pca.explained_variance_ratio_[1]:.1%} 變異）')
plt.title('PCA 降維後的乳癌資料分布')
plt.colorbar(scatter, label='0=惡性  1=良性')
plt.grid(True, alpha=0.3)
plt.show()
```

### 課堂重點提問

1. 為什麼 KNN 一定要標準化？→ 距離計算受量級影響，年齡(0-100)和收入(0-100萬)不標準化，收入會主宰距離
2. K=1 和 K=20 分別有什麼問題？→ K=1 對雜訊敏感（過擬合），K=20 太模糊（欠擬合）
3. PCA 降到 2 維準確率掉多少？值得嗎？→ 視覺化很值得，但預測通常用更多維度

---

## 教學主題 2：支持向量機 SVM（Ch8，約 45 分鐘）

### 核心觀念

- SVM 找的是**最大間隔超平面**：讓兩類資料之間的間隔最大
- 支持向量：距離超平面最近的點，決定邊界位置
- 核函數（Kernel）：線性不可分時，用核函數把資料映射到高維空間
  - `linear`：線性可分時用
  - `rbf`（預設）：大多數情況的好選擇
- 關鍵參數：
  - **C**：正則化。C 大 → 嚴格分類（可能過擬合）、C 小 → 容許錯誤（可能欠擬合）
  - **gamma**：rbf 核的影響範圍。gamma 大 → 每個點影響範圍小（複雜邊界）、gamma 小 → 影響範圍大（平滑邊界）

### 課堂範例：鐵達尼號 SVM 分類

```python
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# === 載入 Titanic 資料 ===
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)
titanic = titanic[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp']].dropna()

X = titanic[['Pclass', 'Age', 'Fare', 'SibSp']]
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 比較不同核函數 ===
kernels = ['linear', 'rbf']
for kernel in kernels:
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel=kernel, random_state=42))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n=== SVM kernel={kernel} ===')
    print(f'準確率：{acc:.4f}')
    print(classification_report(y_test, y_pred))

# === C 值對準確率的影響 ===
C_values = [0.01, 0.1, 1, 10, 100]
results_c = []
for c in C_values:
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=c, kernel='rbf', random_state=42))])
    pipe.fit(X_train, y_train)
    results_c.append(pipe.score(X_test, y_test))
    print(f'C={c:>6}  準確率：{results_c[-1]:.4f}')

plt.figure(figsize=(8, 4))
plt.plot(C_values, results_c, 'ro-')
plt.xscale('log')
plt.xlabel('C（log scale）')
plt.ylabel('準確率')
plt.title('SVM：C 值 vs 準確率')
plt.grid(True, alpha=0.3)
plt.show()
```

### 課堂重點提問

1. linear 和 rbf 哪個比較好？→ 取決於資料，線性可分用 linear 更快，否則 rbf 通常較好
2. C=100 比 C=0.01 好嗎？→ 不一定。C 太大可能過擬合訓練集
3. SVM 和 KNN 有什麼差別？→ SVM 建立模型（支持向量），KNN 不建模型（懶惰學習）；SVM 預測快，KNN 預測慢

---

## 教學主題 3：決策樹（Ch9，約 35 分鐘）

### 核心觀念

- 決策樹用**一連串 if-else 規則**分類，直覺好懂
- 分裂準則：Gini 不純度或 Entropy（資訊增益）
- **過擬合是決策樹最大的問題**：不限制深度會長成「完美記住訓練資料」的大樹
- 控制過擬合：`max_depth`、`min_samples_split`、`min_samples_leaf`
- 特徵重要性：決策樹可以告訴你哪個特徵最有用

### 課堂範例：鐵達尼號決策樹 + 過擬合控制

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# === 用 Titanic 資料（延續主題 2）===

# 不限制深度 vs 限制深度
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

dt_pruned = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
dt_pruned.fit(X_train, y_train)

print('=== 不限制深度 ===')
print(f'訓練準確率：{dt_full.score(X_train, y_train):.4f}')
print(f'測試準確率：{dt_full.score(X_test, y_test):.4f}')
print(f'樹深度：{dt_full.get_depth()}，葉節點數：{dt_full.get_n_leaves()}')

print('\n=== 限制深度（max_depth=3）===')
print(f'訓練準確率：{dt_pruned.score(X_train, y_train):.4f}')
print(f'測試準確率：{dt_pruned.score(X_test, y_test):.4f}')
print(f'樹深度：{dt_pruned.get_depth()}，葉節點數：{dt_pruned.get_n_leaves()}')

# === 視覺化決策樹 ===
plt.figure(figsize=(16, 8))
plot_tree(dt_pruned, feature_names=X.columns, class_names=['Dead', 'Survived'],
          filled=True, rounded=True, fontsize=10)
plt.title('決策樹（max_depth=3）')
plt.tight_layout()
plt.show()
```

### 特徵重要性

```python
# === 特徵重要性 ===
importances = pd.Series(dt_pruned.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 4))
importances.plot(kind='barh', color='steelblue')
plt.xlabel('重要性')
plt.title('決策樹：特徵重要性排名')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print('各特徵重要性：')
for feat, imp in importances.sort_values(ascending=False).items():
    print(f'  {feat}: {imp:.4f}')
```

### 三種模型比較總整理

```python
# === W8 三模型比較 ===
models = {
    'KNN (K=5)': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))]),
    'SVM (rbf)': Pipeline([('scaler', StandardScaler()), ('clf', SVC(random_state=42))]),
    'Decision Tree': Pipeline([('clf', DecisionTreeClassifier(max_depth=3, random_state=42))]),
}

print('=== 三模型比較（Titanic）===')
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)
    print(f'{name:20s}  訓練：{train_acc:.4f}  測試：{test_acc:.4f}  差距：{train_acc-test_acc:+.4f}')
```

### 課堂重點提問

1. 不限制深度時，訓練準確率接近 100% 但測試很差，這叫什麼？→ 過擬合
2. 決策樹需要標準化嗎？→ **不需要**！決策樹用的是 if-else 規則，不算距離
3. 特徵重要性最高的是什麼？→ Fare（票價）或 Pclass（艙等），反映社經地位對存活率的影響

---

## 教學與三模型對應總覽

| 模型 | 需標準化？ | 關鍵參數 | 優點 | 缺點 |
|------|----------|---------|------|------|
| KNN | **必須** | K 值 | 簡單直覺 | 預測慢、維度詛咒 |
| SVM | **必須** | C、gamma、kernel | 高維表現好 | 大資料集慢、參數敏感 |
| 決策樹 | **不需要** | max_depth、min_samples | 可解釋、特徵重要性 | 容易過擬合 |

---

# 作業

> **下週為期中考（W9），本週作業主題為交叉驗證與網格搜尋（Ch11-Ch12），**
> **作為期中考前的綜合練習，將本週學的三種模型搭配交叉驗證做比較。**

## 作業資訊

| 項目 | 說明 |
|------|------|
| 對應教科書 | Ch11 交叉驗證、Ch12 模型參數挑選和網格搜尋 |
| 繳交方式 | 在 Fork 的 week08/ 資料夾中建立三個檔案，push 到 Fork |
| 繳交期限 | 下週上課前 |
| PR 標題 | 學號_姓名（僅首次繳交時建立，之後 push 自動更新） |

---

## 第 1 題：5 折交叉驗證多模型比較

### 任務說明

使用 5 折交叉驗證，正確比較至少四種分類模型在同一份資料集上的效能。

### 測試資料

請使用以下程式碼載入資料：

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"資料形狀：{X.shape}")
print(f"類別分布：\n{pd.Series(y).value_counts()}")
print(f"類別名稱：{data.target_names}")
```

### Python 程式要求

撰寫程式碼完成以下工作：

1. 載入 breast_cancer 資料集
2. 建立至少四種分類模型的 Pipeline（含 StandardScaler）：
   - LogisticRegression
   - KNeighborsClassifier
   - SVC
   - DecisionTreeClassifier
3. 對每個模型使用 cross_val_score 進行 5 折交叉驗證
4. 印出每個模型的 5 次分數、平均分數、標準差
5. 繪製長條圖或箱型圖比較各模型的交叉驗證結果
6. 找出最佳模型

### 作答內容

請建立 `week08/q1_cross_validation.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== 交叉驗證結果 ===
LogisticRegression：5 次分數=???  平均=???  標準差=???
KNN：5 次分數=???  平均=???  標準差=???
SVM：5 次分數=???  平均=???  標準差=???
DecisionTree：5 次分數=???  平均=???  標準差=???

=== 最佳模型 ===
最佳模型：???
平均準確率：???
```

### 提示

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

models = {
    'LogisticRegression': LogisticRegression(max_iter=5000),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"{name}：{scores}  平均={scores.mean():.4f}  標準差={scores.std():.4f}")
```

---

## 第 2 題：GridSearchCV 超參數調校

### 任務說明

使用 GridSearchCV 對 SVM 進行系統化的超參數調校，找出最佳的 C 和 gamma 組合。

### Python 程式要求

使用第 1 題相同的 breast_cancer 資料集，完成以下工作：

1. 切割資料（test_size=0.2, random_state=42）
2. 建立 Pipeline：StandardScaler → SVC
3. 定義參數搜尋空間：
   - C：[0.1, 1, 10, 100]
   - gamma：['scale', 'auto', 0.01, 0.1]
   - kernel：['rbf', 'linear']
4. 使用 GridSearchCV（cv=5）進行搜尋
5. 印出最佳參數組合與對應的交叉驗證分數
6. 使用最佳模型對測試集預測，印出 Classification Report
7. 印出前 5 名的參數組合與分數

### 作答內容

請建立 `week08/q2_grid_search.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== 搜尋空間大小 ===
總共測試了幾種組合：???

=== 最佳結果 ===
最佳參數：???
最佳交叉驗證分數：???
測試集準確率：???

=== 前 5 名參數組合 ===
1. ???
2. ???
3. ???
4. ???
5. ???

=== 最佳模型的 Classification Report ===
（貼上分類報告）
```

### 提示

```python
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1],
    'svm__kernel': ['rbf', 'linear']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"最佳參數：{grid.best_params_}")
print(f"最佳交叉驗證分數：{grid.best_score_:.4f}")
print(f"測試集準確率：{grid.score(X_test, y_test):.4f}")

# 前 5 名
results_df = pd.DataFrame(grid.cv_results_)
top5 = results_df.nsmallest(5, 'rank_test_score')[['params', 'mean_test_score', 'rank_test_score']]
print(top5)
```

---

## 第 3 題：交叉驗證與網格搜尋觀念題

### 作答內容

請建立 `week08/q3_concept.txt`，回答以下問題：

```
姓名：
學號：

Q1：為什麼交叉驗證比單次 train_test_split 更能反映模型的真實效能？
    如果只做一次 train_test_split，可能會遇到什麼問題？
A1：???

Q2：GridSearchCV 中的 cv=5 代表什麼意思？
    如果參數搜尋空間有 4 個 C 值 × 4 個 gamma 值 × 2 個 kernel，
    加上 cv=5，總共需要訓練幾次模型？請列出計算過程。
A2：???

Q3：SVM 的參數 C 控制什麼？C 值很大和很小分別會造成什麼效果？
    這跟過擬合/欠擬合有什麼關係？
A3：???
```

---

## 繳交 Checklist

- [ ] week08/q1_cross_validation.txt 包含完整程式碼與四模型交叉驗證比較
- [ ] week08/q2_grid_search.txt 包含完整程式碼與 GridSearchCV 結果
- [ ] week08/q3_concept.txt 包含三題觀念回答
- [ ] 已 push 到自己的 Fork（W3 已建立的 PR 會自動更新）
- [ ] 已確認 PR 中可看到本週 commit

## 常見問題

**Q：cross_val_score 和 GridSearchCV 都有 cv 參數，有什麼差別？**
cross_val_score 是用來評估單一模型的效能，GridSearchCV 則是在多組參數中搜尋最佳組合。兩者都用 K-Fold 交叉驗證，但目的不同。

**Q：GridSearchCV 跑很久怎麼辦？**
加上 n_jobs=-1 可以使用所有 CPU 核心平行運算。也可以先縮小搜尋空間測試程式是否正確，確認後再擴大搜尋。

**Q：Pipeline 裡的參數名稱 'svm__C' 是怎麼來的？**
格式是「步驟名稱__參數名稱」。Pipeline 中 SVC 的步驟名稱是 'svm'，所以它的 C 參數就寫成 'svm__C'。

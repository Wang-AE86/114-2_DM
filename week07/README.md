# 第 7 週｜迴歸分析：線性迴歸與羅吉斯迴歸

## 週次資訊

| 項目 | 說明 |
|------|------|
| 對應教科書 | Ch4 簡單線性迴歸、Ch5 多元迴歸、Ch6 羅吉斯迴歸 |
| 日期 | 4/9（三）09:10–12:00 |
| Colab 程式 | [Ch4](https://colab.research.google.com/drive/18nr_4pOkjzvs-5uJoPWfKW5cU97KH0qP)、[Ch5](https://colab.research.google.com/drive/1iXifxI-AoJlkVMravR5rXP_2pDn8yYtS)、[Ch6](https://colab.research.google.com/drive/1gaq4UpvRb4CXUrgt_k6lJhMUAsakd1Ly) |
| 作業資料集 | `data/Ship_Performance_Dataset.csv`（船舶營運績效資料，全學期共用） |

---

# 教學內容

## 教學主題 1：簡單線性迴歸（Ch4，約 50 分鐘）

### 核心觀念

- 什麼是迴歸？用一條直線描述 X 與 Y 的關係
- 迴歸公式：ŷ = β × x + α（斜率 × 特徵 + 截距）
- 最小平方法（Least Squares）：讓所有預測誤差的平方和最小
- 評估指標：R²（決定係數）、RMSE（均方根誤差）

### 課堂範例：廣告花費 vs 銷售額

用一組簡單的廣告資料，示範從資料到模型的完整流程。

**Cell 1：匯入套件與建立資料**

```python
# 匯入需要的套件
import numpy as np                        # 數學運算（亂數、開根號等）
import pandas as pd                       # 表格資料處理
import matplotlib.pyplot as plt           # 繪圖
from sklearn.linear_model import LinearRegression       # 線性迴歸模型
from sklearn.model_selection import train_test_split    # 資料切割（訓練集/測試集）
from sklearn.metrics import r2_score, mean_squared_error  # 評估指標（R²、MSE）

# === 建立範例資料：廣告花費 vs 銷售額 ===
np.random.seed(42)                                      # 固定亂數種子，確保每次結果一樣
ad_spend = np.random.uniform(10, 100, 50)               # 隨機產生 50 筆廣告花費（10~100 萬元）
sales = 2.5 * ad_spend + np.random.normal(0, 15, 50) + 30  # 銷售額 = 2.5×廣告 + 雜訊 + 基礎值
df = pd.DataFrame({'ad_spend': ad_spend, 'sales': sales})   # 組成 DataFrame 表格

# === 資料切割：70% 訓練、30% 測試 ===
X = df[['ad_spend']]    # 特徵（用雙括號保持二維格式，sklearn 要求）
y = df['sales']          # 目標值
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f'訓練集：{X_train.shape[0]} 筆，測試集：{X_test.shape[0]} 筆')
```

**Cell 2：訓練模型 + 預測 + 評估**

```python
# === 建立並訓練線性迴歸模型 ===
model = LinearRegression()        # 建立模型物件
model.fit(X_train, y_train)       # 用訓練資料擬合模型（找出最佳的斜率和截距）

# 印出模型學到的參數
print(f'斜率 (coefficient): {model.coef_[0]:.4f}')      # 斜率：X 每增加 1，Y 增加多少
print(f'截距 (intercept):   {model.intercept_:.4f}')     # 截距：X=0 時 Y 的值
print(f'迴歸公式：sales = {model.coef_[0]:.2f} × ad_spend + {model.intercept_:.2f}')

# === 用模型預測訓練集和測試集 ===
y_pred_train = model.predict(X_train)   # 預測訓練集的銷售額
y_pred_test = model.predict(X_test)     # 預測測試集的銷售額

# === 評估模型表現 ===
# R²（決定係數）：0~1，越接近 1 表示模型越好，代表模型解釋了多少比例的變異
# RMSE（均方根誤差）：數值越小越好，代表預測值和實際值的平均偏差
print(f'\n訓練 R²: {r2_score(y_train, y_pred_train):.4f}')
print(f'測試 R²: {r2_score(y_test, y_pred_test):.4f}')
print(f'訓練 RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}')
print(f'測試 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}')
```

**Cell 3：視覺化**

```python
# === 視覺化：散佈圖 + 迴歸線 ===
plt.figure(figsize=(8, 5))                          # 設定圖片大小
plt.scatter(X_test, y_test, color='blue', label='Actual')   # 藍點 = 實際值
# 排序後畫線，避免線條鋸齒（因為 X_test 的順序是隨機的）
sort_idx = X_test.values.flatten().argsort()         # 取得由小到大的排序索引
plt.plot(X_test.values.flatten()[sort_idx], y_pred_test[sort_idx],
         color='red', linewidth=2, label='Predicted')  # 紅線 = 預測值
plt.xlabel('Ad Spend (萬元)')       # X 軸標籤
plt.ylabel('Sales (萬元)')          # Y 軸標籤
plt.title('簡單線性迴歸：廣告花費 vs 銷售額')
plt.legend()                        # 顯示圖例
plt.grid(True, alpha=0.3)           # 顯示網格線（透明度 30%）
plt.show()
```

### 課堂重點提問

1. 斜率 2.5 代表什麼？→ 廣告每多花 1 萬，銷售額增加約 2.5 萬
2. R² = 0.85 代表什麼？→ 模型解釋了 85% 的銷售變異
3. 訓練 R² > 測試 R² 正常嗎？→ 正常，但差太多就是過擬合

---

## 教學主題 2：多元迴歸（Ch5，約 40 分鐘）

### 核心觀念

- 多元迴歸：多個特徵同時預測 Y → ŷ = β₁x₁ + β₂x₂ + ... + α
- 為什麼要用多元？單一特徵可能不夠，多個特徵能提供更多資訊
- PolynomialFeatures：當關係不是直線時，加入多項式特徵

### 課堂範例：廣告花費（擴展為三管道）

從主題 1 的單一特徵擴展為三個廣告管道，示範多元迴歸的優勢。

**Cell 4：建立多管道資料**

```python
# === 建立模擬資料：三個廣告管道 → 銷售額 ===
np.random.seed(42)
n = 100   # 100 筆資料
df_multi = pd.DataFrame({
    'tv_spend': np.random.uniform(10, 300, n),    # 電視廣告花費（10~300 萬）
    'radio_spend': np.random.uniform(5, 50, n),   # 廣播廣告花費（5~50 萬）
    'web_spend': np.random.uniform(1, 80, n),     # 網路廣告花費（1~80 萬）
})
# 銷售額 = 各管道的加權總和 + 雜訊 + 基礎值
df_multi['sales'] = (
    0.05 * df_multi['tv_spend'] +      # 電視每花 1 萬 → 銷售增加 0.05 萬
    0.1 * df_multi['radio_spend'] +     # 廣播每花 1 萬 → 銷售增加 0.1 萬
    0.08 * df_multi['web_spend'] +      # 網路每花 1 萬 → 銷售增加 0.08 萬
    np.random.normal(0, 2, n) + 5       # 雜訊 + 基礎銷售額
)

# 設定特徵（3 個管道）和目標（銷售額）
X = df_multi[['tv_spend', 'radio_spend', 'web_spend']]
y = df_multi['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f'資料筆數：{n}，特徵數：{X.shape[1]}')
```

**Cell 5：簡單 vs 多元迴歸比較**

```python
# === 簡單迴歸：只用 tv_spend 一個特徵 ===
lr_simple = LinearRegression()
lr_simple.fit(X_train[['tv_spend']], y_train)                          # 只用電視廣告訓練
r2_simple = r2_score(y_test, lr_simple.predict(X_test[['tv_spend']]))  # 計算測試 R²

# === 多元迴歸：三個管道全部使用 ===
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)                              # 用全部 3 個特徵訓練
r2_multi = r2_score(y_test, lr_multi.predict(X_test))       # 計算測試 R²

# 比較結果：多元迴歸的 R² 應該比簡單迴歸高
print('=== 簡單迴歸（只用 tv_spend）===')
print(f'測試 R²: {r2_simple:.4f}')

print('\n=== 多元迴歸（三管道）===')
print(f'測試 R²: {r2_multi:.4f}')

# 印出每個特徵的迴歸係數（代表該特徵對銷售的影響力）
print(f'\n各特徵迴歸係數：')
for name, coef in zip(X.columns, lr_multi.coef_):
    print(f'  {name}: {coef:.4f}')
print(f'  截距: {lr_multi.intercept_:.4f}')
```

### 課堂重點提問

1. 多元的 R² 比簡單高多少？→ 多了其他管道的資訊
2. 哪個管道影響最大？→ 看係數大小（注意要標準化後才能比）
3. 加更多特徵一定更好嗎？→ 不一定，可能過擬合或加入雜訊

---

## 教學主題 3：羅吉斯迴歸（Ch6，約 50 分鐘）

### 核心觀念

- 羅吉斯迴歸：用迴歸的方法做二元分類（是/否、故障/正常）
- Sigmoid 函數：把迴歸值壓縮到 0~1，當作機率
- 混淆矩陣：TP、TN、FP、FN 四格的意義
- ROC 曲線與 AUC：衡量分類模型的整體表現

### 課堂範例：鐵達尼號生存預測

用經典的 Titanic 資料，示範從特徵處理到模型評估的完整流程。

**Cell 6：載入 Titanic 資料 + 訓練羅吉斯迴歸**

```python
# 匯入羅吉斯迴歸相關套件
from sklearn.linear_model import LogisticRegression       # 羅吉斯迴歸（分類用）
from sklearn.preprocessing import StandardScaler          # 標準化（讓特徵量級一致）
from sklearn.pipeline import Pipeline                     # 管道器（串接多個步驟）
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)  # 分類評估工具

# === 載入鐵達尼號資料 ===
# 線上載入（從 GitHub 下載 CSV）
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)
# 若網路不通，改用離線備份：
# titanic = pd.read_csv('data/titanic.csv')

# 只選需要的欄位，並移除有遺漏值的列
# Survived=是否存活, Pclass=艙等, Age=年齡, Fare=票價, SibSp=同行兄弟姊妹/配偶數
titanic = titanic[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp']].dropna()
X = titanic[['Pclass', 'Age', 'Fare', 'SibSp']]   # 4 個特徵
y = titanic['Survived']                              # 目標：0=死亡, 1=存活

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 建立 Pipeline：標準化 → 羅吉斯迴歸 ===
# Pipeline 會自動依序執行：先標準化，再訓練模型
pipe = Pipeline([
    ('scaler', StandardScaler()),              # 第一步：將特徵標準化（平均=0, 標準差=1）
    ('lr', LogisticRegression(random_state=42))  # 第二步：羅吉斯迴歸
])
pipe.fit(X_train, y_train)       # 訓練（標準化 + 擬合模型一步完成）
y_pred = pipe.predict(X_test)    # 預測測試集

# === 印出各特徵的迴歸係數 ===
# 正值 = 增加存活機率，負值 = 降低存活機率
print('=== 各特徵迴歸係數（標準化後）===')
for name, coef in zip(X.columns, pipe.named_steps['lr'].coef_[0]):
    print(f'  {name}: {coef:.4f}')

# === 分類報告：精確率、召回率、F1 ===
print('\n=== Classification Report ===')
print(classification_report(y_test, y_pred))
```

**Cell 7：混淆矩陣 + ROC 曲線**

```python
# === 視覺化：混淆矩陣（左）+ ROC 曲線（右）===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))   # 建立 1×2 的子圖

# 左圖：混淆矩陣（顯示 TP/TN/FP/FN 四格數字）
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0])
axes[0].set_title('Confusion Matrix')

# 右圖：ROC 曲線（評估分類模型的整體表現）
y_prob = pipe.predict_proba(X_test)[:, 1]   # 取得每筆資料「存活」的機率
fpr, tpr, _ = roc_curve(y_test, y_prob)     # 計算不同門檻值下的 FPR 和 TPR
roc_auc = auc(fpr, tpr)                     # 計算曲線下面積（AUC）

axes[1].plot(fpr, tpr, color='blue', linewidth=2, label=f'AUC = {roc_auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)   # 對角虛線 = 隨機猜測的基準線
axes[1].set_xlabel('False Positive Rate')    # X 軸：誤報率
axes[1].set_ylabel('True Positive Rate')     # Y 軸：正確預測率
axes[1].set_title('ROC Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()   # 自動調整子圖間距
plt.show()

print(f'\nAUC = {roc_auc:.3f}')   # AUC：0.5=隨機, 0.7-0.8=可接受, 0.8-0.9=好, >0.9=很好
```

### 課堂重點提問

1. Pclass 係數為負代表什麼？→ 艙等越高（數字越小）存活率越高
2. 混淆矩陣的 FN（漏報）在這裡代表什麼？→ 實際生還但預測死亡
3. AUC = 0.80 算好嗎？→ 0.7-0.8 可接受，0.8-0.9 好，>0.9 很好

---

# 作業

> **本學期作業統一使用船舶營運績效資料集**（`data/Ship_Performance_Dataset.csv`）。
> 每週用不同方法分析同一組資料，逐步加深。
>
> **設計原則**：每題作業都是對應教學主題的**小修改**，換成船舶資料，程式碼結構和步驟完全相同。

## 資料集說明

船舶營運績效資料集包含 2,736 筆船舶航次紀錄，涵蓋 4 種船型的營運數據。

載入方式：

```python
import pandas as pd
df = pd.read_csv('data/Ship_Performance_Dataset.csv')
print(df.head())
print(f'\n資料形狀：{df.shape}')
print(f'\n欄位：{df.columns.tolist()}')
```

本週會用到的欄位：

| 欄位 | 說明 | 本週用途 |
|------|------|---------|
| Engine_Power_kW | 引擎功率（kW） | Q1 簡單迴歸的 X |
| Cargo_Weight_tons | 貨物重量（噸） | Q1 多元迴歸特徵 |
| Distance_Traveled_nm | 航行距離（浬） | Q1 多元迴歸特徵 |
| Speed_Over_Ground_knots | 航速（節） | Q1 多元迴歸特徵 |
| Draft_meters | 吃水深度（公尺） | Q1 多元迴歸特徵 |
| Operational_Cost_USD | 營運成本（美元） | **Q1 迴歸預測目標** |
| Maintenance_Status | 維護狀態（Good/Fair/Critical） | **Q2 分類目標** |

---

## 第 1 題：簡單迴歸 + 多元迴歸 — 預測船舶營運成本

> 對應教學主題 1 + 2。
> 課堂用「廣告花費 → 銷售額」，作業換成「引擎功率 → 營運成本」。
> 程式碼結構一模一樣，只需換 `pd.read_csv()` 和欄位名。

### 程式要求（對照課堂範例修改）

| 步驟 | 課堂做的 | 作業要做的（只需改的地方） |
|------|---------|-------------------------|
| 資料 | 廣告花費（自建） | `data/Ship_Performance_Dataset.csv` |
| 簡單迴歸 X | `ad_spend` | `Engine_Power_kW` |
| 簡單迴歸 Y | `sales` | `Operational_Cost_USD` |
| 多元迴歸 X | `tv, radio, web`（3 個） | `Engine_Power_kW, Cargo_Weight_tons, Distance_Traveled_nm, Speed_Over_Ground_knots, Draft_meters`（5 個） |
| 多元迴歸 Y | `sales` | `Operational_Cost_USD` |
| 其餘 | 完全相同 | 完全相同 |

### 作答內容

請建立 `week07/q1_regression.txt`：

```
姓名：
學號：

=== 完整程式碼 ===

=== 簡單迴歸結果 ===
使用特徵：Engine_Power_kW
斜率：???   截距：???
迴歸公式：Cost = ??? × Engine_Power_kW + ???
訓練 R²：???   測試 R²：???
訓練 RMSE：??? 測試 RMSE：???

=== 多元迴歸結果 ===
各特徵迴歸係數：（列出 5 個）
訓練 R²：???   測試 R²：???
訓練 RMSE：??? 測試 RMSE：???

=== 簡單 vs 多元比較 ===
R² 差了多少？RMSE 差了多少？為什麼？
哪個特徵對營運成本的影響最大？
```

---

## 第 2 題：羅吉斯迴歸 — 船舶維護狀態預測

> 對應教學主題 3。
> 課堂用「鐵達尼號生存預測」，作業換成「船舶維護狀態是否為 Critical」。
> Pipeline 結構一模一樣（StandardScaler + LogisticRegression），只需換資料。

### 資料前處理

維護狀態有三類（Good / Fair / Critical），需轉為二元分類：

```python
# 建立二元目標：Critical = 1, 其他 = 0
df['is_critical'] = (df['Maintenance_Status'] == 'Critical').astype(int)

# 移除遺漏值
df_clean = df.dropna(subset=['Maintenance_Status'])

# 選取數值特徵
features = ['Engine_Power_kW', 'Speed_Over_Ground_knots', 'Distance_Traveled_nm',
            'Draft_meters', 'Cargo_Weight_tons', 'Turnaround_Time_hours']
X = df_clean[features]
y = df_clean['is_critical']
```

### 程式要求（對照課堂範例修改）

| 步驟 | 課堂做的（Titanic） | 作業要做的（只需改的地方） |
|------|---------------------|-------------------------|
| 資料 | Titanic CSV | Ship Performance（上方程式碼） |
| 目標 | `Survived` | `is_critical` |
| 特徵 | `Pclass, Age, Fare, SibSp` | `Engine_Power_kW` 等 6 個 |
| Pipeline | StandardScaler + LogisticRegression | 完全相同 |
| 評估 | classification_report + 混淆矩陣 + ROC | 完全相同 |

### 作答內容

請建立 `week07/q2_logistic.txt`：

```
姓名：
學號：

=== 完整程式碼 ===

=== 各特徵迴歸係數 ===
Engine_Power_kW:          ???
Speed_Over_Ground_knots:  ???
Distance_Traveled_nm:     ???
Draft_meters:             ???
Cargo_Weight_tons:        ???
Turnaround_Time_hours:    ???

=== 影響最大的特徵 ===
特徵名稱：???
原因：???

=== 分類報告 ===
（貼上 classification_report 輸出）

=== AUC 值 ===
???

=== 混淆矩陣判讀 ===
TP：???  FP：???
FN：???  TN：???
在船舶維護預測中，FN（漏報 Critical）比 FP（誤報 Critical）更嚴重，因為：???
```

---

## 第 3 題：迴歸觀念題

> 對應教學主題 1-3 的課堂提問延伸。

請建立 `week07/q3_concept.txt`：

```
姓名：
學號：

Q1：課堂上我們用單一特徵（ad_spend）和三個特徵（tv, radio, web）
    分別做了迴歸。作業中你也比較了單一特徵和多特徵。
    多元迴歸的 R² 比簡單迴歸高多少？
    加更多特徵一定會讓模型更好嗎？什麼情況下反而會變差？
A1：???

Q2：R²（決定係數）和 RMSE（均方根誤差）都可以評估迴歸模型。
    它們的差異是什麼？在預測船舶營運成本的情境下，
    你會更看重哪一個指標？為什麼？
A2：???

Q3：課堂上 Titanic 的混淆矩陣中，FN 代表「實際生還但預測死亡」。
    作業中船舶維護的 FN 代表「實際 Critical 但預測正常」。
    在不同的應用情境中（例如：醫療診斷、垃圾郵件過濾、自駕車障礙偵測），
    FN 和 FP 哪個更嚴重？請任選一個情境說明。
A3：???
```

---

## 教學與作業對應總覽

| 教學主題 | 課堂範例 | 作業題目 | 改了什麼 |
|---------|---------|---------|---------|
| 主題 1：簡單迴歸 | 廣告花費 → 銷售額 | Q1 前半：引擎功率 → 營運成本 | 換船舶資料 |
| 主題 2：多元迴歸 | 三管道廣告 → 銷售額 | Q1 後半：5 特徵 → 營運成本 | 換船舶資料、多特徵 |
| 主題 3：羅吉斯迴歸 | Titanic 生存預測 | Q2：船舶維護狀態 Critical 預測 | 換海事情境 |
| 課堂提問延伸 | 三個主題的重點問答 | Q3：觀念題（3 題） | 連結海事情境 |

---


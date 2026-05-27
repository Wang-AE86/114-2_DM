# 第 3 週作業：資料預處理

## 作業資訊

| 項目 | 說明 |
|------|------|
| 對應教科書 | Ch3 資料預處理 |
| 繳交方式 | 在 Fork 的 week03/ 資料夾中建立四個檔案，push 到 Fork |
| 繳交期限 | 下週上課前 |
| PR 標題 | 學號_姓名（僅首次繳交時建立，之後 push 自動更新） |

---

## 第 1 題：Fork + PR 繳交測試

### 任務說明

本題的目的是確認你能順利完成 Fork → 建立檔案 → Commit → Push → 建立 PR（首次）的完整流程。只要成功發出 PR 且檔案內容正確，即為通過。

### 操作步驟

1. 確認你已經 Fork 老師的 Repo 到自己的 GitHub 帳號
2. 在你的 Fork 中建立 `week03/` 資料夾
3. 在 `week03/` 中建立 `q1_hello.txt`
4. 填入以下內容後 Commit 並 Push
5. 到 GitHub 網頁發 Pull Request 回老師的 Repo（僅首次需要，之後每週 push 會自動更新此 PR）

### 作答內容

請建立 `week03/q1_hello.txt`，依照以下格式填寫：

```
姓名：
學號：
系級：

=== 完成確認 ===
我已成功完成以下步驟：
1. Fork 老師的 Repo：（是/否）
2. 建立 week03 資料夾：（是/否）
3. 建立本檔案 q1_hello.txt：（是/否）
4. Commit 並 Push 到自己的 Fork：（是/否）
5. 發出 Pull Request：（是/否）

=== 使用的 Git 操作方式 ===
（請填寫你使用的方式：GitHub 網頁 / VS Code / 終端機指令）

=== 遇到的問題 ===
（如果操作過程中有遇到任何問題，請簡單描述。沒有的話填「無」）
```

---

## 第 2 題：數值型資料預處理與 Pipeline 實作

### 任務說明

使用 Scikit-learn 的轉換器與 Pipeline，對一份含有遺漏值的數值型資料進行預處理。你需要完成遺漏值填補與標準化，並將流程串接成 Pipeline。

### 操作步驟

1. 在 Google Colab 中建立新的 Notebook
2. 建立測試資料（包含遺漏值）
3. 使用 SimpleImputer 填補遺漏值
4. 使用 StandardScaler 進行標準化
5. 將上述步驟串接成 Pipeline
6. 觀察處理前後的資料變化

### 測試資料

請使用以下程式碼建立測試資料：

```python
import pandas as pd
import numpy as np

# 建立含遺漏值的海洋觀測資料
data = {
    'water_temp': [26.5, 27.3, np.nan, 28.1, 26.9, np.nan, 25.8, 27.6, 29.0, 26.2],
    'salinity': [34.2, np.nan, 33.8, 34.5, 33.9, 34.1, np.nan, 34.3, 33.7, 34.0],
    'wave_height': [1.2, 2.5, 1.8, np.nan, 3.1, 2.0, 1.5, np.nan, 2.8, 1.9],
    'wind_speed': [15.3, 22.1, 18.5, 20.0, np.nan, 16.8, 19.2, 21.5, np.nan, 17.4]
}
df = pd.DataFrame(data)
print("原始資料：")
print(df)
print(f"\n資料形狀：{df.shape}")
print(f"\n各欄位遺漏值數量：\n{df.isnull().sum()}")
```

### Python 程式要求

撰寫程式碼完成以下工作：

1. 使用 SimpleImputer（策略為 mean）填補遺漏值
2. 使用 StandardScaler 進行標準化
3. 將 SimpleImputer 和 StandardScaler 串接成 Pipeline
4. 用 Pipeline 的 fit_transform 對資料進行轉換
5. 將轉換結果放回 DataFrame，印出處理後的資料
6. 印出每個欄位的平均值和標準差，確認標準化結果

### 作答內容

請建立 `week03/q2_preprocessing.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 原始資料與遺漏值統計 ===
（貼上原始資料的 print 結果和遺漏值統計）

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== Pipeline 處理後的資料 ===
（貼上 Pipeline 轉換後的 DataFrame）

=== 標準化驗證 ===
（貼上每個欄位的平均值和標準差，確認平均值接近 0、標準差接近 1）
```

### 提示

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 建立 Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 轉換資料
result = num_pipeline.fit_transform(df)
```

---

## 第 3 題：ColumnTransformer 整合數值與類別管道

### 任務說明

實際資料通常同時包含數值型和類別型欄位，需要用不同的方式處理。本題要求你使用 ColumnTransformer 將數值管道和類別管道合併，完成混合型資料的預處理。

### 測試資料

請使用以下程式碼建立測試資料：

```python
import pandas as pd
import numpy as np

# 建立混合型的港口船舶資料
data = {
    'ship_length': [180.5, 220.3, np.nan, 150.8, 310.2, 195.0, np.nan, 275.6, 160.4, 240.1],
    'cargo_weight': [45000, 62000, 38000, np.nan, 85000, 51000, 42000, np.nan, 35000, 58000],
    'ship_type': ['貨櫃船', '散裝船', '油輪', '貨櫃船', '貨櫃船', np.nan, '散裝船', '油輪', '散裝船', np.nan],
    'port': ['高雄港', '基隆港', '高雄港', '台中港', '高雄港', '基隆港', '台中港', '高雄港', np.nan, '基隆港'],
    'is_delayed': [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)
print("原始資料：")
print(df)
print(f"\n各欄位資料型態：\n{df.dtypes}")
print(f"\n各欄位遺漏值：\n{df.isnull().sum()}")
```

### Python 程式要求

撰寫程式碼完成以下工作：

1. 區分數值型欄位和類別型欄位
2. 建立數值管道：SimpleImputer（mean）→ StandardScaler
3. 建立類別管道：SimpleImputer（most_frequent）→ OneHotEncoder
4. 使用 ColumnTransformer 合併兩條管道
5. 對資料進行 fit_transform
6. 印出轉換後的欄位名稱與資料形狀
7. 印出轉換後的前 5 筆資料

### 作答內容

請建立 `week03/q3_column_transformer.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 欄位分類 ===
數值型欄位：???
類別型欄位：???
目標欄位：???

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== 轉換後的資料形狀 ===
（貼上轉換後的 shape）

=== 轉換後的欄位名稱 ===
（貼上 OneHotEncoder 產生的新欄位名稱）

=== 轉換後的前 5 筆資料 ===
（貼上轉換結果）
```

### 提示

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# 定義欄位
num_cols = ['ship_length', 'cargo_weight']
cat_cols = ['ship_type', 'port']

# 建立 ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse_output=False))
        ]), cat_cols)
    ]
)

# 轉換
X = df.drop('is_delayed', axis=1)
y = df['is_delayed']
X_transformed = preprocessor.fit_transform(X)
```

---

## 第 4 題：資料預處理觀念題

### 作答內容

請建立 `week03/q4_concept.txt`，回答以下問題：

```
姓名：
學號：

Q1：為什麼資料預處理需要區分數值型與類別型資料？
    如果把類別型資料直接丟進 StandardScaler 會發生什麼問題？
    如果把數值型資料直接丟進 OneHotEncoder 又會怎樣？
A1：???

Q2：SimpleImputer 的 strategy 參數有 mean、median、most_frequent 三種常用策略。
    請分別說明這三種策略各自適合什麼樣的資料情境，並各舉一個例子。
A2：???

Q3：使用 Pipeline 串接多個轉換器，相比一步一步手動處理，有什麼好處？
    請至少說出兩個優點。
A3：???
```

---

## 繳交 Checklist

- [ ] week03/q1_hello.txt 包含姓名學號與完成確認
- [ ] week03/q2_preprocessing.txt 包含完整程式碼與 Pipeline 處理結果
- [ ] week03/q3_column_transformer.txt 包含完整程式碼與 ColumnTransformer 輸出結果
- [ ] week03/q4_concept.txt 包含三題觀念回答
- [ ] 已建立 PR（首次繳交才需要，標題：學號_姓名）
- [ ] 已 push 到 Fork（確認 PR 中可看到本週 commit）

## 常見問題

**Q：第 1 題只要建立檔案就好嗎？**
是的，第 1 題的目的是確認你能跑通 Fork → Push → PR 的流程。填好姓名學號、完成確認，成功發出 PR 即通過。

**Q：SimpleImputer 的 strategy='mean' 只能用在數值型資料嗎？**
是的，mean 和 median 只能用在數值型欄位。類別型欄位請使用 most_frequent，它會用出現次數最多的值來填補。

**Q：OneHotEncoder 的 sparse_output 參數是什麼意思？**
設為 False 會輸出一般的 NumPy array，方便觀察結果。設為 True 會輸出稀疏矩陣，在欄位非常多的時候比較省記憶體。本次作業請設為 False。

**Q：ColumnTransformer 轉換後的資料怎麼知道每個欄位是什麼？**
可以用 `preprocessor.get_feature_names_out()` 取得轉換後的欄位名稱。如果版本不支援，也可以從 OneHotEncoder 的 `categories_` 屬性推算。

**Q：Pipeline 裡的步驟名稱可以自己取嗎？**
可以，步驟名稱是自訂的字串，用來識別管道中的每個步驟。例如 `('imputer', SimpleImputer())` 中的 'imputer' 就是自訂名稱。

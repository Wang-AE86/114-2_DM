# 資料探勘 — 小考 1

> 10 題選擇題，每題單選，每題 10 分。
> 範圍：Ch0 ~ Ch3（Python 基礎、Pandas DataFrame、資料預處理）
> 時間：15 分鐘

---

**1. 下列何者是正確的 Python 串列（list）操作？**

(A) `my_list = (1, 2, 3)` 建立一個串列
(B) `my_list.append(4)` 會在串列末尾新增元素 4
(C) `my_list[1] = 'a'` 會產生錯誤，因為串列不能混合型別
(D) `len(my_list)` 會回傳串列中最大的值

---

**2. 下列 Python 程式碼的輸出為何？**

```python
data = {'name': ['Alice', 'Bob'], 'score': [85, 92]}
df = pd.DataFrame(data)
print(df.shape)
```

(A) `(2,)`
(B) `(2, 2)`
(C) `(1, 2)`
(D) `(2, 1)`

---

**3. 在 Pandas 中， `df.iloc[0:3]` 會選取哪些列？**

(A) 第 0、1、2、3 列（共 4 列）
(B) 第 0、1、2 列（共 3 列）
(C) 第 1、2、3 列（共 3 列）
(D) 只有第 3 列

---

**4. 下列哪個方法可以偵測 DataFrame 中每個欄位的遺漏值數量？**

(A) `df.dropna()`
(B) `df.fillna(0)`
(C) `df.isnull().sum()`
(D) `df.describe()`

---

**5. `df.loc[df['age'] > 30]` 這行程式碼的作用是什麼？**

(A) 將 age 欄位中大於 30 的值改為 NaN
(B) 篩選出 age 大於 30 的所有列
(C) 計算 age 大於 30 的列數
(D) 刪除 age 大於 30 的列

---

**6. 在 scikit-learn 中， `StandardScaler` 的作用是什麼？**

(A) 將所有數值轉換為 0 或 1
(B) 將每個特徵轉換為平均值 = 0、標準差 = 1
(C) 將遺漏值填補為該欄位的平均值
(D) 將類別型資料轉換為數值

---

**7. `OneHotEncoder` 用來處理什麼類型的資料？**

(A) 連續型數值資料（如身高、體重）
(B) 遺漏值
(C) 類別型資料（如顏色：紅、藍、綠）
(D) 時間序列資料

---

**8. 下列關於 `Pipeline` 的敘述，何者正確？**

(A) Pipeline 只能包含一個步驟
(B) Pipeline 會依序執行所有步驟，前一步的輸出是下一步的輸入
(C) Pipeline 只能用於分類任務，不能用於迴歸
(D) Pipeline 中的步驟順序不影響結果

---

**9. 使用 `train_test_split(X, y, test_size=0.3, random_state=42)` 時，下列敘述何者正確？**

(A) 70% 的資料用來測試，30% 用來訓練
(B) 30% 的資料用來測試，70% 用來訓練
(C) `random_state=42` 表示每次切割的結果都會不同
(D) 切割後 X_train 和 X_test 的欄位數量會不同

---

**10. 下列程式碼中， `ColumnTransformer` 的作用是什麼？**

```python
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'salary']),
    ('cat', OneHotEncoder(), ['city'])
])
```

(A) 對所有欄位套用相同的轉換
(B) 對不同欄位套用不同的預處理方法
(C) 只保留 age 和 salary 欄位，刪除 city
(D) 將三個欄位合併為一個欄位

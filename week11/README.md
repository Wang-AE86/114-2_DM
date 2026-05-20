# 第 11 週｜組合預測器 + AI 輔助模型選擇

> 上課日期：2026/05/06（W11，週三 09:10–12:00） <BR>
> 對應教科書：Ch13 組合預測器 <BR>
> Sup Material [PPT](https://www.dropbox.com/scl/fi/up1dh03c80i2082y68slx/Boosting_adaptive_boosting_AdaBoost_2023_0601_R1.pptx?rlkey=kjcgbrpvmudp6u4lvnctp3k5v&dl=0) <BR>
> Colab 程式：[13 整合預測器 ensemble](https://colab.research.google.com/drive/1IXEwOqbz-zUPLKYYLxuHB_gpmva4K3IC)
> 小考 2：Ch4 ~ Ch13（迴歸、分類演算法、交叉驗證、網格搜尋、組合預測器）

---

## 學習目標

1. 理解「集成學習」的基本想法：多個弱模型如何組合成更強的預測器
2. 看懂並改寫 Bagging、隨機森林、AdaBoost、Gradient Boosting、XGBoost 五種主流集成方法
3. 能用 W10 學的 GridSearchCV 對集成模型做超參數調校
4. 學會用 AI 工具輔助「該選哪個集成模型」「參數怎麼設」的判斷
5. 理解集成模型相較單一模型的優缺點與適用情境

---

## 一、本週課程主軸

### 1. 為什麼需要組合預測器？

回顧前幾週的單一模型：KNN、SVM、決策樹各有強項與盲點。集成學習的核心想法是：

> 與其相信「一位專家的判斷」，不如綜合「多位專家的投票」。

| 比喻 | 對應集成方法 |
|---|---|
| 多位醫師同時看 X 光，多數決 | Bagging、Random Forest |
| 後一位醫師專看前面幾位看錯的病例 | AdaBoost、Gradient Boosting |
| 同上但更精緻、含正則化 | XGBoost |

### 2. Bagging（Bootstrap Aggregating）

- **想法**：對訓練資料做有放回抽樣，產生多份子訓練集，每份訓練一個模型，最後投票/平均
- **特點**：降低變異 variance、減少 overfitting
- **代表**：BaggingClassifier、Random Forest

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,        # 50 棵樹
    max_samples=0.8,        # 每棵看 80% 樣本
    random_state=42
)
bag.fit(X_train, y_train)
print("Bagging accuracy:", bag.score(X_test, y_test))
```

### 3. 隨機森林（Random Forest）

Bagging 的特化版：底層固定用決策樹，且每次分裂時**隨機選一部分特徵**，讓樹之間更不一樣。

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

rf = RandomForestClassifier(
    n_estimators=100,       # 100 棵樹
    max_depth=10,           # 每棵最深 10 層
    max_features="sqrt",    # 每次分裂看 sqrt(n_features) 個特徵
    random_state=42
)
rf.fit(X_train, y_train)

# 重要特徵排名，這是隨機森林一大優點
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
print(importances.sort_values(ascending=False).head(10))
```

### 4. AdaBoost（Adaptive Boosting）

- **想法**：第 1 棵樹預測錯的樣本，在第 2 棵樹被加重權重；逐輪修正
- **特點**：擅長處理複雜決策邊界，但對 noise 敏感

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
```

### 5. Gradient Boosting

- **想法**：每棵新樹學「前面所有樹的殘差」，沿著梯度方向逐步修正
- **特點**：表現通常優於 AdaBoost，但訓練較慢

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
```

### 6. XGBoost

工業界與 Kaggle 常勝軍。在 Gradient Boosting 基礎上加：
- 正則化以避免 overfitting
- 平行化讓速度更快
- 內建處理缺失值
- early stopping 機制

```python
# 安裝：!pip install xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)
```

---

## 二、五種模型對比速查表

| 模型 | 類型 | 主要優點 | 主要缺點 | 何時適用 |
|---|---|---|---|---|
| Bagging | 並行 | 降低 variance、不易 overfit | 對偏差問題改善有限 | 底模容易 overfit 時 |
| Random Forest | 並行 | 開箱即用、可看特徵重要度 | 樹太多時佔記憶體 | 表格資料首選 baseline |
| AdaBoost | 序列 | 簡單、對欠擬合有效 | 對 noise 敏感 | 資料乾淨時 |
| Gradient Boosting | 序列 | 表現穩定、可調 | 訓練慢、不易並行 | 中等規模、追求精度 |
| XGBoost | 序列 | 工業級速度與精度 | 參數多需調 | 大資料、Kaggle 競賽級 |

---

## 三、AI 輔助模型選擇（輕量介入）

> 本週的 AI 輔助練習維持「輕量」原則。對照 APP 開發班的完整介入，本班為對照組。

### 用 AI 工具做模型選型

當你面對新資料集，可以這樣問 AI：

> 我有一份表格資料，特徵 15 個（10 個數值 + 5 個類別），樣本數 5000，二元分類問題，類別比例約 7:3。請問 Random Forest、Gradient Boosting、XGBoost 哪個比較適合作為 baseline？為什麼？

AI 給的建議**不可直接照單全收**。請依以下步驟驗證：

1. 看 AI 給的理由是否符合課堂教過的原則
2. 在 Colab 中跑兩三個候選模型的初步結果
3. 比較交叉驗證分數，再做決定

### 同一 prompt 三家比對 mini demo

老師會在課堂上用同一個 prompt 詢問 ChatGPT / Claude / Gemini，請學生觀察：

| 觀察點 | 為什麼重要 |
|---|---|
| 三家給的模型推薦是否一致 | 不一致代表這個問題沒有絕對答案 |
| 三家的解釋深度 | 哪家的理由你比較聽得懂？ |
| 三家是否提到「先試 baseline 再優化」 | 這是業界標準做法，AI 應該要提到 |

---

## 四、本週重點觀念複習卡

| 觀念 | 一句話記憶 |
|---|---|
| Bagging vs Boosting | Bagging 並行多模型投票，Boosting 序列修正前一輪錯誤 |
| Random Forest 為什麼穩 | 每棵樹資料隨機 + 特徵隨機，多樣性高 |
| AdaBoost 加重權重 | 預測錯的樣本下一輪被「重點關照」 |
| Gradient Boosting 學殘差 | 新樹學的是「前面所有樹加起來的誤差」 |
| XGBoost 為什麼快 | 平行化 + 預排序 + 早停 |
| 集成一定比單一好嗎 | 不一定，資料簡單時單一模型可能更好且更可解釋 |

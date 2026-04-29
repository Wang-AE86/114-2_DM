# 第 10 週｜分類模板、交叉驗證與網格搜尋 + 三大 AI 工具速覽

> 對應教科書：Ch10 分類模板、Ch11 交叉驗證、Ch12 網格搜尋
> 進度：期中考後第一週，本週起在課堂建立**三大 AI 工具**（ChatGPT / Claude / Gemini）的基本認識
> 進階：「怎麼用 AI 才不會被 AI 帶歪」見 [補充教材：AI-RED 五階段框架](../extras/ai-red-framework.md)（學期後半段回來深入）

---

## 學習目標

1. 看懂並改寫分類預測的標準流程（資料前處理 → 切分 → 模型訓練 → 預測 → 評估）
2. 用 K-Fold 交叉驗證評估模型穩定性，避免單次切分誤差
3. 用 `GridSearchCV` 自動找最佳超參數組合
4. **認識 ChatGPT / Claude / Gemini 三家旗艦 AI 工具的差異，能依任務挑選合適的工具**

---

## 一、本週課程主軸（Ch10–Ch12）

### 1. 分類預測模板（Ch10）

![分類預測通用模板（7 大標準化階段）](images/classification_template.png)

> 上圖：本週要建立的「分類預測通用模板」概覽——七大標準化階段（資料載入 → 特徵與目標分離 → Train/Test 切分 → 自動化編碼 → Pipeline 預處理 → 多模型批次訓練 → 結果評估比較），以及 Logistic Regression / SVC / KNN / RandomForest 四模型對比的整體輪廓。

把前幾週學的分類器（KNN、SVM、決策樹）整理成共用的預測模板：

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Colab：[10 分類預測模版](https://colab.research.google.com/drive/1OqudZ0PDJ3YaUQPiOwilPG9vcCX2O9jt)

### 2. K-Fold 交叉驗證（Ch11）

單次 train/test 切分容易因運氣好壞影響結果，**K-Fold 把資料切 K 份輪流當測試集**，得到 K 個分數，取平均更穩定。

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_s, y, cv=5, scoring='accuracy')
print(f"5-Fold mean = {scores.mean():.3f}, std = {scores.std():.3f}")
```

Colab：[11 交叉驗證](https://colab.research.google.com/drive/1YvHf8e4V5-OFlAClYlfgaE6xBJRvxNvo?usp=sharing)

### 3. GridSearchCV 網格搜尋（Ch12）

超參數（如 KNN 的 `n_neighbors`、SVM 的 `C`/`gamma`）會大幅影響表現，`GridSearchCV` **自動枚舉所有組合 + 交叉驗證 + 給最佳組合**。

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_s, y)
print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)
```

Colab：[12 模型參數挑選和網格搜尋](https://colab.research.google.com/drive/1o-I1M7RAbANMsawstOypcshUuDmNBaQ2?usp=sharing)

---

## 二、三大 AI 工具速覽：ChatGPT / Claude / Gemini

> 課程後半段三家工具會頻繁在課堂與作業中出現。先建立基本認識，再決定何時用哪一家。

### 為什麼要認識三家？

不同任務、不同價格策略、不同模態能力——**用錯工具**會降低學習效率，甚至卡住整個作業。本週先做工具地圖，後續再談「怎麼用才不會被 AI 帶歪」（→ 補充教材）。

### 三家對照表

| 項目 | **ChatGPT**（OpenAI）| **Claude**（Anthropic）| **Gemini**（Google）|
|---|---|---|---|
| 旗艦模型 | GPT-5 | Claude Opus 4.7 / Sonnet 4.6 | Gemini 2.5 Pro |
| 免費版 | 有限額度（GPT-5 mini）| 有限額度（Sonnet 4.6）| 額度大方（2.5 Flash）|
| 強項 | 通用對話、生態系最廣 | 程式撰寫、長文閱讀、推理 | 多模態（圖像/影片）、Google 服務整合 |
| 弱項 | 數學易錯、會幻覺 | 預設無上網（要開 Web Search 工具）| 中文偶爾跳英文、上下文一致性較弱 |
| IDE 整合 | Copilot / Cursor | Cursor / Windsurf / Claude Code | Gemini Code Assist |
| 課程適用場景 | 通用查詢、概念解釋 | sklearn code 解釋、文件閱讀 | Colab 整合查詢、視覺化建議 |
| 取用方式 | chat.openai.com | claude.ai | gemini.google.com |

### 直覺判斷流程

| 任務 | 第一選擇 | 理由 |
|---|---|---|
| 寫 / 改 sklearn code | **Claude** | 程式碼解釋穩定、長文不掉細節 |
| 看截圖、影片、圖像 | **Gemini** | 多模態能力強、Google 服務整合 |
| 一般概念解釋、不確定哪個好 | **ChatGPT** | 生態系最廣、模型熟悉度高 |

### 課堂提示

| # | 提示 |
|---|---|
| 1 | 課堂上老師會用 **Claude.ai** 做 sklearn demo，因為程式碼解釋最穩定 |
| 2 | 期末做題建議**同一 prompt 在三家比對**，挑出最合理的回答 |
| 3 | ⚠️ 不要把作業整題貼給 AI 讓它直接回答——違反學術倫理且學不到東西 |
| 4 | ⚠️ AI 說的具體事實（函式名、參數、論文 DOI）**都要回原始來源驗證**（→ 補充教材會詳述） |

> **進階主題**：「AI 什麼時候會胡說？怎麼系統化驗證？」
> 見 [補充教材：AI-RED 五階段框架](../extras/ai-red-framework.md)（學期後半段回來深入）

---

## 三、本週課堂演練（20 min · 同一 prompt 三家比對）

老師會在課堂上做一次三家 AI 比對 demo：

1. **同一個 prompt** 同時打到 ChatGPT / Claude / Gemini：
   ```
   寫一個 KNN 分類 + GridSearchCV 找最佳 n_neighbors 的 sklearn code，
   用 iris 資料集，並解釋為什麼要用 StandardScaler
   ```
2. 觀察三家回答：
   - 哪家解釋最清楚？
   - 哪家 code 最完整（有 import 嗎？有資料切分嗎？）？
   - 哪家解釋了 StandardScaler 的必要性？
3. 全班討論：以後寫 sklearn code 你會優先選哪一家？為什麼？

→ Demo 對話會貼到本週 issue 作為作業範例

---

## 四、課後作業（4 題 · 三家 AI 工具實戰比對）

繳交方式：fork 114-2_DM repo → 在你個人 fork 的 `homework/week10/` 建一個 markdown, W10_homework.md，回答以下 4 題，提交 Pull Request

| # | 題目 |
|---|---|
| 1 | 你目前最常用哪一家 AI（ChatGPT / Claude / Gemini）？為什麼？（100 字） |
| 2 | 用同一個 prompt（自己設計，與本週 Ch10/11/12 內容相關）打到三家 AI，把三份回答完整貼上 |
| 3 | 比較三家回答：哪家最準確？哪家最易讀？哪家有錯？指出**具體**不同點（至少 3 點） |
| 4 | 做完這次比對後，未來會調整你選擇 AI 的策略嗎？怎麼調整？（100 字內） |

繳交期限：下週上課前

---

## 五、本週重點觀念複習卡

| 觀念 | 一句話記憶 |
|---|---|
| 分類模板 | `train_test_split → fit → predict → evaluate` 五步驟 |
| K-Fold 交叉驗證 | 切 K 份輪流當測試集，取平均分數比較穩 |
| GridSearchCV | 暴力枚舉所有超參數組合 + 交叉驗證 = 最佳組合 |
| 三家 AI 工具 | ChatGPT 廣 / Claude 寫 code 強 / Gemini 多模態 |
| 用 AI 的最低限度 | 同一 prompt 三家比對，挑最合理的回答 |

---

## 六、延伸學習（補充教材）

- 📖 [AI-RED 五階段框架——AI 使用的進階檢核](../extras/ai-red-framework.md)
  > 「怎麼用 AI 才不會被 AI 帶歪」的系統化框架。本週**不在課堂講**，但建議自行讀過一次，建立批判使用 AI 的習慣。學期後半段會回來做一次完整實作。

---

*下週起進入組合預測器（Ch13），請先熟悉本週三個 Colab。*

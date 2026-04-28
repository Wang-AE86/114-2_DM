# 第 10 週｜分類模板、交叉驗證與網格搜尋 + AI-RED 框架介紹

> 對應教科書：Ch10 分類模板、Ch11 交叉驗證、Ch12 網格搜尋
> 進度：期中考後第一週，本週起導入 **AI-RED 框架**作為 AI 輔助學習的核心工具

---

## 學習目標

1. 看懂並改寫分類預測的標準流程（資料前處理 → 切分 → 模型訓練 → 預測 → 評估）
2. 用 K-Fold 交叉驗證評估模型穩定性，避免單次切分誤差
3. 用 `GridSearchCV` 自動找最佳超參數組合
4. **認識 AI-RED 五階段框架，並在本週作業中各完成一次練習**

---

## 一、本週課程主軸（Ch10–Ch12）

### 1. 分類預測模板（Ch10）

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

## 二、AI-RED 框架介紹

> 出處：**陳育詮（松山高中）**「Build AI Critical Radar with AI-RED Framework」（數位時代）。
> 本課程將此框架從高中端**操作化**並導入大學資料探勘課程，**本週為首次實施**。

研究顯示（Shen & Tamkin 2026），把 AI 當「直接寫答案的工具」會損害學習成效；當作「幫你思考的對話夥伴」才能保留學習。**AI-RED 是把這個原則變成可執行檢核的具體框架**。

### AI-RED 五階段速覽卡（A4 一頁）

| 階段 | 全名 | 關鍵動作 | 對抗的壞習慣 |
|---|---|---|---|
| **A** | **A**scribe（標示出處）| 在貼上 AI 內容前，先寫「我用了 ___（AI 名稱+版本），任務是 ___」 | 完全委託、把 AI 答案當自己的 |
| **I** | **I**nquire（概念提問）| 不直接要程式碼，先問「為什麼這樣寫」「如果改成 X 會怎樣」 | 只要答案、不要理解 |
| **R** | **R**eference（查證對抗幻覺）| AI 給的函式/參數/論文/資料源 → **去官方文件或 sklearn API doc 驗證** | 相信 AI 的所有輸出 |
| **E** | **E**valuate（先理解再使用）| 用自己一句話重述 AI 的解釋，再決定要不要用這段 code | 逐字抄寫、盲目修改 |
| **D** | **D**ocument（記錄對話）| 把 prompt + AI 回應 + 你的判斷貼進作業 markdown | 對話用完就丟 |

### 一個踩雷的真實案例（R 階段為什麼必要）

老師我自己在寫 SAR 海洋油污論文時，請 AI 幫忙整理 50 篇文獻的 DOI（Digital Object Identifier，論文唯一識別碼）。AI 給了一份漂亮的清單。**結果其中 14 篇 DOI 是錯的，錯誤率 28%**——AI 「幻覺」(hallucination) 出了**根本不存在的 DOI**。如果直接送投稿期刊，整篇論文會被退稿。

修正方法是逐筆去 CrossRef API 查證（這就是 R = Reference 的精神）。本學期後半段會教你們用 Python 自動做這件事。但**這週起，請先把「AI 給的任何東西都要去原始來源驗證」當成肌肉記憶**。

---

## 三、本週課堂演練（30 min AI-RED demo）

老師會在課堂上做一次完整 demo：
1. **故意**用 ChatGPT 寫一段 sklearn 分類流程的 code（含 3 處小錯）
2. 全班用 **R**（查 sklearn 官方文件）和 **E**（自己讀過再決定要不要用）抓出 3 處錯
3. 對照 **A**（標示來源）和 **D**（紀錄對話）的標準寫法

→ 這份 demo 的對話會放在本週作業 issue，作為 A/D 階段的範例。

---

## 四、課後作業（5 題對應 A/I/R/E/D 各一階段）

繳交方式：fork 114-2_DM repo → 在你個人 fork 的 `homework/week10/` 建一個 markdown，回答以下 5 題，提交 Pull Request

| # | 階段 | 題目 |
|---|---|---|
| 1 | **A**scribe | 寫一段 100 字的「AI 使用聲明」：你本週做作業會用哪個 AI（ChatGPT / Claude / Copilot / 其他）？哪個版本？你打算讓它幫你做什麼、不讓它做什麼？ |
| 2 | **I**nquire | 不問「給我一個 GridSearchCV 範例」，改問「為什麼 GridSearchCV 預設用 5-fold？fold 數對結果的影響是什麼？」把你的問題、AI 的回答、你的後續追問貼上 |
| 3 | **R**eference | 對照上面 AI 的回答，到 [sklearn 官方文件](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 查證「`cv` 參數的預設值是什麼」「文件怎麼描述 fold 數的選擇」。**有沒有發現 AI 寫錯或不精確的地方？指出來** |
| 4 | **E**valuate | 用你**自己的話**重述「為什麼要用交叉驗證而不是單次 train/test split」（最多 150 字、不要貼 AI 的話）。然後判斷：你會把上面 AI 給的 code 直接貼進你的作業嗎？為什麼會 / 為什麼不會？ |
| 5 | **D**ocument | 把第 1–4 題的所有 AI 對話**完整截圖**或貼成 markdown 程式碼區塊，標明時間順序，作為附件 |

繳交期限：下週上課前

---

## 五、本週重點觀念複習卡

| 觀念 | 一句話記憶 |
|---|---|
| 分類模板 | `train_test_split → fit → predict → evaluate` 五步驟 |
| K-Fold 交叉驗證 | 切 K 份輪流當測試集，取平均分數比較穩 |
| GridSearchCV | 暴力枚舉所有超參數組合 + 交叉驗證 = 最佳組合 |
| AI-RED | **A**scribe / **I**nquire / **R**eference / **E**valuate / **D**ocument |
| 為什麼要 AI-RED | 沒有它 → 學了 AI 反而學不到課業；有它 → AI 幫你思考而不是替你思考 |

---

## 六、AI-RED 卡（可下載印出貼桌前）

```
┌────────────────────────────────────────────────┐
│  AI-RED  使用 AI 的五個自我檢核                │
├────────────────────────────────────────────────┤
│  A  Ascribe   我用了哪個 AI？任務是什麼？      │
│  I  Inquire   我有先問「為什麼」嗎？           │
│  R  Reference 我有去原始來源查證嗎？           │
│  E  Evaluate  我能用自己的話重述嗎？           │
│  D  Document  我有記錄對話過程嗎？             │
└────────────────────────────────────────────────┘
```

---

*本週是課程後半的轉折點，從這週起每次作業都會要求你跑過一輪 AI-RED。下週起進入組合預測器（Ch13），請先熟悉本週三個 Colab。*

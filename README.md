# 114-2 資料探勘

## 目錄

- [課程資訊](#課程資訊)
- [課程說明](#課程說明)
- [作業繳交方式](#作業繳交方式)
- [課程進度表](#課程進度表)
- [教科書章節與週次對應表](#教科書章節與週次對應表)
- [小考說明](#小考說明)
- [期末專題](#期末專題)
- [課程工具對應表](#課程工具對應表)
- [評分比例](#評分比例)
- [Git 基本流程](#git-基本流程)
- [Git 基本指令](#git-基本指令)
- [作業資料夾結構](#作業資料夾結構)
- [課程軟體與環境](#課程軟體與環境)
- [注意事項](#注意事項)

---

## 課程資訊

| 項目 | 說明 |
|------|------|
| 課程名稱 | 資料探勘 |
| 課程代碼 | 0644301 |
| 學期 | 114 學年度第 2 學期 |
| 授課對象 | 大二以上 |
| 每週上課 | 3 小時 |
| 教科書 | 機器學習課程教學大綱 Ch0 ~ Ch21 |
| 課程 Colab 程式 | [Colab 人工智慧二版](https://sites.google.com/view/oneline-python/colab%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E4%BA%8C%E7%89%88?authuser=0) |
| 實作環境 | Google Colab、Python + Anaconda、GitHub、VS Code |

## 課程說明

本課程涵蓋資料探勘的核心理論與實作技術，從 Python 基礎與 Pandas 資料處理出發，循序漸進涵蓋資料預處理、迴歸分析、分類預測、模型評估與優化、組合預測器、不均衡資料處理、文字探勘、集群分析，以及深度學習入門。全程搭配 Python 生態系工具與實務專案進行實作。課程後半導入 AI 輔助開發流程，培養學生運用 AI 工具加速資料分析與模型建構的能力。

### 課程核心主題

**Python 資料科學基礎**：使用 Anaconda 建立 Python 開發環境，熟悉 Pandas DataFrame 操作、資料定位、篩選、遺漏值處理等核心技能。

**資料前處理與管道器**：學習數值型與類別型資料的預處理方法，包含 SimpleImputer、StandardScaler、OneHotEncoder 等轉換器，並使用 Pipeline 與 ColumnTransformer 建構可重複使用的前處理流程。

**監督式學習：迴歸與分類**：實作線性迴歸、多元迴歸、羅吉斯迴歸、KNN、SVM、決策樹等演算法。學習混淆矩陣、ROC 曲線、交叉驗證、GridSearchCV 等模型評估與調校方法。

**組合預測器與實務專案**：實作投票組合器、隨機森林、AdaBoost、Gradient Boosting、XGBoost 等集成學習方法。透過員工流失率預測、客戶流失率預測、信用偵測三個實務專案，練習完整的建模流程與不均衡資料處理。

**文字探勘**：掌握英文與中文文字的處理技術，包含詞袋模型、TF-IDF、LDA 主題探索、jieba 中文斷詞與文字雲視覺化。

**非監督式學習與深度學習入門**：實作 K-Means 集群分析與手肘法，使用 Keras 建構深度學習模型進行手寫數字辨識。

**AI 工具輔助開發**：從第 10 週起導入 AI 工具於資料分析與模型開發流程中。學習 Prompt Engineering 技巧，掌握如何對 ChatGPT、Copilot 等 AI 工具下達有效指令，輔助資料清洗、程式碼產生、模型選擇與結果解讀，同時培養驗證 AI 產出正確性的能力。核心原則為「先理解再使用」，確保學生具備獨立判斷與修正 AI 產出的能力。

## 作業繳交方式

本課程採用 **Fork + Pull Request** 流程繳交作業：

1. Fork 本 Repo 到你的 GitHub 帳號
2. 在你的 Fork 中完成每週作業
3. 發 Pull Request 回本 Repo 繳交

每週作業放在對應的資料夾中，例如 `week01/`、`week02/`。

---

## 課程進度表

### 第 1 週｜課程導論與開發環境建置

- 對應教科書：Ch0 機器學習介紹、Ch1 Python 基本功能介紹
- Colab 程式：[1 AI-python](https://colab.research.google.com/drive/1OD6BU0CpZ27ixsG2OsA2PHWAF6IfFTVg?usp=sharing)
- 教學內容：
  - 課程大綱、評分方式、作業繳交流程說明
  - 人工智慧、機器學習、深度學習的定義與差異
  - 訓練集與測試集的觀念建立
  - 註冊 GitHub 帳號、Git 基本操作：clone、add、commit、push
  - Anaconda 安裝與環境設定、Google Colab 入門操作
  - Python 基礎語法複習：變數、函數、串列、迴圈、字典

### 第 2 週｜Pandas DataFrame 與資料操作

- 對應教科書：Ch2 Pandas DataFrame 介紹
- Colab 程式：[2 Pandas DataFrame 介紹](https://colab.research.google.com/drive/1-Etp0nO0UANTN0Ynd0j9HiZcuM3R5LlB?usp=sharing)
- 教學內容：
  - 創立 DataFrame：字典方式建立、列索引鍵設定
  - DataFrame 重要屬性、Series 與 DataFrame 差異
  - NaN 遺漏值偵測、loc / iloc 定位
  - axis 觀念、布林條件篩選、apply() 函數
  - **Fork + PR 作業繳交流程教學**

### 第 3 週｜資料預處理

- 對應教科書：Ch3 資料預處理
- Colab 程式：[3 資料預處理](https://colab.research.google.com/drive/12jJkVB2cxrVVw8JyNYpP3eL9q5pVi2NI?usp=sharing)
- 教學內容：
  - 數值型預處理：SimpleImputer、StandardScaler
  - 類別型預處理：OneHotEncoder
  - Pipeline、ColumnTransformer、KBinsDiscretizer
  - **專題分組、組長建立專題 Repo、公布題目參考清單**

### 第 4 週｜協同教學：海洋測深與水深資料分析

- 授課講師：林姿廷、陳秋廷（3/18）
- 教學內容：海洋測深儀器介紹、水深資料介紹及分析實作
- **個人題目探索：每人提出 1-3 個題目構想**
- 作業：上課心得

### 第 5 週｜協同教學：潮汐、聲速與 GNSS 資料分析

- 授課講師：林姿廷、陳秋廷（3/25）
- 教學內容：潮汐資料下載及分析實作、聲速資料蒐集及分析實作、GNSS 資料介紹
- **組內討論投票決定題目、繳交專題提案**
- 作業：上課心得

### 第 6 週｜協同教學：用科技探索海底世界

- 授課講師：盧翊維（4/1）
- 教學內容：人類如何看見海底？用科技探索看不見的海底世界
- 作業：上課心得

### 第 7 週｜放假（清明連假補假）

- 原定內容（迴歸分析 + 小考 1）順延至第 8 週

### 第 8 週｜迴歸分析：線性迴歸與羅吉斯迴歸 + **小考 1**

- 對應教科書：Ch4 簡單線性迴歸、Ch5 多元迴歸、Ch6 羅吉斯迴歸
- Colab 程式：[4 簡單線性迴歸](https://colab.research.google.com/drive/18nr_4pOkjzvs-5uJoPWfKW5cU97KH0qP?usp=sharing)、[5 多元線性迴歸](https://colab.research.google.com/drive/1iXifxI-AoJlkVMravR5rXP_2pDn8yYtS?usp=sharing)、[6 羅吉斯迴歸](https://colab.research.google.com/drive/1gaq4UpvRb4CXUrgt_k6lJhMUAsakd1Ly?usp=sharing)
- 教學內容：
  - LinearRegression、PolynomialFeatures、多元迴歸
  - 羅吉斯迴歸、混淆矩陣、ROC 曲線與 AUC
- **小考 1**：Ch0 ~ Ch3（Python 語法、Pandas 操作、資料預處理）10 題選擇題

### 第 9 週｜期中考

- 考試範圍：第 1 至 8 週，涵蓋 Ch0 ~ Ch9
- 考試形式：筆試 + Colab 實作
- 協同教學的海洋測量相關內容不列入期中考範圍

### 第 10 週｜分類模板、交叉驗證與網格搜尋 + Prompt Engineering 入門

- 對應教科書：Ch10 分類模板、Ch11 交叉驗證、Ch12 網格搜尋
- Colab 程式：[10 分類預測模版](https://colab.research.google.com/drive/1OqudZ0PDJ3YaUQPiOwilPG9vcCX2O9jt)、[11 交叉驗證](https://colab.research.google.com/drive/1YvHf8e4V5-OFlAClYlfgaE6xBJRvxNvo?usp=sharing)、[12 模型參數挑選和網格搜尋](https://colab.research.google.com/drive/1o-I1M7RAbANMsawstOypcshUuDmNBaQ2?usp=sharing)
- 教學內容：
  - 分類預測模板、K-Fold 交叉驗證、GridSearchCV
  - **Prompt Engineering 入門**：好 Prompt 的五個要素
  - AI 輔助開發三大原則

### 第 11 週｜組合預測器 + AI 輔助 + **小考 2**

- 對應教科書：Ch13 組合預測器
- Colab 程式：[13 整合預測器 ensemble](https://colab.research.google.com/drive/1iUh3CxSCpC2AvcsxSpwLI40bRHIcp8gK?usp=sharing)
- 教學內容：
  - Bagging、隨機森林、AdaBoost、Gradient Boosting、XGBoost
  - AI 輔助模型選擇與調校

### 第 12 週｜實務專案：員工流失率預測與不均衡資料處理

- 對應教科書：Ch14 員工流失率預測、Ch15 客戶流失率預測、Ch16 信用偵測
- Colab 程式：[14 Employee Churn](https://colab.research.google.com/drive/17SBqnkGvdgzrs5SiQrtmPPJjbA4CDyzM?usp=sharing)、[15 Telcom Customer Churn](https://colab.research.google.com/drive/1JbjxMhG2G3YsAfuYeOoaXr7V-TN1GpXI?usp=sharing)、[16 Credit Card](https://colab.research.google.com/drive/1jiHDRH4iymBd70dJUcOCj9GsuwQqrtOM?usp=sharing)
- 教學內容：
  - 完整建模流程、class_weight、向下取樣、向上取樣
  - AI 輔助不均衡資料處理

### 第 13 週｜文字處理與商品評論分析

- 對應教科書：Ch17 文字處理、Ch18 Amazon 商品評論分析
- Colab 程式：[17 NewsGroup](https://colab.research.google.com/drive/1L8HjJQEZC02gYgh8eUz-rgDJvwe4ewaM?usp=sharing)、[18 Amazon 商品評論分析](https://colab.research.google.com/drive/1ke8WmJjgI2l4S3fJF612tNK8YOiGkuzG?usp=sharing)
- 教學內容：
  - 詞袋模型、n-gram、TF-IDF、LDA 主題探索
  - Amazon 評論分析、Naive Bayes

### 第 14 週｜中文文字處理與 K-Means 集群分析 + **小考 3**

- 對應教科書：Ch19 中文文字處理、Ch20 K-means 集群分析
- Colab 程式：[19 中文文字處理](https://colab.research.google.com/drive/1Srjshd1h18g3jvChUaQ5YHjC0k7sjDsO?usp=sharing)、[20 Kmeans](https://colab.research.google.com/drive/1V3CgZazr9Le0hZYAUEwsXf7maHxAok0H?usp=sharing)
- 教學內容：
  - jieba 中文斷詞、文字雲
  - K-Means、Inertia、手肘法

### 第 15 週｜Keras 深度學習入門 + **小考 4**

- 對應教科書：Ch21 Keras 深度學習
- Colab 程式：[21 Keras 深度學習](https://colab.research.google.com/drive/18-yVk7MjXC7Cde971SC8FrOCngIfP9R7?usp=sharing)
- 教學內容：
  - 手寫數字辨識、Keras 架構、KerasClassifier
  - 傳統機器學習 vs 深度學習比較

### 第 16 週｜專題開發與 AI 輔助完整分析

- AI 輔助完整資料探勘分析流程
- 專題開發時間、教師個別諮詢

### 第 17 週｜專題整合與報告準備

- 分析報告撰寫、專題整合測試、繳交期末報告與投影片

### 第 18 週｜期末專題發表

- 口頭報告：每組 15 分鐘 + 5 分鐘 Q&A
- 組間互評

---

## 教科書章節與週次對應表

| 週次 | 教科書章節 | 主題 |
|------|-----------|------|
| 第 1 週 | Ch0、Ch1 | 機器學習介紹、Python 基本功能 |
| 第 2 週 | Ch2 | Pandas DataFrame 介紹 |
| 第 3 週 | Ch3 | 資料預處理 |
| 第 4 週 | 協同教學 | 海洋測深儀器、水深資料分析 |
| 第 5 週 | 協同教學 | 潮汐、聲速、GNSS 資料分析 |
| 第 6 週 | 協同教學 | 用科技探索海底世界 |
| 第 7 週 | Ch4、Ch5、Ch6 | 線性迴歸、多元迴歸、羅吉斯迴歸 |
| 第 8 週 | Ch7、Ch8、Ch9 | KNN、SVM、決策樹 |
| 第 9 週 | 期中考 | Ch0 ~ Ch9 |
| 第 10 週 | Ch10、Ch11、Ch12 | 分類模板、交叉驗證、網格搜尋 + Prompt Engineering |
| 第 11 週 | Ch13 | 組合預測器 + AI 輔助 |
| 第 12 週 | Ch14、Ch15、Ch16 | 實務專案、不均衡資料處理 + AI 輔助 |
| 第 13 週 | Ch17、Ch18 | 文字處理、商品評論分析 |
| 第 14 週 | Ch19、Ch20 | 中文文字處理、K-Means 集群分析 |
| 第 15 週 | Ch21 | Keras 深度學習 |
| 第 16 週 | 專題開發 | AI 輔助完整分析、專題整合 |
| 第 17 週 | 專題整合 | 報告撰寫與準備 |
| 第 18 週 | 期末發表 | 口頭報告 + Demo |

---

## 小考說明

| 次數 | 週次 | 範圍 | 涵蓋章節 |
|------|------|------|---------|
| 小考 1 | 第 8 週（原第 7 週放假順延） | Python 語法、Pandas 操作、資料預處理 | Ch0 ~ Ch3 |
| 小考 2 | 第 11 週 | 迴歸、分類演算法、交叉驗證、網格搜尋、組合預測器 | Ch4 ~ Ch13 |
| 小考 3 | 第 14 週 | 不均衡資料處理、文字處理、K-Means | Ch14 ~ Ch20 |
| 小考 4 | 第 15 週 | K-Means、深度學習 | Ch20 ~ Ch21 |

---

## 期末專題

### 專題說明

本課程包含分組期末專題，要求運用資料探勘技術，針對一份與**海事或海洋**相關的資料集，完成完整的探勘分析流程。專題須包含以下環節：資料收集與描述、資料前處理與特徵工程、至少兩種探勘演算法的實作與比較、模型評估、結果視覺化與價值解讀。

### 專題題目方向參考

以下為建議的專題題目方向，各組可從中選擇或自行提出相關主題：

**方向一：船舶交通與航運分析**

- **港口船舶進出模式分群分析**：利用 AIS 或港口統計資料，分析不同港口的船舶進出時間模式，使用分群演算法識別繁忙型、季節波動型與穩定型港口。建議技術：K-Means、時間序列特徵萃取
- **航運貨物量預測**：根據歷史貨物吞吐量，結合季節因子與經濟指標，建立迴歸預測模型。建議技術：線性迴歸、XGBoost、交叉驗證
- **船舶類型自動分類**：依據噸位、長度、寬度、吃水深度、航速等特徵，訓練分類模型辨識船舶類型。建議技術：決策樹、隨機森林、SVM
- **航運事故風險因子探勘**：分析天候、船齡、航線等因素對海事事故的影響，找出最具影響力的風險因子。建議技術：隨機森林特徵重要性、羅吉斯迴歸

**方向二：海洋環境與生態**

- **海水溫度變化趨勢分析與異常偵測**：分析台灣周邊海域海水表面溫度的長期趨勢，建立異常偵測模型。建議技術：迴歸預測、異常值偵測
- **海洋廢棄物分布模式分群**：利用淨灘調查資料分析不同海岸的廢棄物組成與數量模式，識別熱點區域。建議技術：K-Means、DBSCAN
- **珊瑚礁白化風險因子分析**：結合水溫、酸鹼值、溶氧等環境因子，探勘影響珊瑚白化的關鍵因子。建議技術：隨機森林、PCA
- **海洋生物多樣性調查分析**：分析不同海域的物種豐富度與群落結構，識別具有相似生態特性的海域群組。建議技術：階層式分群、t-SNE

**方向三：漁業資料探勘**

- **漁獲量預測模型**：結合季節、海溫、洋流等環境因子，建立漁獲量預測模型。建議技術：多元迴歸、XGBoost
- **漁場空間分群分析**：分析高產漁場的空間分布特徵，識別漁場群聚區域。建議技術：DBSCAN、K-Means
- **漁船作業型態分類**：從航跡資料萃取速度、轉向角度、停留時間等特徵，辨識不同作業狀態。建議技術：KNN、隨機森林、SVM
- **水產養殖水質預警模型**：利用感測器資料建立水質異常偵測與預警模型。建議技術：異常值偵測、決策樹

**方向四：港口營運與物流**

- **港口貨櫃裝卸效率分析**：識別影響作業效率的關鍵因素，找出效率瓶頸。建議技術：K-Means、迴歸分析
- **船舶到港時間預測**：結合 AIS 軌跡、氣象、航線距離等因子建立預測模型。建議技術：XGBoost、特徵工程
- **貨物與航線關聯分析**：探勘哪些貨物經常同時出現在特定航線上。建議技術：頻繁項目集分析
- **港口能源消耗模式分析**：識別不同消耗模式，提出節能建議。建議技術：分群分析、迴歸預測

**方向五：氣象海象與災害**

- **颱風路徑分群與影響分析**：將颱風依路徑模式分類，分析不同群組對台灣各海域的影響。建議技術：K-Means、軌跡特徵萃取
- **波浪高度預測**：結合風速、風向、氣壓等因子建立預測模型。建議技術：線性迴歸、XGBoost
- **海岸侵蝕風險因子探勘**：分析影響海岸侵蝕的關鍵因子，建立風險分類模型。建議技術：隨機森林、羅吉斯迴歸
- **海上搜救資源部署分析**：利用歷史事故紀錄識別熱點區域與集中時段。建議技術：DBSCAN、時間模式分析

**方向六：文字探勘與海事資訊**

- **海事新聞情感分析與主題分類**：蒐集海事新聞，完成主題分類或情感分析。建議技術：TF-IDF、Naive Bayes、文字雲
- **海事法規文本分析**：探索法規內容的重點領域。建議技術：詞頻統計、LDA 主題模型

### 建議公開資料來源

| 資料來源 | 網址 | 可取得資料 |
|---------|------|----------|
| 政府資料開放平臺 | data.gov.tw | 港口統計、漁業資料、海洋觀測 |
| 交通部航港局 | www.motcmpb.gov.tw | 商港營運統計、船舶進出港資料 |
| 海洋委員會海洋保育署 | www.oca.gov.tw | 海洋廢棄物調查、海洋生態資料 |
| 中央氣象署 | www.cwa.gov.tw | 波浪、潮汐、海溫觀測資料 |
| 漁業署 | www.fa.gov.tw | 漁獲統計、養殖漁業資料 |
| MarineTraffic | www.marinetraffic.com | 船舶 AIS 軌跡資料 |
| NOAA | www.noaa.gov | 全球海洋環境觀測資料 |
| Copernicus Marine | marine.copernicus.eu | 歐洲海洋環境資料 |
| Kaggle | www.kaggle.com | 各類海洋與航運相關資料集 |

### 專題時程

| 週次 | 階段 | 說明 |
|------|------|------|
| 第 3 週 | 分組與建立 Repo | 公布分組名單和題目參考清單，組長從專題模板建立小組 Repo，邀請組員、老師、助教為 Collaborator |
| 第 4 週 | 個人題目探索 | 每人在專題 Repo 的 `my-topics/` 提出 1-3 個題目構想 |
| 第 5 週 | 決定題目與提案 | 各組從所有成員的題目中討論、投票，選出小組題目，組長繳交 proposal/proposal.md |
| 第 6 週 | 提案審查 | 教師審查提案，提供修改建議 |
| 第 7-8 週 | 前期研究 | 資料來源確認、初步資料收集與探索 |
| 第 10-12 週 | 模型開發 | 資料清洗、特徵工程、模型訓練與調校 |
| 第 13-15 週 | 進階分析 | 文字探勘、集群分析、結果視覺化 |
| 第 16 週 | 整合測試 | 確保分析流程完整、修正問題 |
| 第 17 週 | 準備發表 | 繳交期末報告和投影片 |
| 第 18 週 | 期末發表 | 口頭報告 + Demo |

### 專題 Repo 建立方式

1. 前往專題模板：`https://github.com/pychang-ai/114-2_DM-project-template`
2. 點選「**Use this template**」>「**Create a new repository**」
3. Repository name 填入 `114-2_DM-G01`（替換為組別編號）
4. 設為 **Public**，點選 **Create repository**
5. 進入 Settings > Collaborators > Add people，依序加入：
   - 所有組員（權限：Write）
   - 老師 `pychang-ai`（權限：Write）
   - 助教帳號（權限：Write）
6. 被邀請的人需到 GitHub 通知中點選「**Accept invitation**」才能存取

### 專題繳交項目

| 項目 | 說明 |
|------|------|
| proposal/proposal.md | 專題提案：題目、動機、資料來源、預計使用方法 |
| data/ | 資料集與資料說明 |
| notebooks/ | Jupyter Notebook 分析過程 |
| src/ | Python 程式碼 |
| docs/report.md | 期末報告 |
| docs/slides/ | 發表投影片 |
| requirements.txt | Python 套件清單 |
| README.md | 專題說明與執行方式 |

### 專題評分標準

專題採用四維評分架構，綜合教師、同學與 AI 的多角度評量：

#### 評分維度總覽

| 維度 | 評量對象 | 配分 | 評分方式 |
|------|---------|------|---------|
| 教師評分 | 整組 | 40% | 教師依下方細項給分 |
| 組間互評 | 整組 | 20% | 期末發表時各組互評 |
| 組內互評 | 個人 | 20% | Google 表單匿名評分 |
| AI 評分 | 個人 | 20% | Python 腳本分析 GitHub 數據 |

#### 教師評分細項

| 項目 | 配分 | 說明 |
|------|------|------|
| 專題提案 | 5% | 題目合理性、可行性、創新性 |
| 資料品質與前處理 | 8% | 資料清洗完整、缺失值處理適當、特徵工程合理 |
| 探勘方法與模型 | 10% | 演算法選擇適當、模型評估完整、結果有洞察 |
| 視覺化與解讀 | 8% | 圖表清楚、分析結論有意義 |
| 程式碼品質 | 5% | 結構清楚、有註解、可讀性高 |
| 口頭報告與 Demo | 4% | 表達清楚、Demo 順暢、回答提問 |

#### 組間互評

期末發表時，每組對其他所有組進行評分：

| 評分項目 | 說明 |
|---------|------|
| 簡報表達 | 報告邏輯清楚、Demo 順暢 |
| 技術難度 | 使用技術的深度與廣度 |
| 實用價值 | 專題的實際應用性與創意 |
| 整體完成度 | 整體品質與完整性 |

每項 1-5 分，取各組評分的平均值。

#### 組內互評

每位組員匿名評分其他組員的貢獻度：

| 評分項目 | 說明 |
|---------|------|
| 工作投入 | 是否按時完成分工、主動參與 |
| 技術貢獻 | 程式碼、分析、報告的實際貢獻 |
| 團隊合作 | 溝通協調、配合度 |

每人有 100 分分配給其他組員（不含自己）。偏離平均值過大的將被標記審查。

#### AI 評分

透過 Python 腳本自動分析每位組員在 GitHub 上的實際貢獻：

| 檢查項目 | 說明 |
|---------|------|
| Commit 次數 | 每位組員的 commit 數量與頻率 |
| Commit 時間分布 | 是否穩定開發，而非截止前集中趕工 |
| 程式碼貢獻行數 | 每人新增與修改的程式碼量 |
| Commit 訊息品質 | 訊息是否具描述性 |
| 檔案覆蓋範圍 | 是否參與多個模組 |
| 專題結構完整度 | proposal、src、notebooks、docs 各資料夾是否齊全 |

AI 評分結果將結合組內互評交叉比對，確保評分公平。若 AI 偵測到貢獻度明顯偏低的組員，教師將個別確認後調整該組員的個人分數。

---

## 課程工具對應表

| 工具/平台 | 導入週次 | 使用範圍 |
|----------|---------|---------|
| GitHub + Git | 第 1 週 | 全學期作業繳交與版本控制 |
| VS Code | 第 1 週 | 全學期程式開發 IDE |
| Anaconda + Python | 第 1 週 | 全學期 Python 開發環境 |
| Google Colab | 第 1 週 | 全學期雲端實作環境 |
| Fork + Pull Request | 第 2 週 | 全學期作業繳交方式 |
| Pandas + NumPy | 第 2 週 | 資料處理與數值運算 |
| Matplotlib + Seaborn | 第 7 週 | 資料視覺化 |
| Scikit-learn | 第 3 週 | 機器學習演算法與預處理工具 |
| ChatGPT / Copilot | 第 10 週 | AI 輔助程式開發與資料分析 |
| XGBoost | 第 11 週 | 集成學習 |
| jieba | 第 14 週 | 中文斷詞 |
| Keras / TensorFlow | 第 15 週 | 深度學習 |

## 評分比例

| 項目 | 比例 | 評分方式 |
|------|------|---------|
| 平時作業（每週 Fork + PR） | 20% | 通過／不通過（Pass / Fail） |
| 小考（4 次） | 20% | 依答題正確率 |
| 期中考 | 20% | 依答題正確率 |
| 期末專題 | 20% | 依專題評分維度 |
| 課堂參與 | 20% | 出席 + 互動 |

> **作業評分方式**：每週作業採 **通過／不通過（Pass / Fail）** 制。完成所有要求項目即為通過，不另計分數。未繳交或明顯不完整為不通過。

---

## Git 基本流程

### 首次設定（只需做一次）

安裝 Git 後，設定你的姓名和 Email：

```bash
git config --global user.name "你的姓名"
git config --global user.email "你的Email"
```

### Fork + Clone（每學期做一次）

1. 在老師的 Repo 頁面右上角點選 **Fork**，將 Repo 複製到你的帳號下
2. Clone 你的 Fork 到本機：

```bash
git clone https://github.com/你的帳號/114-2_DM.git
cd 114-2_DM
```

### 每週作業流程

```
建立作業資料夾 → 編輯檔案 → add → commit → push → 發 PR
```

Step 1：建立當週作業資料夾並完成作業

```bash
mkdir week01
cd week01
# 建立作業檔案，完成作業內容
```

Step 2：將作業加入版本控制並推送

```bash
git add .
git commit -m "完成第1週作業"
git push origin main
```

Step 3：到 GitHub 網頁發 Pull Request

1. 進入你的 Fork 頁面
2. 點選 **Contribute** > **Open pull request**
3. 確認內容後點選 **Create pull request**
4. PR 標題格式：`學號_姓名_week01`

### 同步老師的最新版本

當老師更新了課程 Repo，你的 Fork 不會自動更新。請依照以下步驟取得最新版本：

1. 進入你的 Fork 頁面：`https://github.com/你的帳號/114-2_DM`
2. 頁面上方會出現提示：`This branch is X commits behind pychang-ai:main`
3. 點選「**Sync fork**」按鈕
4. 點選「**Update branch**」
5. 回到本機，執行：
   ```bash
   cd 114-2_DM
   git pull origin main
   ```

這樣你的 Fork 和本機都會更新到最新版本。你自己建立的作業資料夾不受影響。

---

## Git 基本指令

### 版本控制基礎指令

| 指令 | 功能 | 使用範例 |
|------|------|---------|
| `git clone <url>` | 複製遠端 Repo 到本機 | `git clone https://github.com/user/repo.git` |
| `git status` | 查看目前檔案狀態 | `git status` |
| `git add <file>` | 將檔案加入暫存區 | `git add week01/q1_env.txt` |
| `git add .` | 將所有變更加入暫存區 | `git add .` |
| `git commit -m "訊息"` | 提交變更並附上說明 | `git commit -m "完成第1週作業"` |
| `git push origin main` | 將本機提交推送到 GitHub | `git push origin main` |
| `git pull origin main` | 從 GitHub 拉取最新版本 | `git pull origin main` |
| `git log` | 查看提交歷史紀錄 | `git log --oneline` |

### Fork 與 Pull Request 指令

| 指令 | 功能 | 使用範例 |
|------|------|---------|
| Fork | 在 GitHub 網頁上操作，複製老師的 Repo 到你的帳號 | 點選 Repo 右上角 Fork 按鈕 |
| Pull Request | 在 GitHub 網頁上操作，將你的作業提交回老師的 Repo | 點選 Contribute > Open pull request |

### 常用輔助指令

| 指令 | 功能 | 使用範例 |
|------|------|---------|
| `git diff` | 查看檔案變更的內容 | `git diff` |
| `git log --oneline` | 用精簡格式查看歷史紀錄 | `git log --oneline -5` |
| `git remote -v` | 查看遠端 Repo 的連結 | `git remote -v` |
| `git branch` | 查看目前所在的分支 | `git branch` |

---

## 作業資料夾結構

```
114-2_DM/
├── README.md
├── my-topics/             ← 個人題目探索（第 4-5 週）
│   ├── topic1_xxx.md
│   └── topic2_xxx.md
├── week01/
│   ├── q1_env.txt
│   ├── q2_pandas.txt
│   └── q3_concept.txt
├── week02/
│   └── ...
└── ...
```

## 課程軟體與環境

### Google Colab

本課程主要使用 Google Colab 作為雲端實作環境，免安裝、免設定，開啟瀏覽器即可使用。

Colab 網址：[https://colab.research.google.com](https://colab.research.google.com)

### 本機環境

如需在本機執行，請安裝以下軟體：

| 軟體 | 用途 |
|------|------|
| Git | 版本控制 |
| VS Code | 程式開發 IDE |
| Anaconda / Miniconda | Python 開發環境與套件管理 |

### Python 套件清單

本課程會使用的主要套件：

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jieba
wordcloud
keras
tensorflow
jupyter
```

---

## 注意事項

1. 每週作業請在截止日前發 PR 繳交
2. PR 標題請統一格式：`學號_姓名_weekXX`
3. Colab Notebook 請下載為 .ipynb 檔案後放入作業資料夾
4. 作業中使用 AI 工具輔助時，請註明使用的工具名稱與 Prompt
5. 第 10 週起的作業會包含 AI 輔助開發的相關題目
6. 期末專題的資料集請放在專題 Repo 的 data/ 資料夾中
7. 專題報告請同時繳交 Jupyter Notebook 與投影片

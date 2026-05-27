# 第 14 週｜作業
- 修改課本第12章程式碼，運用AI製作修改程式記錄的MD檔
  - 修改了什麼
  - 該程式區塊的功能說明
- 放置於README.MD中

# W14-Exercise
## Error01
ValueError: could not convert string to float: 'sales' 表示 df.corr() 方法嘗試對非數值型資料（特別是 'sales' 欄位，它包含字串值）計算相關性。corr() 方法只能處理數值型資料。
為了解決這個問題，我修改了程式碼，在 df.corr() 方法中添加了 numeric_only=True。這樣做會指示 corr() 方法只考慮資料框中的數值型欄位來計算相關性，從而避免了嘗試將字串轉換為浮點數的錯誤。
```
#sns.heatmap(df.corr().round(2), annot=True, cmap='coolwarm');
sns.heatmap(df.corr(numeric_only=True).round(2), annot=True, cmap='coolwarm');
```
這行程式碼的目的是繪製一個相關性熱力圖（heatmap），用來視覺化資料框 df 中各數值型欄位之間的相關性。具體來說：
- df.corr(numeric_only=True): 計算資料框 df 中所有數值型欄位之間的皮爾遜相關係數。numeric_only=True 確保只對數值型欄位進行計算，避免非數值型欄位導致的錯誤。
- .round(2): 將相關係數的結果四捨五入到小數點後兩位，使圖表更易讀
- sns.heatmap(...): 使用 Seaborn 函式庫中的 heatmap 函式來繪製熱力圖。
- annot=True: 在熱力圖的每個單元格中顯示相關係數的數值。
- cmap='coolwarm': 設定熱力圖的顏色映射。'coolwarm' 是一種常見的色譜，通常用於表示正相關（暖色）和負相關（冷色），並且中間色表示接近零的相關性。
## Error02
這個錯誤 ValueError: could not convert string to float: 'sales' 表示 corrwith() 方法嘗試將非數值型的 'sales' 欄位（以及其他非數值型欄位，例如 'salary'）轉換為浮點數來計算相關性，但這是不可能的。相關性計算只能針對數值型資料執行。為了修正這個問題，我會修改程式碼，確保在計算相關性之前，只選取資料框中的數值型欄位。
```
# df.drop('left', axis=1).corrwith(df['left']).round(2)
df.drop('left', axis=1).select_dtypes(include=np.number).corrwith(df['left']).round(2)
```
這段程式碼計算並顯示了資料集中所有數值型特徵與目標變數 left（是否離職）之間的相關係數，並且結果四捨五入到小數點後兩位。

這些相關係數的意義是：

- satisfaction_level (滿意度): -0.39 - 呈現負相關，表示員工滿意度越低，離職的可能性越高。
- last_evaluation (上次考評): 0.01 - 相關性非常弱，幾乎沒有線性關係。
- number_project (專案數量): 0.02 - 相關性非常弱，幾乎沒有線性關係。
- average_montly_hours (平均月工時): 0.07 - 相關性非常弱，幾乎沒有線性關係。
- time_spend_company (在公司待的時間): 0.14 - 呈現弱的正相關，表示在公司待的時間越長，離職的可能性稍高（這可能需要進一步分析，例如考慮過長時間工作等因素）。
- Work_accident (工作事故): -0.15 - 呈現弱的負相關，表示發生工作事故的員工離職的可能性略低。
- promotion_last_5years (近5年是否晉升): -0.06 - 呈現弱的負相關，表示近5年有晉升的員工離職的可能性略低。
總體來看，satisfaction_level 與離職率有最強的負相關，意味著員工滿意度是影響離職與否的重要因素。其他數值型特徵與離職率的線性相關性較弱。
## Error 03

```
# print('正確率：', accuracy_score(y_test, y_pred).round(2))
print('正確率：', round(accuracy_score(y_test, y_pred), 2))

```
## Error 04
```
print('測試集的結果', round(score, 3))
```
## Error 05
```
# print('測試集的結果', score.round(3))
print('測試集的結果', round(score, 3))
```


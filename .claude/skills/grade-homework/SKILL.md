---
name: grade-homework
description: 批改學生作業 PR。當 PY 提到批改、改作業、看 PR、review、學生交了沒、作業狀況時觸發。
argument-hint: [week 週次 或 PR 編號，可省略則批改全部]
allowed-tools: Read Write Edit Bash Glob Grep WebFetch
---

# 作業批改 Skill

批改 114-2_DM 學生的 Fork + PR 作業。採 Pass / Fail 制。

## 前置

```bash
cd /c/Users/User/Claude_projs/114-2_DM
gh auth status
```

若 gh 未認證，提醒 PY 執行 `! gh auth login`。

## 步驟 1：列出待批改 PR

```bash
# 列出所有 open PR
gh pr list --state open --limit 100 --json number,title,author,createdAt,files

# 若指定週次，過濾標題含 weekNN
gh pr list --state open --search "week07 in:title" --json number,title,author
```

輸出摘要表：

| PR # | 學生 | 標題 | 提交日期 | 檔案數 |
|---|---|---|---|---|

## 步驟 2：逐一讀取 PR 內容

對每個 PR：

```bash
# 查看 PR 變更的檔案列表
gh pr diff {PR_NUMBER} --name-only

# 查看完整 diff（讀取學生作業內容）
gh pr diff {PR_NUMBER}
```

## 步驟 3：AI 逐題檢查

### 檢查項目（以 W7 為例）

先讀取該週的作業要求：

```bash
cat week07/README.md
```

然後對每個 PR 的 diff 逐題檢查：

**Q1（迴歸）檢查清單：**

| 項目 | 檢查方式 | Pass 條件 |
|---|---|---|
| 檔案存在 | diff 中有 `q1_regression.txt` | 有 |
| 程式碼 | 搜尋 `LinearRegression` | 有使用 |
| 資料集 | 搜尋 `Ship_Performance` | 用對資料 |
| 簡單迴歸結果 | 搜尋 `斜率` `截距` `R²` `RMSE` | 四項都有填 |
| 多元迴歸結果 | 搜尋 `各特徵迴歸係數` | 有列出係數 |
| 比較分析 | 搜尋 `簡單 vs 多元` | 有寫分析 |

**Q2（羅吉斯迴歸）檢查清單：**

| 項目 | 檢查方式 | Pass 條件 |
|---|---|---|
| 檔案存在 | diff 中有 `q2_logistic.txt` | 有 |
| Pipeline | 搜尋 `StandardScaler` `LogisticRegression` | 有使用 |
| 資料處理 | 搜尋 `is_critical` | 有建立二元目標 |
| 係數分析 | 搜尋 `迴歸係數` | 有列出 |
| 分類報告 | 搜尋 `precision` `recall` `f1` | 有貼報告 |
| 混淆矩陣 | 搜尋 `TP` `FP` `FN` `TN` | 四格都有填 |
| ROC/AUC | 搜尋 `AUC` | 有值 |
| FN 情境分析 | 搜尋 `漏報` 或 `嚴重` | 有寫分析 |

**Q3（觀念題）檢查清單：**

| 項目 | 檢查方式 | Pass 條件 |
|---|---|---|
| 檔案存在 | diff 中有 `q3_concept.txt` | 有 |
| Q1 回答 | `A1` 後有實質內容 | 非空、非複製題目 |
| Q2 回答 | `A2` 後有實質內容 | 非空、非複製題目 |
| Q3 回答 | `A3` 後有實質內容 | 非空、非複製題目 |

### AI 深度分析（每份 PR 都做）

除了檢查清單，還要：

1. **讀懂學生的程式碼**：確認邏輯正確，不是亂湊
2. **檢查數字合理性**：R² 在 0~1 之間、RMSE 合理範圍、AUC 在 0.5~1 之間
3. **檢查是否抄襲**：比對不同 PR 的程式碼和文字，相似度過高標記
4. **觀念題品質**：回答是否有自己的理解，還是 AI 生成的制式回答

## 步驟 4：產出批改報告

每個 PR 產出：

```
PR #{number} — {學生姓名}（{學號}）
判定：✅ Pass / ❌ Fail

Q1 迴歸：✅ / ❌
  - 程式碼：✅
  - 簡單迴歸結果：✅
  - 多元迴歸結果：✅
  - 比較分析：✅
  - AI 評語：{一句話}

Q2 羅吉斯：✅ / ❌
  - Pipeline：✅
  - 係數分析：✅
  - 混淆矩陣：✅
  - ROC/AUC：✅
  - FN 情境分析：✅
  - AI 評語：{一句話}

Q3 觀念：✅ / ❌
  - Q1 回答品質：{好/普通/不足}
  - Q2 回答品質：{好/普通/不足}
  - Q3 回答品質：{好/普通/不足}
  - AI 評語：{一句話}

整體 AI 回饋：{2-3句話，鼓勵+建議}
```

## 步驟 5：執行批改動作

彙整所有 PR 的批改結果，請 PY 確認後批次執行：

**Pass 的 PR：**
```bash
gh pr review {NUMBER} --approve --body "$(cat <<'MSG'
✅ Pass

{AI 回饋}
MSG
)"
gh pr merge {NUMBER} --merge
```

**Fail 的 PR：**
```bash
gh pr review {NUMBER} --request-changes --body "$(cat <<'MSG'
❌ 未通過，請補充以下項目後重新提交：

{缺漏清單}
MSG
)"
```

## 步驟 6：產出班級總覽

```
=== W7 作業批改總覽 ===
總 PR 數：{N}
Pass：{N}（{%}）
Fail：{N}（{%}）
未繳：{N}

| # | 學生 | Q1 | Q2 | Q3 | 判定 | 備註 |
|---|---|---|---|---|---|---|
```

## 注意事項

- **不主動 merge**：列出結果讓 PY 確認，PY 說「執行」才批次 merge
- **Fail 要具體說明**：告訴學生缺什麼、怎麼補
- **鼓勵為主**：即使 Fail 也要肯定做得好的部分
- **抄襲標記**：只標記不直接判 Fail，交由 PY 決定

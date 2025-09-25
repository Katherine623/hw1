# Simple Linear Regression | CRISP-DM 全流程示範

本專案是一個以 Streamlit 製作的互動式簡單線性回歸教學網頁，完整展示 CRISP-DM 六大步驟，並可讓使用者自訂資料產生參數，體驗資料科學專案流程。

---

## 功能特色

- **CRISP-DM 六大步驟說明**：每個步驟皆有中文提示與說明。
- **互動式資料產生**：可調整斜率 (a)、截距 (b)、雜訊 (noise)、資料點數等參數。
- **即時視覺化**：自動繪製資料點與回歸線。
- **模型訓練與評估**：顯示模型參數與評估指標。
- **資料與模型下載**：可下載產生的資料與模型資訊。
- **部署教學**：說明如何將專案部署為網頁應用。

---

## 使用方式

1. **安裝必要套件**

   ```bash
   pip install streamlit numpy pandas scikit-learn
   ```

2. **執行程式**

   ```bash
   streamlit run app.py
   ```
   若無法直接執行，請用：
   ```bash
   python -m streamlit run app.py
   ```

3. **開啟瀏覽器**

   在瀏覽器輸入網址 [http://localhost:8501](http://localhost:8501)

---

## CRISP-DM 步驟簡介

1. **Business Understanding（業務/問題理解）**  
   說明專案目標與線性回歸應用場景。

2. **Data Understanding（資料理解）**  
   互動調整資料產生參數，觀察資料分布。

3. **Data Preparation（資料準備）**  
   產生並預覽資料，進行必要的前處理。

4. **Modeling（建模）**  
   使用 scikit-learn 線性回歸模型進行訓練。

5. **Evaluation（評估）**  
   顯示模型參數、MSE、R² 等評估指標。

6. **Deployment（部署）**  
   提供資料與模型下載，並說明如何部署應用。

---

## 檔案說明

- `app.py`：主程式，Streamlit 應用入口。
- `requirements.txt`：所需套件清單（可選）。
- 其他：專案規劃、紀錄等 Markdown 文件。

---

## 聯絡方式

如有問題，歡迎提出 issue 或聯絡專案作者。
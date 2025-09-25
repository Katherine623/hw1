# app.py
# Simple Linear Regression with full CRISP-DM walkthrough (Streamlit)
# Minimal deps: streamlit, numpy, pandas
import json
import io
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------
# Helper functions
# --------------------------
def generate_linear_data(a=2.0, b=1.0, noise_std=1.0, n=50, seed=42, x_min=0.0, x_max=10.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n)
    eps = rng.normal(0.0, noise_std, size=n)
    y = a * x + b + eps
    df = pd.DataFrame({"x": x, "y": y})
    return df

def fit_ols(df):
    # Closed-form OLS for y = a*x + b
    x = df["x"].values
    y = df["y"].values
    X = np.column_stack([x, np.ones_like(x)])     # [x, 1]
    # theta = [a_hat, b_hat]
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a_hat, b_hat = theta
    y_hat = a_hat * x + b_hat
    residuals = y - y_hat

    mse = float(np.mean((y - y_hat) ** 2))
    mae = float(np.mean(np.abs(y - y_hat)))
    # R^2
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "a_hat": float(a_hat),
        "b_hat": float(b_hat),
        "y_hat": y_hat,
        "residuals": residuals,
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }

def to_csv_download(df, filename="data.csv"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ 下載資料 CSV", data=csv_bytes, file_name=filename, mime="text/csv")

def to_json_download(obj, filename="model.json"):
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("⬇️ 匯出模型係數 JSON", data=data, file_name=filename, mime="application/json")

st.caption("© CRISP-DM 線性回歸 Demo — 可自由修改與延伸")

# --------------------------
# Streamlit UI (with error handling)
# --------------------------
try:
    st.set_page_config(page_title="CRISP-DM 線性回歸教學", page_icon="📈", layout="wide")

    st.title("📈 Simple Linear Regression | CRISP-DM 全流程示範")

    with st.sidebar:
        st.header("⚙️ 資料生成參數")
        a_true = st.slider("真實斜率 a", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
        b_true = st.slider("真實截距 b", min_value=-20.0, max_value=20.0, value=1.0, step=0.5)
        noise_std = st.slider("高斯噪聲標準差 σ", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        n_points = st.slider("資料點數 n", min_value=10, max_value=2000, value=100, step=10)
        seed = st.number_input("隨機種子", value=42, step=1)
        x_min, x_max = st.slider("x 範圍", 0.0, 100.0, (0.0, 10.0), step=0.5)
        st.caption("提示：調整 a、噪聲與 n 可以觀察擬合穩定度與 R² 變化。")

        st.header("🧪 建模設定")
        run_fit = st.button("🚀 產生資料並擬合")

    with st.expander("1) Business Understanding（業務/問題理解）✅", expanded=True):
        st.markdown(
            """
            **目標**：示範如何在存在雜訊的線性資料上，估計未知參數（斜率 `a` 與截距 `b`），並用 MSE/MAE/R² 等指標評估表現。  
            **情境**：例如在量測實驗或商業資料中，數據常呈線性趨勢但帶有噪聲；我們需要簡單、可解釋的模型快速估測關係。
            """
        )

    with st.expander("2) Data Understanding（資料理解）✅", expanded=True):
        st.markdown("先用你設定的參數**生成合成資料**，檢視散佈與統計描述。")
        df = generate_linear_data(a=a_true, b=b_true, noise_std=noise_std, n=n_points, seed=int(seed), x_min=x_min, x_max=x_max)
        st.dataframe(df.head(10))
        st.write("資料描述：")
        st.write(df.describe())

        st.markdown("**散佈圖（含真實關係線）**")
        chart_df = df.copy()
        chart_df["y_true_no_noise"] = a_true * chart_df["x"] + b_true
        st.scatter_chart(chart_df, x="x", y=["y", "y_true_no_noise"], height=300)

    with st.expander("3) Data Preparation（資料準備）✅", expanded=True):
        st.markdown(
            """
            - 本案例資料乾淨無缺失，但仍示範常見步驟：  
              1) 檢查缺值、去除重複  
              2) 依 x 排序（利於視覺化）  
              3) （選配）切分訓練/測試（本例直接全量建模）  
            """
        )
        df = df.drop_duplicates().sort_values("x").reset_index(drop=True)
        missing = df.isna().sum()
        st.write("缺值統計：")
        st.write(missing)

    with st.expander("4) Modeling（建模）✅", expanded=True):
        st.markdown("使用 **最小平方法 OLS** 擬合 `y = a·x + b` 的閉式解。")
        if run_fit:
            fit = fit_ols(df)
            a_hat, b_hat = fit["a_hat"], fit["b_hat"]

            st.subheader("參數估計")
            st.metric("估計斜率 â", f"{a_hat:.4f}", delta=f"(真值 {a_true:.2f})")
            st.metric("估計截距 b̂", f"{b_hat:.4f}", delta=f"(真值 {b_true:.2f})")

            st.subheader("評估指標")
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{fit['mse']:.4f}")
            col2.metric("MAE", f"{fit['mae']:.4f}")
            col3.metric("R²", f"{fit['r2']:.4f}")

            # 視覺化：擬合線
            st.markdown("**擬合線 vs. 觀測值**")
            plot_df = df.copy()
            plot_df["y_hat"] = fit["y_hat"]
            st.line_chart(plot_df[["x", "y_hat"]].set_index("x"), height=240)
            st.scatter_chart(plot_df, x="x", y="y", height=300)

            # 殘差
            st.markdown("**殘差分佈**（y - ŷ）")
            res_df = pd.DataFrame({"residual": fit["residuals"]})
            st.bar_chart(res_df["residual"], height=180)

            # 下載
            st.markdown("**下載/匯出**")
            to_csv_download(df, filename="synthetic_linear_data.csv")
            to_json_download({"a_hat": a_hat, "b_hat": b_hat, "mse": fit["mse"], "mae": fit["mae"], "r2": fit["r2"]}, filename="linear_model.json")
        else:
            st.info("點擊左側「🚀 產生資料並擬合」開始建模。")

    with st.expander("5) Evaluation（評估）✅", expanded=True):
        st.markdown(
            """
            - **R²** 越接近 1 代表線性關係越能解釋 y 的變異。  
            - 增加資料點數 `n` 通常能讓估計更穩定；提高噪聲 `σ` 會使 MSE/MAE 變大、R² 下降。  
            - 若模型對你的情境足夠，便可進入部署；若不夠，可考慮多項式回歸、特徵工程或更換模型。
            """
        )

    with st.expander("6) Deployment（部署）✅", expanded=True):
        st.markdown(
            """
            - 本應用以 **Streamlit** 直接作為互動網頁展示，方便教學與內部分享。  
            - 你可以：  
              1) `streamlit run app.py` 在本機啟動  
              2) 將專案包進 Docker，或部署到 Streamlit Cloud / Render / Fly.io / Azure Web App  
            - 匯出 `linear_model.json` 後，也可在其他服務載入係數完成線上推論（ŷ = â·x + b̂）。
            """
        )

    st.caption("© CRISP-DM 線性回歸 Demo — 可自由修改與延伸")
except Exception as e:
    st.error(f"執行時發生錯誤：{e}")
    import traceback
    st.text(traceback.format_exc())

import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =====================
# Load data
# =====================
@st.cache_data
def load_data():
    model_metrics = pd.read_csv("AppDemo/data/model_metrics.csv")
    strategy_metrics = pd.read_csv("AppDemo/data/strategy_metrics.csv")
    with open("AppDemo/data/signals_demo.json", "r") as f:
        signals_demo = json.load(f)
    return model_metrics, strategy_metrics, signals_demo

model_metrics, strategy_metrics, signals_demo = load_data()

# =====================
# Helper: Gauge chart
# =====================
def render_gauge(value, label):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "black"},
            "steps": [
                {"range": [0, 40], "color": "red"},
                {"range": [40, 60], "color": "yellow"},
                {"range": [60, 100], "color": "green"}
            ]
        },
        number={"font": {"size": 32}},
        title={"text": label}
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# =====================
# Simple Mode
# =====================
def simple_mode(signals):
    st.header("What should I trade now?")

    min_strength = st.slider("Min strength", 0, 100, 50)
    show_gauges = st.toggle("Show gauges", value=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("BUY")
        buys = [s for s in signals if s["decision"] == "BUY" and s["strength"] >= min_strength]
        if not buys:
            st.write("No BUY signals found right now.")
        for sig in buys:
            st.markdown(f"**{sig['asset']}** — {sig['decision']} (Strength {sig['strength']})")
            st.caption(sig["reason"])
            if show_gauges:
                st.plotly_chart(render_gauge(sig["strength"], "Strength"), use_container_width=True)

    with col2:
        st.subheader("SELL")
        sells = [s for s in signals if s["decision"] == "SELL" and s["strength"] >= min_strength]
        if not sells:
            st.write("No SELL signals found right now.")
        for sig in sells:
            st.markdown(f"**{sig['asset']}** — {sig['decision']} (Strength {sig['strength']})")
            st.caption(sig["reason"])
            if show_gauges:
                st.plotly_chart(render_gauge(sig["strength"], "Strength"), use_container_width=True)

# =====================
# Advanced Tabs
# =====================

# --- Model Comparison ---
def model_comparison_tab(model_metrics, asset, freq, dataset_code):
    st.subheader("Model Performance")
    df = model_metrics.query("asset == @asset and freq == @freq and dataset == @dataset_code")

    if df.empty:
        st.warning("No data available for this selection.")
        return

    cls = df[df["task"] == "CLS"][["model", "auc"]].sort_values("auc", ascending=False)
    reg = df[df["task"] == "REG"][["model", "mae"]].sort_values("mae", ascending=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Direction Models (AUC ↑ better)**")
        st.dataframe(cls.reset_index(drop=True))
    with col2:
        st.markdown("**Return Models (MAE ↓ better)**")
        st.dataframe(reg.reset_index(drop=True))

# --- Keyword Explorer ---
def keyword_explorer_tab(model_metrics, asset, freq):
    st.subheader("Keyword effect (Market only vs Market+Keywords)")

    df = model_metrics.query("asset == @asset and freq == @freq")
    if df.empty:
        st.warning("No keyword data for this selection.")
        return

    # Compute average metrics per dataset
    metrics = ["AUC", "ACC", "F1", "MAE", "RMSE", "Spearman"]
    base = df[df["dataset"] == "base"][metrics].mean()
    ext = df[df["dataset"] == "ext"][metrics].mean()
    uplift = ext - base

    # KPI cards for ΔAUC and ΔMAE
    col1, col2 = st.columns(2)
    col1.metric("Δ AUC", f"{uplift['AUC']:+.3f}")
    col2.metric("Δ MAE", f"{uplift['MAE']:+.3f}")

    # Direction vs Return metrics
    st.markdown("**Direction metrics**")
    dir_table = pd.DataFrame({
        "Metric": ["AUC", "Accuracy", "F1"],
        "Market only": [base["AUC"], base["ACC"], base["F1"]],
        "Market+Keywords": [ext["AUC"], ext["ACC"], ext["F1"]],
        "Uplift": [uplift["AUC"], uplift["ACC"], uplift["F1"]]
    })
    st.dataframe(dir_table.style.format("{:.3f}").applymap(
        lambda v: f"color: {'green' if v > 0 else 'red'}" if isinstance(v, (int, float)) else ""
    , subset=["Uplift"]))

    st.markdown("**Return metrics**")
    ret_table = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "Spearman"],
        "Market only": [base["MAE"], base["RMSE"], base["Spearman"]],
        "Market+Keywords": [ext["MAE"], ext["RMSE"], ext["Spearman"]],
        "Uplift": [uplift["MAE"], uplift["RMSE"], uplift["Spearman"]]
    })
    st.dataframe(ret_table.style.format("{:.3f}").applymap(
        lambda v: f"color: {'green' if v < 0 else 'red'}" if isinstance(v, (int, float)) else ""
    , subset=["Uplift"]))

    # Heatmap in expander
    with st.expander("Show uplift heatmap"):
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame(uplift).T, annot=True, cmap="RdYlGn", center=0, ax=ax)
        st.pyplot(fig)

# --- Strategy Insights ---
def strategy_insights_tab(strategy_metrics, asset, freq, dataset_code):
    st.subheader("Trading Strategy Performance")
    df = strategy_metrics.query("asset == @asset and freq == @freq and dataset == @dataset_code")
    if df.empty:
        st.warning("No strategy data for this selection.")
        return

    cols = ["family", "sharpe", "max_dd", "ann_return"]
    st.dataframe(df[cols].sort_values("sharpe", ascending=False).reset_index(drop=True))

# --- Context ---
def context_tab(asset, freq, dataset_code):
    st.subheader("Context view")
    st.write(f"Sentiment gauge and trending keywords could go here for {asset} ({freq}, {dataset_code}).")

# =====================
# Advanced Mode
# =====================
def advanced_mode(model_metrics, strategy_metrics, signals_demo):
    st.header("Show me why")

    assets = model_metrics["asset"].unique().tolist()
    asset = st.selectbox("Asset", assets)
    freq = st.radio("Frequency", ["D", "W"], horizontal=True)
    dataset_code = st.radio("Dataset", ["base", "ext"], horizontal=True)

    tabs = st.tabs(["Model Comparison", "Keyword Explorer", "Strategy Insights", "Context"])

    with tabs[0]:
        model_comparison_tab(model_metrics, asset, freq, dataset_code)

    with tabs[1]:
        keyword_explorer_tab(model_metrics, asset, freq)  # <- dataset_code removed

    with tabs[2]:
        strategy_insights_tab(strategy_metrics, asset, freq, dataset_code)

    with tabs[3]:
        context_tab(asset, freq, dataset_code)

# =====================
# Main App
# =====================
st.set_page_config(layout="wide")

mode = st.radio("Mode", ["Simple", "Advanced"], horizontal=True)
if mode == "Simple":
    simple_mode(signals_demo)
else:
    advanced_mode(model_metrics, strategy_metrics, signals_demo)

import streamlit as st
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ARTE_DIR = Path(__file__).resolve().parent / "artefacts"

# ------------------------------------------------------------
# Helper: load metrics
# ------------------------------------------------------------
def load_metrics(file: Path):
    if file.exists():
        try:
            return pd.read_csv(file)
        except Exception as e:
            st.warning(f"Failed to read {file.name}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# ------------------------------------------------------------
# Landing Page
# ------------------------------------------------------------
def landing():
    st.title("📊 Market Model Dashboard")

    st.markdown("### 🔹 Baseline Metrics")
    tab1, tab2 = st.tabs(["Daily (D)", "Weekly (W)"])
    with tab1:
        df_base_D = load_metrics(ARTE_DIR / "metrics_summary_D.csv")
        if df_base_D.empty:
            st.info("No baseline metrics for Daily.")
        else:
            st.dataframe(df_base_D)
    with tab2:
        df_base_W = load_metrics(ARTE_DIR / "metrics_summary_W.csv")
        if df_base_W.empty:
            st.info("No baseline metrics for Weekly.")
        else:
            st.dataframe(df_base_W)

    st.markdown("---")
    st.markdown("### 🔹 Keyword Model Metrics")
    tab3, tab4 = st.tabs(["Daily (D)", "Weekly (W)"])
    with tab3:
        df_kw_D = load_metrics(ARTE_DIR / "metrics_keywords_D.csv")
        if df_kw_D.empty:
            st.info("No keyword metrics for Daily.")
        else:
            st.dataframe(df_kw_D)
    with tab4:
        df_kw_W = load_metrics(ARTE_DIR / "metrics_keywords_W.csv")
        if df_kw_W.empty:
            st.info("No keyword metrics for Weekly.")
        else:
            st.dataframe(df_kw_W)

# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
def main():
    pages = {
        "Landing": landing,
        # future: "Signal Explorer": signal_explorer,
        # future: "Backtest": backtest,
    }

    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()

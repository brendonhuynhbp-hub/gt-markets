# app.py – Streamlit Demo App
import streamlit as st
import pandas as pd
import json
from pathlib import Path

# ==== Paths ====
ARTEFACTS_DIR = Path("AppDemo/artefacts")

# ==== Helpers ====
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def load_json(path: Path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def list_pairs() -> list:
    return [p.name for p in ARTEFACTS_DIR.iterdir() if p.is_dir()]

def load_metrics(pair: str, freq: str, metrics_type: str) -> pd.DataFrame:
    # metrics_type: "baseline" or "keywords"
    path = ARTEFACTS_DIR / pair / f"metrics_{metrics_type}_{freq}.csv"
    return load_csv(path)

def load_leaderboard(pair: str, freq: str) -> pd.DataFrame:
    path = ARTEFACTS_DIR / pair / f"leaderboard_{freq}.csv"
    return load_csv(path)

def show_fig(pair: str, fig_name: str):
    path = ARTEFACTS_DIR / pair / "figs" / fig_name
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.info(f"⚠️ Figure not found: {fig_name}")

# ==== Streamlit UI ====
st.set_page_config(page_title="GT Markets Demo", layout="wide")
st.title("📊 GT Markets – Demo App")

# Sidebar
st.sidebar.header("Navigation")
view = st.sidebar.radio("Choose view:", ["Overview", "Metrics Explorer", "Leaderboard", "Signals & Charts"])

# ==== Overview ====
if view == "Overview":
    st.subheader("Project Overview")
    st.markdown("""
    This demo app showcases:
    - **Baseline strategies** (SMA, EMA, …)
    - **Keyword-enhanced models** (ML/DL models with same strategies)
    - Metrics & leaderboards
    - Backtested equity curves & signal charts
    
    Data prepared from artefacts in **AppDemo/artefacts/**.
    """)

# ==== Metrics Explorer ====
elif view == "Metrics Explorer":
    st.subheader("Metrics Explorer")

    pairs = list_pairs()
    pair = st.selectbox("Select Pair", pairs)
    freq = st.radio("Frequency", ["D", "W"])
    metrics_type = st.radio("Metrics type", ["baseline", "keywords"])

    df = load_metrics(pair, freq, metrics_type)
    if df.empty:
        st.warning("No metrics available.")
    else:
        # Expected columns: [strategy, type, sharpe, return, drawdown, ...]
        st.dataframe(df)

        # Strategy filter
        strategies = sorted(df["strategy"].unique())
        strategy = st.selectbox("Filter by Strategy", strategies)
        filtered = df[df["strategy"] == strategy]
        st.write(filtered)

# ==== Leaderboard ====
elif view == "Leaderboard":
    st.subheader("Leaderboard")

    pairs = list_pairs()
    pair = st.selectbox("Select Pair", pairs)
    freq = st.radio("Frequency", ["D", "W"])

    df = load_leaderboard(pair, freq)
    if df.empty:
        st.warning("No leaderboard data.")
    else:
        st.dataframe(df)

# ==== Signals & Charts ====
elif view == "Signals & Charts":
    st.subheader("Signals & Charts")

    pairs = list_pairs()
    pair = st.selectbox("Select Pair", pairs)
    freq = st.radio("Frequency", ["D", "W"])

    figs_dir = ARTEFACTS_DIR / pair / "figs"
    if not figs_dir.exists():
        st.warning("No figures available.")
    else:
        figs = sorted([f.name for f in figs_dir.iterdir() if f.suffix in [".png", ".jpg"]])
        fig = st.selectbox("Select Figure", figs)
        show_fig(pair, fig)

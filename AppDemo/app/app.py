# GT Markets – Demo App
# -------------------------------------------------------------
# Pages:
# 1) Landing
# 2) Compare (Baseline vs Keywords)
# 3) Keyword Lab
# 4) Signals & Audit
# 5) Backtest
# 6) Diagnostics
# -------------------------------------------------------------

import os
from pathlib import Path
import json
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
APP_TITLE = "GT Markets – Demo App"

# Artefact root: default to repo folder, can override with env var
ARTE_ROOT = Path(os.environ.get("ARTE_ROOT", "AppDemo/artefacts"))

KEYWORD_DIR = Path("keyword_sets")
KEYWORD_FILE = KEYWORD_DIR / "keyword_sets.json"
KEYWORD_DIR.mkdir(parents=True, exist_ok=True)

ASSET_ORDER = ["GOLD", "BTC", "OIL", "USDCNY"]
FREQ_LABELS = {"D": "Daily", "W": "Weekly"}
REQUIRED_BASELINE = ["metrics_baseline_D.csv", "metrics_baseline_W.csv"]

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_assets(arte_root: Path) -> List[str]:
    if not arte_root.exists():
        return []
    assets = sorted([p.name for p in arte_root.iterdir() if p.is_dir()])
    order = [a for a in ASSET_ORDER if a in assets]
    rest = [a for a in assets if a not in ASSET_ORDER]
    return order + rest

def _csv_safe_read(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            df = pd.read_csv(path)
            for c in ["Date", "date", "timestamp", "time"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="ignore")
            return df
    except Exception as e:
        st.warning(f"Failed to read {path.name} — {e}")
    return None

@st.cache_data(show_spinner=False)
def load_metrics(asset: str, freq: str):
    root = ARTE_ROOT / asset
    mb = root / f"metrics_baseline_{freq}.csv"
    mk = root / f"metrics_keywords_{freq}.csv"
    return _csv_safe_read(mb), _csv_safe_read(mk)

@st.cache_data(show_spinner=False)
def load_signals(asset: str, strategy: str, freq: str):
    root = ARTE_ROOT / asset
    sig = root / f"signals_{strategy}_{freq}.csv"
    return _csv_safe_read(sig)

@st.cache_data(show_spinner=False)
def load_leaderboard(asset: str, freq: str):
    return _csv_safe_read(ARTE_ROOT / asset / f"leaderboard_{freq}.csv")

@st.cache_data(show_spinner=False)
def load_features_text(asset: str):
    path = ARTE_ROOT / asset / "features_used.txt"
    if path.exists():
        return path.read_text(errors="ignore")
    return None

@st.cache_data(show_spinner=False)
def find_equity_figs(asset: str, freq: str) -> List[Path]:
    figs_dir = ARTE_ROOT / asset / "figs"
    if not figs_dir.exists():
        return []
    return sorted([p for p in figs_dir.glob(f"*_{freq}.png")])

@st.cache_data(show_spinner=False)
def discover_strategies(asset: str, freq: str) -> List[str]:
    root = ARTE_ROOT / asset
    if not root.exists():
        return []
    names = set()
    for p in root.glob(f"signals_*_{freq}.csv"):
        parts = p.stem.split("_")
        if len(parts) >= 3:
            names.add("_".join(parts[1:-1]))
    return sorted(names)

# -------------------------------------------------------------
# Backtest engine (MVP)
# -------------------------------------------------------------
def run_backtest(df: pd.DataFrame, threshold_long=0.6, threshold_short=0.4,
                 stop_loss=None, take_profit=None, cost_bps=0.0):
    """Simple backtest using prob_up or signal columns."""
    if df is None or df.empty:
        return None, None

    df = df.copy()
    date_col = next((c for c in ["Date","date","time","timestamp"] if c in df.columns), None)
    price_col = next((c for c in ["Close","close","price"] if c in df.columns), None)
    if date_col is None or price_col is None:
        st.error("Signals must include Date and Close columns.")
        return None, None

    # Signal source
    if "prob_up" in df.columns:
        df["pos"] = 0
        df.loc[df["prob_up"] > threshold_long, "pos"] = 1
        df.loc[df["prob_up"] < threshold_short, "pos"] = -1
    elif "signal" in df.columns:
        df["pos"] = df["signal"].clip(-1,1)
    else:
        st.error("No prob_up or signal column found.")
        return None, None

    df = df.dropna(subset=[price_col]).reset_index(drop=True)
    df["ret"] = df[price_col].pct_change().fillna(0)
    df["strat_ret"] = df["pos"].shift(1) * df["ret"]

    # Apply costs
    trades = df["pos"].diff().fillna(0).ne(0)
    df.loc[trades, "strat_ret"] -= cost_bps/10000

    equity = (1+df["strat_ret"]).cumprod()
    kpis = {
        "CAGR": equity.iloc[-1]**(252/len(df)) - 1 if len(df)>0 else np.nan,
        "Sharpe": np.mean(df["strat_ret"])/np.std(df["strat_ret"])*np.sqrt(252) if df["strat_ret"].std()!=0 else np.nan,
        "MaxDD": (equity/ equity.cummax() -1).min(),
        "WinRate": (df["strat_ret"]>0).mean()
    }
    return equity, kpis

# -------------------------------------------------------------
# Pages
# -------------------------------------------------------------
def page_landing(all_assets):
    st.header("Landing")
    if not all_assets:
        st.error("No artefacts found.")
        return
    asset = st.selectbox("Asset", all_assets, 0)
    freq = st.radio("Frequency", list(FREQ_LABELS.keys()), horizontal=True,
                    format_func=lambda x: FREQ_LABELS[x])
    mb, mk = load_metrics(asset, freq)
    st.subheader("Baseline Metrics")
    st.dataframe(mb if mb is not None else pd.DataFrame())
    st.subheader("Keyword Metrics")
    st.dataframe(mk if mk is not None else pd.DataFrame())

def page_compare(all_assets):
    st.header("Compare — Baseline vs Keywords")
    if not all_assets:
        return
    asset = st.selectbox("Asset", all_assets, 0)
    freq = st.radio("Frequency", list(FREQ_LABELS.keys()), horizontal=True,
                    format_func=lambda x: FREQ_LABELS[x])
    mb, mk = load_metrics(asset, freq)
    c1,c2 = st.columns(2)
    c1.write("Baseline"); c1.dataframe(mb)
    c2.write("Keywords"); c2.dataframe(mk)

def page_signals_audit(all_assets):
    st.header("Signals & Audit")
    if not all_assets:
        return
    asset = st.selectbox("Asset", all_assets, 0)
    freq = st.radio("Frequency", list(FREQ_LABELS.keys()), horizontal=True,
                    format_func=lambda x: FREQ_LABELS[x])
    strats = discover_strategies(asset, freq)
    if not strats:
        st.info("No signals available.")
        return
    strat = st.selectbox("Strategy", strats)
    df = load_signals(asset, strat, freq)
    st.dataframe(df.tail(20) if df is not None else pd.DataFrame())

def page_backtest(all_assets):
    st.header("Backtest (MVP)")
    if not all_assets:
        return
    asset = st.selectbox("Asset", all_assets, 0)
    freq = st.radio("Frequency", list(FREQ_LABELS.keys()), horizontal=True,
                    format_func=lambda x: FREQ_LABELS[x])
    strats = discover_strategies(asset, freq)
    if not strats:
        st.info("No signals to backtest.")
        return
    strat = st.selectbox("Strategy", strats)
    df = load_signals(asset, strat, freq)
    if df is None or df.empty:
        st.warning("No signal data.")
        return

    thr_long = st.slider("Threshold Long", 0.5, 1.0, 0.6, 0.01)
    thr_short = st.slider("Threshold Short", 0.0, 0.5, 0.4, 0.01)
    cost_bps = st.number_input("Cost (bps)", 0.0, 50.0, 5.0)
    equity, kpis = run_backtest(df, thr_long, thr_short, cost_bps=cost_bps)
    if equity is not None:
        st.line_chart(equity)
        st.json(kpis)

def page_diagnostics(all_assets):
    st.header("Diagnostics")
    st.write("ARTE_ROOT:", ARTE_ROOT)
    for a in all_assets:
        st.subheader(a)
        for req in REQUIRED_BASELINE:
            if not (ARTE_ROOT/a/req).exists():
                st.warning(f"{a}/{req} missing")

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(f"Artefacts root: {ARTE_ROOT.resolve()}")
    assets = list_assets(ARTE_ROOT)
    page = st.sidebar.radio("Navigate",
        ["Landing","Compare","Keyword Lab","Signals & Audit","Backtest","Diagnostics"])
    if page=="Landing": page_landing(assets)
    elif page=="Compare": page_compare(assets)
    elif page=="Signals & Audit": page_signals_audit(assets)
    elif page=="Backtest": page_backtest(assets)
    elif page=="Diagnostics": page_diagnostics(assets)

if __name__=="__main__":
    main()

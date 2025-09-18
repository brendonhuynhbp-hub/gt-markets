# app.py
# Streamlit App — Experiments → Backtest Demo
# ------------------------------------------------------------
# HOW TO RUN:
#   streamlit run app.py
#
# WHAT YOU NEED (optional but recommended):
#   - Google Drive mounted locally (or copy the folder)
#   - Rolling metrics CSVs:
#       weekly/.../50_leaderboards/rolling_ml_metrics.csv
#       weekly/.../50_leaderboards/rolling_dl_metrics.csv
#       daily/... /50_leaderboards/rolling_ml_metrics.csv
#       daily/... /50_leaderboards/rolling_dl_metrics.csv
#   - Optional signals CSV to backtest, schema (any of these sets):
#       A) date, asset, prob, ret
#       B) date, asset, prob, fwd_ret
#       C) date, asset, score, ret
#       D) date, asset, prob, price  (we will compute ret from price pct change shifted -1)
#
# NOTES:
#  - If no signals file is provided or found, you can check "Use synthetic demo"
#    and the app will generate a small synthetic series to showcase the backtest UX.
#  - Keywords toggle is reflected by dataset selection:
#      with keywords: eng_base, eng_ext, raw_ext
#      without keywords: raw_base
# ------------------------------------------------------------

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Market Experiments & Backtest", layout="wide")

BASE = Path(os.environ.get("GT_MARKETS_BASE", "/content/drive/MyDrive/gt-markets/app-demo/extracted"))
WEEKLY_DIR = BASE / "weekly"
DAILY_DIR  = BASE / "daily"

METRIC_COLS = ["auc","acc","f1"]
ID_COLS     = ["asset","dataset","model"]

DATASETS_WITH_KW = {"eng_base","eng_ext","raw_ext"}
DATASETS_NO_KW   = {"raw_base"}

# -------------------------------
# UTILS
# -------------------------------
def latest_run(run_root: Path, prefix: str):
    if not run_root.exists():
        return None
    cands = [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not cands:
        return None
    def ts(name):
        try:
            return datetime.strptime(name.split("_")[-1], "%Y%m%d-%H%M%S")
        except Exception:
            return datetime.min
    cands.sort(key=lambda p: ts(p.name), reverse=True)
    return cands[0]

def lb_paths(run_dir: Path):
    if run_dir is None: return {}
    lb = run_dir / "50_leaderboards"
    return {
        "ml": lb / "rolling_ml_metrics.csv",
        "dl": lb / "rolling_dl_metrics.csv",
        "out_dir": lb
    }

@st.cache_data(show_spinner=False)
def read_csv_safe(path: Path):
    if not path or not path.exists(): return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def clean_metrics(df: pd.DataFrame):
    if df is None or df.empty: return df
    df = df.copy()
    for c in METRIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "folds" in df.columns:
        df["folds"] = pd.to_numeric(df["folds"], errors="coerce")
    keep = [c for c in (ID_COLS + METRIC_COLS + ["folds"]) if c in df.columns]
    return df[keep].dropna(subset=["asset","dataset","model"], how="any")

def summarise(df: pd.DataFrame, freq: str, kind: str):
    if df is None or df.empty: return None
    mcols = [c for c in METRIC_COLS if c in df.columns]
    out = df.groupby(ID_COLS, dropna=False)[mcols].mean().reset_index()
    out = out.rename(columns={c: f"{c}_mean" for c in mcols})
    out["freq"] = freq
    out["kind"] = kind
    return out

def load_all_summaries():
    lw = latest_run(WEEKLY_DIR, "w_prod")
    ld = latest_run(DAILY_DIR,  "d_prod")
    paths_w = lb_paths(lw) if lw else {}
    paths_d = lb_paths(ld) if ld else {}

    ml_w = clean_metrics(read_csv_safe(paths_w.get("ml")))
    dl_w = clean_metrics(read_csv_safe(paths_w.get("dl")))
    ml_d = clean_metrics(read_csv_safe(paths_d.get("ml")))
    dl_d = clean_metrics(read_csv_safe(paths_d.get("dl")))

    sum_w_ml = summarise(ml_w, "weekly", "ml")
    sum_w_dl = summarise(dl_w, "weekly", "dl")
    sum_d_ml = summarise(ml_d, "daily",  "ml")
    sum_d_dl = summarise(dl_d, "daily",  "dl")

    parts = [x for x in [sum_w_ml,sum_w_dl,sum_d_ml,sum_d_dl] if x is not None]
    if parts:
        out = pd.concat(parts, ignore_index=True)
        # pick metric preference
        metric = "auc_mean" if "auc_mean" in out.columns else ("acc_mean" if "acc_mean" in out.columns else ("f1_mean" if "f1_mean" in out.columns else None))
        return out, metric
    return None, None

def metric_label(colname: str):
    return {"auc_mean":"AUC", "acc_mean":"Accuracy", "f1_mean":"F1"}.get(colname, colname)

# -------------------------------
# BACKTEST ENGINE (vectorized)
# -------------------------------
def coerce_signals(df: pd.DataFrame):
    """Normalize columns: date, asset, prob, ret"""
    df = df.copy()
    # date
    if "date" not in df.columns:
        raise ValueError("Signals CSV must have a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # probability / score
    prob_col = None
    for c in ["prob","score","signal","pred_proba","p"]:
        if c in df.columns:
            prob_col = c; break
    if prob_col is None:
        raise ValueError("Signals CSV should include a probability-like column: one of ['prob','score','signal','pred_proba','p']")

    # returns
    ret_col = None
    for c in ["ret","fwd_ret","return","target_ret","y_ret"]:
        if c in df.columns:
            ret_col = c; break

    if ret_col is None:
        # Try compute from price
        price_col = None
        for c in ["price","close","asset_price"]:
            if c in df.columns:
                price_col = c; break
        if price_col is None:
            raise ValueError("Signals CSV requires a forward return column (ret/fwd_ret/return) or a price column (price/close) to compute returns.")
        # forward pct change as next-period ret
        df["ret"] = df[price_col].pct_change().shift(-1)
    else:
        df["ret"] = pd.to_numeric(df[ret_col], errors="coerce")

    df["prob"] = pd.to_numeric(df[prob_col], errors="coerce")
    # asset optional; if missing treat as single-asset
    if "asset" not in df.columns:
        df["asset"] = "ASSET"

    # drop NaNs in prob/ret
    df = df.dropna(subset=["prob","ret"])
    return df[["date","asset","prob","ret"]]

def strategy_threshold(df: pd.DataFrame, upper=0.6, lower=0.4, side="longshort", confirm_n=1):
    """
    side: 'longonly', 'shortonly', 'longshort'
    confirm_n: require N consecutive prob>upper (or <lower) to enter
    """
    df = df.copy()
    # classification
    long_raw  = (df["prob"] > upper).astype(int)
    short_raw = (df["prob"] < lower).astype(int)

    # confirmation windows
    if confirm_n > 1:
        long_sig = long_raw.rolling(confirm_n).sum() == confirm_n
        short_sig = short_raw.rolling(confirm_n).sum() == confirm_n
    else:
        long_sig = long_raw.astype(bool)
        short_sig = short_raw.astype(bool)

    pos = np.zeros(len(df))
    if side == "longonly":
        pos = np.where(long_sig, 1.0, 0.0)
    elif side == "shortonly":
        pos = np.where(short_sig, -1.0, 0.0)
    else:  # longshort
        pos = np.where(long_sig, 1.0, np.where(short_sig, -1.0, 0.0))

    df["position"] = pos
    df["strategy_ret_raw"] = df["position"] * df["ret"]
    return df

def apply_risk_overlays(df: pd.DataFrame, stop_loss=None, take_profit=None):
    """
    stop_loss, take_profit: e.g., 0.05 for 5% (per-period cap)
    Applies simple per-period cap. For more realism you'd need intraperiod path.
    """
    df = df.copy()
    strat = df["strategy_ret_raw"].values
    if stop_loss is not None:
        strat = np.where(strat < -abs(stop_loss), -abs(stop_loss), strat)
    if take_profit is not None:
        strat = np.where(strat >  abs(take_profit),  abs(take_profit), strat)
    df["strategy_ret"] = strat
    return df

def equity_metrics(df: pd.DataFrame, capital=1.0):
    x = (1 + df["strategy_ret"].fillna(0)).cumprod() * capital
    # daily/weekly agnostic Sharpe (assumes returns series periodicity already)
    r = df["strategy_ret"].dropna()
    sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252)  # if weekly data, you can change to 52
    # Max drawdown
    roll_max = x.cummax()
    drawdown = (x - roll_max) / roll_max
    max_dd = drawdown.min()
    # CAGR (approx)
    n = max((df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25, 1e-9)
    cagr = (x.iloc[-1] / x.iloc[0])**(1/n) - 1 if len(x) > 1 else 0.0
    # Win rate
    trades = (df["position"].diff().fillna(0) != 0).sum()  # change points
    win_rate = (df["strategy_ret"] > 0).mean()
    return {
        "Final Equity": float(x.iloc[-1]),
        "CAGR": float(cagr),
        "Sharpe (annualized)": float(sharpe),
        "Max Drawdown": float(max_dd),
        "Win Rate": float(win_rate),
        "Obs": int(len(df)),
        "Trades (position changes)": int(trades)
    }, x, drawdown

def synthetic_demo_series(n=200, seed=7, trend=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    # latent signal
    z = rng.normal(0, 1, size=n).cumsum()*0.02 + trend*np.linspace(0,1,n)
    prob = 0.5 + (z - z.mean())/(z.std()+1e-9)*0.1  # squash around 0.5 +/- ~0.1
    prob = np.clip(prob, 0.0, 1.0)
    # returns partially aligned to prob
    ret = rng.normal(0, 0.01, size=n) + (prob-0.5)*0.04
    df = pd.DataFrame({"date":dates, "asset":"DEMO", "prob":prob, "ret":ret})
    return df

# -------------------------------
# UI — SIDEBAR: Experiment Selection
# -------------------------------
st.sidebar.title("⚙️ Experiment Setup")

summary_all, pref_metric = load_all_summaries()
if summary_all is None:
    st.error("Could not find rolling metrics CSVs. Check the paths or mount your Drive.")
    st.stop()

metric_col = st.sidebar.selectbox("Score metric", [c for c in ["auc_mean","acc_mean","f1_mean"] if c in summary_all.columns], index=[c for c in ["auc_mean","acc_mean","f1_mean"] if c in summary_all.columns].index(pref_metric) if pref_metric in summary_all.columns else 0)
metric_name = metric_label(metric_col)

assets = sorted(summary_all["asset"].unique().tolist())
asset = st.sidebar.selectbox("Asset", assets, index=assets.index("USDCNY") if "USDCNY" in assets else 0)

freqs = sorted(summary_all["freq"].unique().tolist())
default_freq = "daily" if asset in ["USDCNY"] else ("weekly" if "weekly" in freqs else freqs[0])
freq = st.sidebar.selectbox("Frequency", freqs, index=freqs.index(default_freq) if default_freq in freqs else 0)

kinds = sorted(summary_all["kind"].unique().tolist())  # ml/dl
default_kind = "ml" if "ml" in kinds else kinds[0]
kind = st.sidebar.selectbox("Model type", kinds, index=kinds.index(default_kind))

# limit dataset choices by kw toggle (optional UX sugar)
kw_toggle = st.sidebar.selectbox("Keywords", ["Auto (best)", "Force WITH keywords", "Force WITHOUT keywords"])
if kw_toggle == "Force WITH keywords":
    ds_opts = sorted([d for d in summary_all["dataset"].unique() if d in DATASETS_WITH_KW])
elif kw_toggle == "Force WITHOUT keywords":
    ds_opts = sorted([d for d in summary_all["dataset"].unique() if d in DATASETS_NO_KW])
else:
    ds_opts = sorted(summary_all["dataset"].unique())

dataset = st.sidebar.selectbox("Dataset", ds_opts)

# -------------------------------
# MAIN — Experiments view
# -------------------------------
st.title("🧪 Experiments → 📈 Backtest")
st.caption("Pick an asset + frequency + model type + dataset, review performance, then run a backtest with strategies.")

# Filter table for context
sub = summary_all[(summary_all["asset"]==asset) & (summary_all["freq"]==freq) & (summary_all["kind"]==kind)].copy()
if sub.empty:
    st.warning("No rows for this selection. Try another combination.")
else:
    # Rank by chosen metric
    sub = sub.sort_values(metric_col, ascending=False).reset_index(drop=True)
    st.subheader(f"Experiment Results — {asset} | {freq} | {kind.upper()}")
    st.dataframe(sub[[ "dataset","model", metric_col ]], use_container_width=True)
    best_row = sub[sub["dataset"]==dataset].head(1)
    if best_row.empty:
        best_row = sub.head(1)
        dataset = best_row["dataset"].iloc[0]

    # Summary boxes
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected Dataset", dataset)
    c2.metric("Top Model", best_row["model"].iloc[0])
    c3.metric(f"Best {metric_name}", f"{best_row[metric_col].iloc[0]:.3f}")
    # edge vs random for AUC
    if metric_col == "auc_mean":
        c4.metric("Excess vs Random", f"+{best_row[metric_col].iloc[0]-0.5:.3f}")
    else:
        c4.metric("Rows", f"{len(sub)}")

# -------------------------------
# BACKTEST Inputs
# -------------------------------
st.markdown("---")
st.header("Backtest the Chosen Configuration")

st.write("**Provide a signals CSV** (preferred) or use a small **synthetic demo**.")
up = st.file_uploader("Upload signals CSV (date, asset, prob/score, ret or price).", type=["csv"])

use_synth = st.checkbox("Use synthetic demo data if no CSV", value=True)

colA, colB, colC, colD = st.columns(4)
upper = colA.slider("Long threshold (prob > ...)", 0.5, 0.9, 0.6, 0.01)
lower = colB.slider("Short threshold (prob < ...)", 0.1, 0.5, 0.4, 0.01)
confirm_n = colC.slider("Confirmation candles", 1, 5, 1, 1)
side = colD.selectbox("Trading side", ["longshort","longonly","shortonly"])

colE, colF = st.columns(2)
stop_loss   = colE.slider("Stop-loss per period (%)", 0.0, 20.0, 0.0, 0.5) / 100.0
take_profit = colF.slider("Take-profit per period (%)", 0.0, 20.0, 0.0, 0.5) / 100.0

run_bt = st.button("▶️ Run Backtest")

# -------------------------------
# BACKTEST Execution
# -------------------------------
if run_bt:
    try:
        if up is not None:
            sig_raw = pd.read_csv(up)
        elif use_synth:
            st.info("Using synthetic demo series (randomized but aligned with positive edge).")
            sig_raw = synthetic_demo_series(n=240, trend=0.5)
        else:
            st.warning("Please upload a signals CSV or enable synthetic demo.")
            st.stop()

        sig = coerce_signals(sig_raw)
        # If multiple assets in uploaded file, keep chosen asset
        if asset in sig["asset"].unique():
            sig = sig[sig["asset"]==asset].copy()
        # Strategy
        strat = strategy_threshold(sig, upper=upper, lower=lower, side=side, confirm_n=confirm_n)
        strat = apply_risk_overlays(strat, stop_loss=(stop_loss if stop_loss>0 else None),
                                    take_profit=(take_profit if take_profit>0 else None))
        metrics, equity, dd = equity_metrics(strat)

        st.subheader("Results")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Final Equity", f"{metrics['Final Equity']:.2f}×")
        mcol2.metric("CAGR", f"{metrics['CAGR']*100:.2f}%")
        mcol3.metric("Sharpe (ann.)", f"{metrics['Sharpe (annualized)']:.2f}")
        mcol4, mcol5, mcol6 = st.columns(3)
        mcol4.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")
        mcol5.metric("Win Rate", f"{metrics['Win Rate']*100:.1f}%")
        mcol6.metric("Trades", f"{metrics['Trades (position changes)']}")

        # Charts
        st.line_chart(pd.DataFrame({"date": strat["date"], "Equity": equity.values}).set_index("date"))
        st.area_chart(pd.DataFrame({"date": strat["date"], "Drawdown": dd.values}).set_index("date"))

        with st.expander("Preview signals & strategy frame"):
            st.dataframe(strat.head(200), use_container_width=True)

        st.success("Backtest completed.")

    except Exception as e:
        st.error(f"Backtest failed: {e}")

# -------------------------------
# INSIGHT PANELS
# -------------------------------
st.markdown("---")
st.header("Insights & Guidance")

i1, i2, i3 = st.columns(3)
i1.info("**Pick assets with real edge**: USDCNY (daily) & Oil (weekly) first; Gold next; BTC is tough.")
i2.info("**Models**: Start with LR (stable). Try GRU for Oil weekly (higher upside, more variance).")
i3.info("**Keywords**: Useful for USDCNY & Gold; minor for Oil; avoid for BTC.")

st.caption("Tip: Use the Experiments table to select dataset/model/frequency, then backtest with thresholds, confirmations and risk overlays.")

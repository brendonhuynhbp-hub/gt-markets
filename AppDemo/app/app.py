# app.py — GT Markets Demo (metrics browser)
# Compatible with the new Block 1 outputs (strategy=SMA/EMA, model separate)

import os
from pathlib import Path
import re
import io

import pandas as pd
import streamlit as st

# ---------- Config ----------
APP_TITLE = "GT Markets · Model & Strategy Metrics"
# Prefer the repo artefacts if app is deployed from repo; fall back to Drive path locally
CANDIDATE_ARTEFACT_DIRS = [
    Path(__file__).parent / "AppDemo" / "artefacts",                       # repo relative
    Path.cwd() / "AppDemo" / "artefacts",                                  # cwd
    Path("/content/gt-markets/AppDemo/artefacts"),                         # Colab repo
    Path("/content/drive/MyDrive/gt-markets/AppDemo/artefacts"),           # Drive
]
# --------------------------------


# ---------- Utilities ----------
def find_artefacts_dir() -> Path:
    for p in CANDIDATE_ARTEFACT_DIRS:
        if p.exists():
            return p
    raise FileNotFoundError("Could not locate AppDemo/artefacts in any known location.")

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Failed to read {path.name}: {e}")
        return pd.DataFrame()

LEGACY_KW_RE = re.compile(
    r"^KW(?:_(?P<model>LR|RF|XGB|GRU|LSTM|MLP))?(?:_w(?P<win>\d+))?_(?P<src>BASE|EXT)_(?P<freq>D|W)$",
    re.IGNORECASE,
)

def normalize_keyword_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bring keyword metrics into the new schema if a legacy 'strategy' like
    'KW_XGB_BASE_D' is present. We:
      - keep strategy = SMA/EMA if already provided
      - otherwise set strategy = 'EMA' (default in our notebook trading rule)
      - extract model/window/source/freq from legacy tokens when needed
    """
    if df.empty:
        return df.copy()

    df = df.copy()

    # If we already have the new columns, just coerce types and return.
    have_new = {"strategy", "model", "window", "source"}.issubset(df.columns)
    if have_new:
        # Coerce and tidy
        if "window" in df.columns:
            df["window"] = pd.to_numeric(df["window"], errors="coerce").astype("Int64")
        if "strategy" in df.columns:
            df["strategy"] = df["strategy"].str.upper().replace({"SMA": "SMA", "EMA": "EMA"})
        if "source" in df.columns:
            df["source"] = df["source"].str.upper().replace({"BASE":"BASE", "EXT":"EXT"})
        if "model" in df.columns:
            df["model"] = df["model"].str.upper()
        return df

    # Otherwise, try to parse legacy 'strategy' like KW_XGB_BASE_D, KW_MLP_w30_EXT_W, etc.
    # We'll create the new columns (strategy/model/window/source) and leave existing metrics intact.
    model_col = []
    window_col = []
    source_col = []
    new_strategy = []

    for s in df.get("strategy", "").astype(str):
        m = LEGACY_KW_RE.match(s)
        if m:
            model = (m.group("model") or "").upper()
            win = m.group("win")
            src = (m.group("src") or "").upper()
            # Default strategy for keyword backtests in our latest design
            strat = "EMA"
        else:
            # If we can't parse, keep the text visible but still set model/source none-ish
            model = ""
            win = None
            src = ""
            # Heuristic: if the text already contains SMA/EMA we keep it, else default to EMA
            if "SMA" in s.upper():
                strat = "SMA"
            elif "EMA" in s.upper():
                strat = "EMA"
            else:
                strat = "EMA"

        model_col.append(model)
        window_col.append(int(win) if (win is not None and win.isdigit()) else pd.NA)
        source_col.append(src if src in ("BASE", "EXT") else pd.NA)
        new_strategy.append(strat)

    df["model"] = df.get("model", pd.Series(model_col))
    df["window"] = df.get("window", pd.Series(window_col, dtype="Int64"))
    df["source"] = df.get("source", pd.Series(source_col))
    df["strategy"] = df.get("strategy_trading", pd.Series(new_strategy))  # prefer explicit if it existed

    return df

def load_metrics(artefacts: Path, kind: str, freq: str) -> pd.DataFrame:
    """
    kind: 'baseline' or 'keywords'
    freq: 'D' or 'W'
    """
    file_map = {
        ("baseline", "D"): "metrics_baseline_D.csv",
        ("baseline", "W"): "metrics_baseline_W.csv",
        ("keywords", "D"): "metrics_keywords_D.csv",
        ("keywords", "W"): "metrics_keywords_W.csv",
    }
    path = artefacts / file_map[(kind, freq)]
    df = safe_read_csv(path)

    if df.empty:
        return df

    # Normalize column names (case/spacing)
    df.columns = [c.strip() for c in df.columns]

    # Ensure standard columns exist
    base_cols = ["type", "asset", "freq", "strategy"]
    for c in base_cols:
        if c not in df.columns:
            # Guess or fill
            if c == "freq":
                df["freq"] = freq
            elif c == "type":
                df["type"] = kind
            else:
                df[c] = pd.NA

    if kind == "keywords":
        df = normalize_keyword_rows(df)

    # Friendly column order if present
    first_cols = [c for c in ["type", "asset", "freq", "strategy", "model", "window", "source"] if c in df.columns]
    metric_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + metric_cols]

    # Sort by Sharpe if available
    if "Sharpe" in df.columns:
        df = df.sort_values("Sharpe", ascending=False, kind="mergesort")

    return df.reset_index(drop=True)

def to_download_bytes(df: pd.DataFrame) -> bytes:
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8")
# --------------------------------


# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

artefacts_dir = find_artefacts_dir()
st.caption(f"Artefacts: `{artefacts_dir}`")

colA, colB, colC, colD = st.columns([1.2, 1, 1, 1.2])
with colA:
    kind = st.radio("Metrics type", ["baseline", "keywords"], horizontal=True, index=1)
with colB:
    freq_label = st.radio("Frequency", ["Daily (D)", "Weekly (W)"], horizontal=True)
freq = "D" if "D" in freq_label else "W"

# Load
df_all = load_metrics(artefacts_dir, kind, freq)

if df_all.empty:
    st.warning("No metrics found for the current selection.")
    st.stop()

assets = sorted(df_all["asset"].dropna().unique().tolist())
with colC:
    asset = st.selectbox("Asset", assets, index=0 if assets else None)

# Filter by asset
df = df_all[df_all["asset"] == asset].copy()

# Sorting / columns
metric_candidates = [c for c in ["Sharpe", "Return_Ann", "MaxDD", "WinRate", "Trades"] if c in df.columns]
with colD:
    sort_by = st.selectbox("Sort by", ["Sharpe"] + [m for m in metric_candidates if m != "Sharpe"])
    ascending = st.toggle("Ascending", value=(sort_by != "Sharpe"))

if sort_by in df.columns:
    df = df.sort_values(sort_by, ascending=ascending, kind="mergesort")

# Pretty names
rename = {
    "freq": "freq",
    "strategy": "strategy",
    "model": "model",
    "window": "window",
    "source": "source",
    "Return_Ann": "Return_Ann",
    "MaxDD": "MaxDD",
    "WinRate": "WinRate",
}
df_display = df.rename(columns=rename)

# Show only the most useful columns up front
front_cols = [c for c in ["asset", "freq", "strategy", "model", "window", "source", "Return_Ann", "Sharpe", "MaxDD", "WinRate", "Trades"] if c in df_display.columns]
other_cols = [c for c in df_display.columns if c not in front_cols]
df_display = df_display[front_cols + other_cols]

st.subheader(f"{asset} · {kind} · {freq}")
st.dataframe(df_display, use_container_width=True, hide_index=True)

# Download
st.download_button(
    label="Download CSV",
    data=to_download_bytes(df_display),
    file_name=f"{asset}_{kind}_{freq}.csv",
    mime="text/csv",
)

# Helpful note
if kind == "keywords":
    st.info(
        "Keyword metrics now use the SAME trading strategies as baseline (SMA / EMA). "
        "The **model** column tells you which ML/DL model generated the probabilities; "
        "**window** is the lookback for the SMA/EMA overlay; **source** says whether features were BASE or EXT."
    )
else:
    st.caption("Baseline metrics are pure SMA/EMA strategies on price, no ML probabilities involved.")
# --------------------------------

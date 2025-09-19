# app.py — GT Markets Demo (metrics browser)
# Fix: robust string handling for strategy/model/source/window in keyword metrics

import os
from pathlib import Path
import re
import io

import pandas as pd
import streamlit as st

# ---------- Config ----------
APP_TITLE = "GT Markets · Model & Strategy Metrics"
CANDIDATE_ARTEFACT_DIRS = [
    Path(__file__).parent / ".." / "artefacts",                           # /AppDemo/app/app.py -> ../artefacts
    Path(__file__).parent / "AppDemo" / "artefacts",                       # repo relative (alt layout)
    Path.cwd() / "AppDemo" / "artefacts",                                  # cwd
    Path("/content/gt-markets/AppDemo/artefacts"),                         # Colab repo
    Path("/content/drive/MyDrive/gt-markets/AppDemo/artefacts"),           # Drive
]
# --------------------------------


# ---------- Utilities ----------
def find_artefacts_dir() -> Path:
    for p in CANDIDATE_ARTEFACT_DIRS:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("Could not locate AppDemo/artefacts in any known location.")

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Failed to read {path.name}: {e}")
        return pd.DataFrame()

# KW_XGB_BASE_D / KW_MLP_w30_EXT_W etc.
LEGACY_KW_RE = re.compile(
    r"^KW(?:_(?P<model>LR|RF|XGB|GRU|LSTM|MLP))?(?:_w(?P<win>\d+))?_(?P<src>BASE|EXT)_(?P<freq>D|W)$",
    re.IGNORECASE,
)

def to_str_series(s: pd.Series) -> pd.Series:
    """Coerce any series to pandas 'string' dtype safely (keeps <NA>), then uppercase helper."""
    # If series missing, create empty string series of right length
    if s is None:
        return pd.Series(pd.array([], dtype="string"))
    return s.astype("string")

def normalize_keyword_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize keyword metrics to the new schema:
      - strategy is SMA/EMA (never the model token)
      - model: LR/RF/XGB/GRU/LSTM/MLP
      - window: Int64 (nullable)
      - source: BASE/EXT
    Back-compat: if legacy 'strategy' like KW_XGB_BASE_D exists, parse it.
    """
    if df.empty:
        return df.copy()

    df = df.copy()

    # Ensure columns exist
    for c in ("strategy", "model", "window", "source"):
        if c not in df.columns:
            df[c] = pd.NA

    # Coerce to safe dtypes
    df["strategy"] = to_str_series(df["strategy"])
    df["model"]    = to_str_series(df["model"])
    df["source"]   = to_str_series(df["source"])
    # window -> nullable integer where possible
    try:
        df["window"] = pd.to_numeric(df["window"], errors="coerce").astype("Int64")
    except Exception:
        df["window"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # If we already have SMA/EMA in strategy (new format), just normalize casing and source/model
    has_new_style = df["strategy"].str.upper().isin(["SMA", "EMA"]).any()
    if has_new_style:
        df["strategy"] = df["strategy"].str.upper().where(df["strategy"].notna(), None)
        df["model"]    = df["model"].str.upper().where(df["model"].notna(), None)
        df["source"]   = df["source"].str.upper().replace({"BASE": "BASE", "EXT": "EXT"})
        return df

    # Legacy parsing path
    new_model, new_window, new_source, new_strategy = [], [], [], []
    raw_strategy = to_str_series(df["strategy"]).fillna("")

    for s in raw_strategy:
        s_up = (s or "").upper()
        m = LEGACY_KW_RE.match(s_up)
        if m:
            model = (m.group("model") or "")
            win   = m.group("win")
            src   = (m.group("src") or "")
            # Our unified trading overlay default for keyword metrics
            strat = "EMA"
            new_model.append(model)
            new_window.append(int(win) if win and win.isdigit() else pd.NA)
            new_source.append(src if src in ("BASE", "EXT") else pd.NA)
            new_strategy.append(strat)
        else:
            # Could be e.g. plain 'EMA'/'SMA' or something unexpected
            if "SMA" in s_up:
                strat = "SMA"
            elif "EMA" in s_up:
                strat = "EMA"
            else:
                strat = "EMA"  # safe default
            new_model.append("")
            new_window.append(pd.NA)
            new_source.append(pd.NA)
            new_strategy.append(strat)

    df["model"]    = pd.Series(new_model, dtype="string").str.upper()
    df["window"]   = pd.Series(new_window, dtype="Int64")
    df["source"]   = pd.Series(new_source, dtype="string").str.upper()
    df["strategy"] = pd.Series(new_strategy, dtype="string")

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

    # Normalize column names (trim)
    df.columns = [c.strip() for c in df.columns]

    # Ensure these exist
    for c in ("type", "asset", "freq", "strategy"):
        if c not in df.columns:
            if c == "freq":
                df[c] = freq
            elif c == "type":
                df[c] = kind
            else:
                df[c] = pd.NA

    # Coerce minimal types
    df["asset"] = to_str_series(df["asset"]).str.upper()
    df["freq"]  = to_str_series(df["freq"]).str.upper()
    df["type"]  = to_str_series(df["type"]).str.lower()

    if kind == "keywords":
        df = normalize_keyword_rows(df)
    else:
        # Baseline: ensure strategy is only SMA/EMA
        df["strategy"] = to_str_series(df["strategy"]).str.upper().replace({"SMA": "SMA", "EMA": "EMA"})

    # Column order
    first_cols = [c for c in ["type", "asset", "freq", "strategy", "model", "window", "source"] if c in df.columns]
    metric_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + metric_cols]

    # Sort by Sharpe if present
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

df_all = load_metrics(artefacts_dir, kind, freq)

if df_all.empty:
    st.warning("No metrics found for the current selection.")
    st.stop()

assets = sorted(df_all["asset"].dropna().unique().tolist())
with colC:
    asset = st.selectbox("Asset", assets, index=0 if assets else None)

df = df_all[df_all["asset"] == asset].copy()

metric_candidates = [c for c in ["Sharpe", "Return_Ann", "MaxDD", "WinRate", "Trades"] if c in df.columns]
with colD:
    sort_by = st.selectbox("Sort by", ["Sharpe"] + [m for m in metric_candidates if m != "Sharpe"])
    ascending = st.toggle("Ascending", value=(sort_by != "Sharpe"))

if sort_by in df.columns:
    df = df.sort_values(sort_by, ascending=ascending, kind="mergesort")

rename = {
    "Return_Ann": "Return_Ann",
    "MaxDD": "MaxDD",
    "WinRate": "WinRate",
}
df_display = df.rename(columns=rename)

front_cols = [c for c in ["asset", "freq", "strategy", "model", "window", "source", "Return_Ann", "Sharpe", "MaxDD", "WinRate", "Trades"] if c in df_display.columns]
other_cols = [c for c in df_display.columns if c not in front_cols]
df_display = df_display[front_cols + other_cols]

st.subheader(f"{asset} · {kind} · {freq}")
st.dataframe(df_display, use_container_width=True, hide_index=True)

st.download_button(
    label="Download CSV",
    data=to_download_bytes(df_display),
    file_name=f"{asset}_{kind}_{freq}.csv",
    mime="text/csv",
)

if kind == "keywords":
    st.info(
        "Keyword metrics use the SAME trading strategies as baseline (SMA / EMA). "
        "The **model** column shows which ML/DL model produced the probabilities; "
        "**window** is the SMA/EMA lookback; **source** is BASE or EXT."
    )
else:
    st.caption("Baseline metrics are pure SMA/EMA overlays on price (no ML probabilities).")
# --------------------------------

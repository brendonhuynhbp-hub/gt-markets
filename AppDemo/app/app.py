# AppDemo/app/app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Tuple

import pandas as pd
import streamlit as st


# ===== Paths =====
# Repo layout: AppDemo/app/app.py  <-- this file
#              AppDemo/artefacts   <-- metrics live here
HERE = Path(__file__).resolve()
APP_ROOT = HERE.parent
DEMO_ROOT = APP_ROOT.parent
ARTEFACTS_DIR = Path(
    os.getenv("ARTEFACTS_DIR", DEMO_ROOT / "artefacts")
).resolve()

# ===== UI =====
st.set_page_config(page_title="GT Markets · Model & Strategy Metrics", layout="wide")
st.title("Model & Strategy Metrics")
st.caption(f"Artefacts: `{ARTEFACTS_DIR}`")

kind = st.radio(
    "Metrics type",
    options=["baseline", "keywords", "baseline + keywords"],
    index=2,
    horizontal=False,
)

freq = st.radio(
    "Frequency",
    options=[("Daily (D)", "D"), ("Weekly (W)", "W")],
    index=0,
    format_func=lambda x: x[0],
    horizontal=False,
)
freq_code = freq[1]


# ===== Helpers =====
REQUIRED_ANY = {"type", "asset", "freq", "strategy"}
# We’ll normalize your current column names to these canonical ones.
CANON_COLS = {
    # current -> canonical
    "Return_Ann": "return",
    "WinRate": "hitrate",
    "Sharpe": "sharpe",
    "MaxDD": "maxdd",
    # already-canonical examples (no-op if present)
    "return": "return",
    "hitrate": "hitrate",
    "sharpe": "sharpe",
    "maxdd": "maxdd",
    "model": "model",
    "market": "market",
}


def load_csv_safely(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        st.warning(f"`{p.name}` not found in artefacts.")
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        st.error(f"Failed to read `{p.name}`: {e}")
        return None
    if df.empty:
        st.warning(f"`{p.name}` is empty.")
        return None
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure basic identity columns exist
    missing_any = REQUIRED_ANY - set(df.columns)
    if missing_any:
        # Some old files might use different casing; try to recover.
        lower = {c.lower(): c for c in df.columns}
        fixes = {}
        for need in REQUIRED_ANY:
            if need not in df.columns and need in lower:
                fixes[lower[need]] = need
        if fixes:
            df = df.rename(columns=fixes)

    # Map current names -> canonical names (without dropping originals)
    rename_map = {c: CANON_COLS[c] for c in df.columns if c in CANON_COLS and CANON_COLS[c] != c}
    if rename_map:
        df = df.rename(columns=rename_map)

    # If expected canonical metrics are still missing, create nullable placeholders
    for col in ["return", "hitrate", "sharpe", "maxdd"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Convert numerics if present
    for col in ["return", "hitrate", "sharpe", "maxdd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Make sure string-ish columns are strings (prevents .str accessor errors)
    for col in ["type", "asset", "freq", "strategy", "model"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Optional column: market (benchmark). If missing, we won’t require it.
    if "market" in df.columns:
        df["market"] = pd.to_numeric(df["market"], errors="coerce")

    return df


def load_metrics(
    artefacts_dir: Path, which: Literal["baseline", "keywords", "both"], freq_code: Literal["D", "W"]
) -> Tuple[pd.DataFrame | None, list[str]]:
    msgs: list[str] = []
    paths = []
    if which in ("baseline", "both"):
        paths.append(artefacts_dir / f"metrics_baseline_{freq_code}.csv")
    if which in ("keywords", "both"):
        paths.append(artefacts_dir / f"metrics_keywords_{freq_code}.csv")

    frames = []
    for p in paths:
        df = load_csv_safely(p)
        if df is None:
            continue
        df = normalize_columns(df)

        # Minimal schema guard for display; do not require 'market'
        needed = REQUIRED_ANY | {"return", "hitrate"}
        missing = needed - set(df.columns)
        if missing:
            msgs.append(f"`{p.name}` missing columns: {sorted(missing)}")
        frames.append(df)

    if not frames:
        return None, msgs

    out = pd.concat(frames, ignore_index=True)
    return out, msgs


# ===== Load & Filter =====
which = (
    "baseline"
    if kind == "baseline"
    else "keywords"
    if kind == "keywords"
    else "both"
)

df_all, notes = load_metrics(ARTEFACTS_DIR, which, freq_code)

if notes:
    for m in notes:
        st.warning(m)

if df_all is None or df_all.empty:
    st.error("No metrics files found in artefacts.")
    st.stop()

# Quick filters
cols = st.columns(4)
assets = sorted(df_all["asset"].dropna().unique().tolist()) if "asset" in df_all else []
chosen_assets = cols[0].multiselect("Asset", assets, default=assets)

strats = sorted(df_all["strategy"].dropna().unique().tolist()) if "strategy" in df_all else []
chosen_strats = cols[1].multiselect("Strategy", strats, default=strats)

models = sorted(df_all["model"].dropna().unique().tolist()) if "model" in df_all else []
default_models = models if models else []
chosen_models = cols[2].multiselect("Model (if any)", models, default=default_models)

sort_by = cols[3].selectbox("Sort by", ["return", "sharpe", "hitrate", "maxdd"])

# Apply filters (guard if columns are absent)
df_view = df_all.copy()
if "asset" in df_view and chosen_assets:
    df_view = df_view[df_view["asset"].isin(chosen_assets)]
if "strategy" in df_view and chosen_strats:
    df_view = df_view[df_view["strategy"].isin(chosen_strats)]
if "model" in df_view and chosen_models and "model" in df_view.columns:
    df_view = df_view[df_view["model"].isin(chosen_models)]

# Order columns for display
display_cols = [c for c in ["type", "asset", "freq", "strategy", "model", "return", "sharpe", "hitrate", "maxdd", "market"] if c in df_view.columns]
if not display_cols:
    display_cols = df_view.columns.tolist()

# Sort (fallback if column missing)
if sort_by in df_view.columns:
    df_view = df_view.sort_values(sort_by, ascending=False, na_position="last")

st.dataframe(df_view[display_cols], use_container_width=True, height=600)

st.caption(
    "Notes: This app accepts metrics files that export `Return_Ann` and `WinRate` "
    "from the notebook; these are normalized to `return` and `hitrate` automatically. "
    "`market` is optional."
)

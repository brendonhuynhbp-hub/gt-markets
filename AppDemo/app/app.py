# app.py — GT Markets Demo (with Debug + clean keyword strategy labels)

from __future__ import annotations
import os, re
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import streamlit as st

# --------------------------- Path discovery ---------------------------

def _candidate_roots(start: Path) -> List[Path]:
    here = start.resolve()
    cands = [
        Path(os.getenv("GT_ARTEFACTS_ROOT")) if os.getenv("GT_ARTEFACTS_ROOT") else None,
        here / "AppDemo" / "artefacts",
        here.parent / "AppDemo" / "artefacts",
        here.parent.parent / "AppDemo" / "artefacts",
        Path("/mount/src/gt-markets/AppDemo/artefacts"),
        Path("/mount/src/AppDemo/artefacts"),
        Path("/app/gt-markets/AppDemo/artefacts"),
        Path("/workspace/gt-markets/AppDemo/artefacts"),
    ]
    for up in [here, here.parent, here.parent.parent]:
        for m in up.rglob("AppDemo/artefacts"):
            cands.append(m)
    # dedupe
    out, seen = [], set()
    for c in cands:
        if c is None:
            continue
        s = str(c)
        if s not in seen:
            seen.add(s); out.append(c)
    return out

def _likely_valid(root: Path) -> bool:
    exp = [
        root / "metrics_summary_D.csv",
        root / "metrics_summary_W.csv",
        root / "metrics_keywords_D.csv",
        root / "metrics_keywords_W.csv",
    ]
    return any(p.exists() for p in exp)

@st.cache_data(show_spinner=False)
def detect_artefacts_root() -> Tuple[Path | None, List[str]]:
    checked = []
    for c in _candidate_roots(Path(__file__).parent):
        checked.append(str(c))
        if _likely_valid(c):
            return c, checked
    for c in _candidate_roots(Path(__file__).parent):
        checked.append(str(c))
        if c.exists():
            return c, checked
    return None, checked

def list_csvs_with_sizes(root: Path) -> List[Tuple[str,int]]:
    files = []
    for f in sorted(root.glob("**/*.csv")):
        try: size = f.stat().st_size
        except Exception: size = -1
        try: rel = str(f.relative_to(root))
        except Exception: rel = str(f)
        files.append((rel, size))
    return files

# ----------------------------- IO helpers -----------------------------

@st.cache_data(show_spinner=False)
def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path is None or not path.exists():
            return pd.DataFrame()
        if path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def files_in(folder: Path, pattern: str) -> List[Path]:
    return sorted(folder.glob(pattern)) if folder and folder.exists() else []

# ---------------------------- Table helpers ---------------------------

def _read_per_asset_kw(root: Path, freq: str) -> pd.DataFrame:
    """Concat per-asset keyword metrics; skip header-only tiny files."""
    parts = []
    for asset_dir in sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        f = asset_dir / f"metrics_keywords_{freq}.csv"
        if not f.exists() or f.stat().st_size < 20:
            continue
        df = safe_read_csv(f)
        if df.empty:
            continue
        if "asset" not in df.columns:
            df["asset"] = asset_dir.name
        if "freq" not in df.columns:
            df["freq"] = freq
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def _normalise_keyword_table(df: pd.DataFrame, hide_no_trades: bool) -> pd.DataFrame:
    """Clean keyword table for display:
       - ensure strategy column shows the trading rule name (FOLLOW_SIGNAL by default
         or whatever has been written in the CSV after your pipeline update)
       - optionally hide rows with N_Trades == 0
       - otherwise mark metrics as 'n/a' when N_Trades == 0
    """
    if df.empty:
        return df

    # If old CSVs carried model identifiers in 'strategy' (KW_*), map to FOLLOW_SIGNAL
    if "strategy" in df.columns:
        mask_kw = df["strategy"].astype(str).str.startswith("KW_", na=False)
        df.loc[mask_kw, "strategy"] = "FOLLOW_SIGNAL"

    # Handle no-trade rows
    if "N_Trades" in df.columns:
        if hide_no_trades:
            df = df[df["N_Trades"] > 0].copy()
        else:
            m = df["N_Trades"] == 0
            for c in ["Return_Ann","Sharpe","MaxDD","WinRate"]:
                if c in df.columns:
                    df.loc[m, c] = "n/a"
    return df

def load_root_tables(root: Path, hide_no_trades: bool) -> dict:
    base_D = safe_read_csv(root / "metrics_summary_D.csv")
    base_W = safe_read_csv(root / "metrics_summary_W.csv")

    kwD_path = root / "metrics_keywords_D.csv"
    kwW_path = root / "metrics_keywords_W.csv"
    kw_D = safe_read_csv(kwD_path)
    kw_W = safe_read_csv(kwW_path)

    if kw_D.empty or (kwD_path.exists() and kwD_path.stat().st_size < 20):
        kw_D = _read_per_asset_kw(root, "D")
    if kw_W.empty or (kwW_path.exists() and kwW_path.stat().st_size < 20):
        kw_W = _read_per_asset_kw(root, "W")

    kw_D = _normalise_keyword_table(kw_D, hide_no_trades)
    kw_W = _normalise_keyword_table(kw_W, hide_no_trades)

    return {"baseline_D": base_D, "baseline_W": base_W, "kw_D": kw_D, "kw_W": kw_W}

# ------------------------------ Pages ---------------------------------

def landing(root: Path, hide_no_trades: bool):
    st.header("🔷 Baseline Metrics")
    tabs = st.tabs(["Daily (D)", "Weekly (W)"])
    tables = load_root_tables(root, hide_no_trades)

    with tabs[0]:
        df = tables["baseline_D"]
        st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("No daily baseline metrics.")
    with tabs[1]:
        df = tables["baseline_W"]
        st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("No weekly baseline metrics.")

    st.header("🔹 Keyword Model Metrics")
    ktabs = st.tabs(["Daily (D)", "Weekly (W)"])
    with ktabs[0]:
        df = tables["kw_D"]
        st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.warning("No keyword metrics for Daily.")
    with ktabs[1]:
        df = tables["kw_W"]
        st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.warning("No keyword metrics for Weekly.")

def explorer(root: Path, hide_no_trades: bool):
    st.header("🧭 Explorer")
    # list assets by folder names
    assets = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not assets:
        st.info("No asset folders found.")
        return
    asset = st.selectbox("Asset", options=assets, index=assets.index("GOLD") if "GOLD" in assets else 0)
    freq = st.radio("Frequency", options=["D","W"], horizontal=True, index=0)
    asset_dir = root / asset

    st.subheader("Leaderboard")
    df_lb = safe_read_csv(asset_dir / f"leaderboard_{freq}.csv")
    st.dataframe(df_lb, use_container_width=True, hide_index=True) if not df_lb.empty else st.info("No leaderboard for this asset/freq.")

    st.subheader("Baseline Metrics (asset)")
    df_base = safe_read_csv(asset_dir / f"metrics_baseline_{freq}.csv")
    st.dataframe(df_base, use_container_width=True, hide_index=True) if not df_base.empty else st.info("No per-asset baseline metrics.")

    st.subheader("Keyword Metrics (asset)")
    df_kw = safe_read_csv(asset_dir / f"metrics_keywords_{freq}.csv")
    df_kw = _normalise_keyword_table(df_kw, hide_no_trades)
    st.dataframe(df_kw, use_container_width=True, hide_index=True) if not df_kw.empty else st.info("No per-asset keyword metrics.")

    with st.expander("📁 Signals available"):
        sigs = files_in(asset_dir, f"signals_*_{freq}.csv")
        st.info("No signals in this asset/freq.") if not sigs else st.code("\n".join([p.name for p in sigs]), language="text")

# ------------------------------- App ----------------------------------

def main():
    st.set_page_config(page_title="GT Markets — Demo App", page_icon="📊", layout="wide")
    root, checked = detect_artefacts_root()
    st.title("📊 GT Markets – Demo App")

    if root is None:
        st.error("Could not locate artefacts folder. Set GT_ARTEFACTS_ROOT or fix repo layout.")
        st.code("\n".join(checked), language="text")
        return

    st.caption(f"Artefacts root:  `{root}`")

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Landing","Explorer"], index=0)
        hide_no_trades = st.toggle("Hide rows with N_Trades = 0", value=True)
        DEBUG = st.toggle("Debug mode", value=False)
        st.caption("Paths checked:")
        st.code("\n".join(checked), language="text")

    if DEBUG:
        st.subheader("🔍 Debug")
        st.write(f"Exists: **{root.exists()}** | Absolute: `{root.resolve()}`")
        files = list_csvs_with_sizes(root)
        st.warning("No CSV files under artefacts root.") if not files else st.code(
            "\n".join([f"{nm}  ({sz} bytes)" for nm, sz in files]), language="text"
        )
        # quick sanity
        _kwd = safe_read_csv(root / "metrics_keywords_D.csv")
        _kww = safe_read_csv(root / "metrics_keywords_W.csv")
        st.write({"metrics_keywords_D_rows": len(_kwd), "metrics_keywords_W_rows": len(_kww)})
        st.divider()

    if page == "Landing":
        landing(root, hide_no_trades)
    else:
        explorer(root, hide_no_trades)

if __name__ == "__main__":
    main()

# app.py
# GT Markets — Demo App (Streamlit) with Debug Mode
# - Robust artefacts root resolver (works on Streamlit Cloud + local)
# - Safe CSV reader with human-friendly error reporting
# - Landing (root summaries) + Explorer (per-asset)
# - Optional Debug panel: shows resolved paths, file list + sizes, quick sanity checks

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st

# ============================================================
# Artefacts root resolution + debug
# ============================================================

def _candidate_roots(start: Path) -> List[Path]:
    """Reasonable places where AppDemo/artefacts could live in a deployed app."""
    here = start.resolve()
    cands = [
        # 0) Explicit override via env
        Path(os.getenv("GT_ARTEFACTS_ROOT")) if os.getenv("GT_ARTEFACTS_ROOT") else None,

        # 1) Common layouts relative to this file
        here / "AppDemo" / "artefacts",
        here.parent / "AppDemo" / "artefacts",
        here.parent.parent / "AppDemo" / "artefacts",

        # 2) Streamlit Cloud build workdir patterns
        Path("/mount/src/gt-markets/AppDemo/artefacts"),
        Path("/mount/src/AppDemo/artefacts"),
        Path("/app/gt-markets/AppDemo/artefacts"),
        Path("/workspace/gt-markets/AppDemo/artefacts"),
    ]
    # 3) Search upwards a bit for any AppDemo/artefacts
    for up in [here, here.parent, here.parent.parent]:
        for match in up.rglob("AppDemo/artefacts"):
            cands.append(match)
    # De-dup + remove None
    seen, out = set(), []
    for c in cands:
        if c is None:
            continue
        s = str(c)
        if s not in seen:
            seen.add(s)
            out.append(c)
    return out

def _likely_valid(root: Path) -> bool:
    """Treat as valid if any expected file exists."""
    expected = [
        root / "metrics_summary_D.csv",
        root / "metrics_summary_W.csv",
        root / "metrics_keywords_D.csv",
        root / "metrics_keywords_W.csv",
    ]
    return any(p.exists() for p in expected)

@st.cache_data(show_spinner=False)
def detect_artefacts_root() -> Tuple[Path | None, List[str]]:
    checked = []
    for c in _candidate_roots(Path(__file__).parent):
        checked.append(str(c))
        if _likely_valid(c):
            return c, checked
    # Fallback to first existing AppDemo/artefacts, even if empty
    for c in _candidate_roots(Path(__file__).parent):
        checked.append(str(c))
        if c.exists():
            return c, checked
    return None, checked

def list_csvs_with_sizes(root: Path) -> List[Tuple[str, int]]:
    files = []
    for f in sorted(root.glob("**/*.csv")):
        try:
            size = f.stat().st_size
        except Exception:
            size = -1
        try:
            rel = str(f.relative_to(root))
        except Exception:
            rel = str(f)
        files.append((rel, size))
    return files

# ============================================================
# I/O helpers
# ============================================================

@st.cache_data(show_spinner=False)
def safe_read_csv(path: Path) -> pd.DataFrame:
    """Defensive CSV loader with readable UI errors."""
    try:
        if path is None or not path.exists():
            st.error(f"Missing file: {path}")
            return pd.DataFrame()
        if path.stat().st_size == 0:
            st.error(f"Empty file: {path.name}")
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception as e:
        # Show a short byte preview to catch HTML/redirects etc.
        try:
            head = path.open("rb").read(200)
            st.error(f"Failed to read {path.name}: {e}")
            st.code(head, language="text")
        except Exception:
            st.error(f"Failed to read {path}: {e}")
        return pd.DataFrame()

def files_in(folder: Path, pattern: str) -> List[Path]:
    return sorted(folder.glob(pattern)) if folder and folder.exists() else []

# ============================================================
# Domain helpers
# ============================================================

ASSETS = ["GOLD", "BTC", "OIL", "USDCNY"]
FREQS = ["D", "W"]

def load_root_tables(root: Path) -> dict:
    return {
        "baseline_D": safe_read_csv(root / "metrics_summary_D.csv"),
        "baseline_W": safe_read_csv(root / "metrics_summary_W.csv"),
        "kw_D":       safe_read_csv(root / "metrics_keywords_D.csv"),
        "kw_W":       safe_read_csv(root / "metrics_keywords_W.csv"),
    }

def discover_assets(root: Path) -> List[str]:
    out = []
    for a in ASSETS:
        if (root / a).exists():
            out.append(a)
    # also include any extra folders that look like assets
    for d in root.iterdir():
        if d.is_dir() and d.name not in out and not d.name.startswith("."):
            out.append(d.name)
    return out

# ============================================================
# UI pages
# ============================================================

def landing(root: Path):
    st.header("🔷 Baseline Metrics")
    tabD, tabW = st.tabs(["Daily (D)", "Weekly (W)"])
    tables = load_root_tables(root)

    with tabD:
        df = tables["baseline_D"]
        if df.empty:
            st.info("No daily baseline metrics.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tabW:
        df = tables["baseline_W"]
        if df.empty:
            st.info("No weekly baseline metrics.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.header("🔹 Keyword Model Metrics")
    tabKD, tabKW = st.tabs(["Daily (D)", "Weekly (W)"])

    with tabKD:
        df = tables["kw_D"]
        if df.empty:
            st.warning("No keyword metrics for Daily.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tabKW:
        df = tables["kw_W"]
        if df.empty:
            st.warning("No keyword metrics for Weekly.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

def explorer(root: Path):
    st.header("🧭 Explorer")
    assets = discover_assets(root)
    if not assets:
        st.info("No asset folders found in artefacts.")
        return

    asset = st.selectbox("Asset", options=assets, index=assets.index("GOLD") if "GOLD" in assets else 0)
    freq = st.radio("Frequency", options=FREQS, horizontal=True, index=0)
    asset_dir = root / asset

    # Leaderboard
    st.subheader("Leaderboard")
    df_lb = safe_read_csv(asset_dir / f"leaderboard_{freq}.csv")
    st.dataframe(df_lb, use_container_width=True, hide_index=True) if not df_lb.empty else st.info("No leaderboard for this asset/freq.")

    # Baseline metrics (per-asset file)
    st.subheader("Baseline Metrics (asset)")
    df_base = safe_read_csv(asset_dir / f"metrics_baseline_{freq}.csv")
    st.dataframe(df_base, use_container_width=True, hide_index=True) if not df_base.empty else st.info("No per-asset baseline metrics.")

    # Keyword metrics (per-asset file)
    st.subheader("Keyword Metrics (asset)")
    df_kw = safe_read_csv(asset_dir / f"metrics_keywords_{freq}.csv")
    st.dataframe(df_kw, use_container_width=True, hide_index=True) if not df_kw.empty else st.info("No per-asset keyword metrics.")

    # Signals catalogue
    with st.expander("📁 Signals available"):
        sigs = files_in(asset_dir, f"signals_*_{freq}.csv")
        if not sigs:
            st.info("No signals in this asset/freq.")
        else:
            st.code("\n".join([p.name for p in sigs]), language="text")

# ============================================================
# App
# ============================================================

def main():
    st.set_page_config(page_title="GT Markets — Demo App", page_icon="📊", layout="wide")

    root, checked = detect_artefacts_root()
    st.title("📊 GT Markets – Demo App")

    if root is None:
        st.error("Could not locate artefacts folder. Set GT_ARTEFACTS_ROOT or fix repo layout.")
        st.code("\n".join(checked), language="text")
        return

    st.caption(f"Artefacts root:  `{root}`")

    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Landing", "Explorer"], index=0)
        DEBUG = st.toggle("Debug mode", value=False)
        st.caption("Paths checked:")
        st.code("\n".join(checked), language="text")

    # Optional Debug section
    if DEBUG:
        st.subheader("🔍 Debug")
        st.write(f"Exists: **{root.exists()}**  |  Absolute: `{root.resolve()}`")
        files = list_csvs_with_sizes(root)
        if files:
            st.markdown("**CSV files (relative path · size in bytes):**")
            st.code("\n".join([f"{nm}  ({sz} bytes)" for nm, sz in files]), language="text")
        else:
            st.warning("No CSV files found under artefacts root.")

        # Quick sanity: try reading the root keyword files specifically
        st.markdown("**Sanity read · root keyword metrics**")
        _kwd = safe_read_csv(root / "metrics_keywords_D.csv")
        _kww = safe_read_csv(root / "metrics_keywords_W.csv")
        st.write({"metrics_keywords_D_rows": len(_kwd), "metrics_keywords_W_rows": len(_kww)})

        st.divider()

    # Pages
    if page == "Landing":
        landing(root)
    else:
        explorer(root)

if __name__ == "__main__":
    main()

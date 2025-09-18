# app.py
# GT Markets – Demo App (Streamlit)
# - No Google Drive dependency
# - Auto-detects artefacts folder in common layouts
# - Clear diagnostics when CSVs are missing/empty
# - Separate panels: Baseline Metrics vs Keyword Model Metrics

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st


# ============================================================
# Config
# ============================================================

APP_TITLE = "📊 Market Model Dashboard"

# Optional override via env var if you want to force a path
# e.g. in Streamlit Cloud:  ARTEFACTS_ROOT="AppDemo/artefacts"
ENV_ARTEFACTS = os.environ.get("ARTEFACTS_ROOT", "").strip()


# ============================================================
# Helpers
# ============================================================

def _candidate_roots() -> List[Path]:
    """
    Return a prioritized list of possible artefacts roots.
    Order matters; first valid one will be used.
    """
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    candidates = [
        # 1) Explicit env var
        Path(ENV_ARTEFACTS) if ENV_ARTEFACTS else None,

        # 2) Side-by-side with this file (AppDemo structure)
        here / "artefacts",
        here / "AppDemo" / "artefacts",

        # 3) From working directory (repo root)
        cwd / "AppDemo" / "artefacts",
        cwd / "artefacts",

        # 4) Fallbacks sometimes used in containers
        Path("/mount/src/gt-markets/AppDemo/artefacts"),
        Path("/workspace/gt-markets/AppDemo/artefacts"),
    ]
    return [p for p in candidates if p is not None]


def _likely_valid(root: Path) -> bool:
    """A root is considered valid if the folder exists and at least one expected file is present."""
    if not root.exists():
        return False
    expected_any = [
        root / "metrics_summary_D.csv",
        root / "metrics_summary_W.csv",
        root / "metrics_keywords_D.csv",
        root / "metrics_keywords_W.csv",
    ]
    return any(p.exists() for p in expected_any)


@st.cache_data(show_spinner=False)
def detect_artefacts_root() -> Tuple[Path | None, list]:
    """
    Find the first artefacts root that looks valid.
    Returns (root_or_none, all_checked_paths_for_debug)
    """
    checked = []
    for candidate in _candidate_roots():
        checked.append(str(candidate))
        if _likely_valid(candidate):
            return candidate, checked
    # If none look valid, still return the first existing 'artefacts' directory (even if empty),
    # so the user sees directory listing + diagnostics
    for candidate in _candidate_roots():
        if candidate.exists():
            return candidate, checked
    return None, checked


@st.cache_data(show_spinner=False)
def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path or not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception as e:
        # Show a small note inline so users know why the table is empty
        st.warning(f"Failed to read {path.name}: {e}")
        return pd.DataFrame()


def list_assets(root: Path) -> List[str]:
    if not root or not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def files_in(root: Path, pattern: str) -> List[str]:
    if not root or not root.exists():
        return []
    return sorted([p.name for p in root.glob(pattern)])


# ============================================================
# UI Sections
# ============================================================

def landing(root: Path):
    st.title(APP_TITLE)
    st.caption(f"Artefacts root: `{root}`")

    # ---------------- Baseline ----------------
    st.markdown("## 🔹 Baseline Metrics")
    tab_d, tab_w = st.tabs(["Daily (D)", "Weekly (W)"])

    with tab_d:
        df_d = safe_read_csv(root / "metrics_summary_D.csv")
        if df_d.empty:
            st.info("No baseline metrics for Daily.")
        else:
            st.dataframe(df_d, use_container_width=True, hide_index=True)

    with tab_w:
        df_w = safe_read_csv(root / "metrics_summary_W.csv")
        if df_w.empty:
            st.info("No baseline metrics for Weekly.")
        else:
            st.dataframe(df_w, use_container_width=True, hide_index=True)

    st.divider()

    # ---------------- Keywords ----------------
    st.markdown("## 🔹 Keyword Model Metrics")
    tab_kd, tab_kw = st.tabs(["Daily (D)", "Weekly (W)"])

    with tab_kd:
        kw_d = safe_read_csv(root / "metrics_keywords_D.csv")
        if kw_d.empty:
            st.info("No keyword metrics for Daily.")
        else:
            st.dataframe(kw_d, use_container_width=True, hide_index=True)

    with tab_kw:
        kw_w = safe_read_csv(root / "metrics_keywords_W.csv")
        if kw_w.empty:
            st.info("No keyword metrics for Weekly.")
        else:
            st.dataframe(kw_w, use_container_width=True, hide_index=True)

    # ------------- Diagnostics ---------------
    with st.expander("🔎 Diagnostics"):
        assets = list_assets(root)
        st.write("**Detected asset folders:**", ", ".join(assets) if assets else "—")
        st.write("**Root files present:**")
        cols = st.columns(2)
        with cols[0]:
            st.write("- metrics_summary_D.csv:", (root / "metrics_summary_D.csv").exists())
            st.write("- metrics_keywords_D.csv:", (root / "metrics_keywords_D.csv").exists())
        with cols[1]:
            st.write("- metrics_summary_W.csv:", (root / "metrics_summary_W.csv").exists())
            st.write("- metrics_keywords_W.csv:", (root / "metrics_keywords_W.csv").exists())
        st.write("**All CSVs in root:**")
        st.code("\n".join(files_in(root, "*.csv")) or "—", language="text")


def explorer(root: Path):
    st.title("🔭 Explorer (Per-Asset Files)")

    assets = list_assets(root)
    if not assets:
        st.info("No asset folders detected in artefacts root.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        asset = st.selectbox("Asset", assets, index=0)
        freq = st.radio("Frequency", ["D", "W"], horizontal=True)

    asset_dir = root / asset
    st.caption(f"Folder: `{asset_dir}`")

    # Leaderboard
    st.markdown("### Leaderboard")
    lb = safe_read_csv(asset_dir / f"leaderboard_{freq}.csv")
    if lb.empty:
        st.info("No leaderboard available.")
    else:
        st.dataframe(lb, use_container_width=True, hide_index=True)

    # Baseline metrics for the asset
    st.markdown("### Baseline Metrics (asset)")
    base = safe_read_csv(asset_dir / f"metrics_baseline_{freq}.csv")
    if base.empty:
        st.info("No per-asset baseline metrics.")
    else:
        st.dataframe(base, use_container_width=True, hide_index=True)

    # Keyword metrics for the asset (optional, if present)
    st.markdown("### Keyword Metrics (asset)")
    kw_asset = safe_read_csv(asset_dir / f"metrics_keywords_{freq}.csv")
    if kw_asset.empty:
        st.info("No per-asset keyword metrics.")
    else:
        st.dataframe(kw_asset, use_container_width=True, hide_index=True)

    # Signals available
    with st.expander("📁 Signals available"):
        sigs = files_in(asset_dir, f"signals_*_{freq}.csv")
        st.code("\n".join(sigs) or "—", language="text")


# ============================================================
# App Entrypoint
# ============================================================

def main():
    root, checked = detect_artefacts_root()
    if root is None:
        st.error("Could not locate an artefacts folder.")
        st.write("Paths checked:")
        st.code("\n".join(checked) or "—")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Landing", "Explorer"])
    st.sidebar.caption("Paths checked:")
    st.sidebar.code("\n".join(checked), language="text")

    if page == "Landing":
        landing(root)
    else:
        explorer(root)


if __name__ == "__main__":
    # Page-wide Streamlit settings
    st.set_page_config(page_title="GT Markets – Demo App", layout="wide")
    main()

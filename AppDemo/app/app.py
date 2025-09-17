# Streamlit Demo App for GT Markets
# -------------------------------------------------------------
# Pages:
# 1) Landing
# 2) Compare (Baseline vs Keywords)
# 3) Keyword Lab
# 4) Signals & Audit
# 5) Diagnostics
#
# Data layout (ARTE_ROOT defaults to ./artefacts):
#   artefacts/<ASSET>/
#       metrics_baseline_D.csv
#       metrics_keywords_D.csv   (optional until Phase 2)
#       metrics_baseline_W.csv
#       metrics_keywords_W.csv   (optional until Phase 2)
#       signals_<STRATEGY>_D.csv (e.g., signals_S1_trend_D.csv)
#       signals_<STRATEGY>_W.csv
#       leaderboard_{D|W}.csv    (optional)
#       features_used.txt        (optional)
#       figs/*_{D|W}.png         (optional)
#       metadata.json            (optional)
#
# Keyword sets live under: ./keyword_sets/keyword_sets.json
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
ARTE_ROOT = Path(os.environ.get("ARTE_ROOT", "artefacts"))
KEYWORD_DIR = Path("keyword_sets")
KEYWORD_FILE = KEYWORD_DIR / "keyword_sets.json"

ASSET_ORDER = ["GOLD", "BTC", "OIL", "USDCNY"]
FREQ_LABELS = {"D": "Daily", "W": "Weekly"}

KEYWORD_DIR.mkdir(parents=True, exist_ok=True)

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
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
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
    # Auto-detect strategy IDs by scanning files: signals_<STRATEGY>_<FREQ>.csv
    root = ARTE_ROOT / asset
    if not root.exists():
        return []
    names = set()
    for p in root.glob(f"signals_*_{freq}.csv"):
        parts = p.stem.split("_")  # e.g., [signals, S1, trend, D]
        if len(parts) >= 3:
            names.add("_".join(parts[1:-1]))
    return sorted(names)


# -------------------------------------------------------------
# Keyword sets
# -------------------------------------------------------------
DEFAULT_KEYWORD_SETS = {
    "core_gold": {"asset": "GOLD", "freq": "D",
                   "keywords": ["gold price", "gold news", "safe haven", "interest rates"]},
    "core_btc": {"asset": "BTC", "freq": "D",
                  "keywords": ["bitcoin", "crypto fear", "halving", "btc etf"]},
}


def _load_keyword_sets() -> Dict[str, Dict]:
    if not KEYWORD_FILE.exists():
        KEYWORD_FILE.write_text(json.dumps(DEFAULT_KEYWORD_SETS, indent=2))
        return DEFAULT_KEYWORD_SETS
    try:
        return json.loads(KEYWORD_FILE.read_text())
    except Exception as e:
        st.warning(f"Failed to read keyword_sets.json — using defaults. Error: {e}")
        return DEFAULT_KEYWORD_SETS


def _save_keyword_sets(data: Dict[str, Dict]) -> None:
    KEYWORD_FILE.write_text(json.dumps(data, indent=2))


# -------------------------------------------------------------
# UI helpers
# -------------------------------------------------------------

def kpi_badge(label: str, value: Optional[float], fmt: str = "{:.4f}"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        st.metric(label, "–")
    else:
        st.metric(label, fmt.format(value))


def verdict_from_deltas(d_auc: Optional[float], d_acc: Optional[float]) -> str:
    if d_auc is None and d_acc is None:
        return "Inconclusive"
    scores = []
    if d_auc is not None and not np.isnan(d_auc):
        scores.append(np.sign(d_auc))
    if d_acc is not None and not np.isnan(d_acc):
        scores.append(np.sign(d_acc))
    if not scores:
        return "Inconclusive"
    s = np.mean(scores)
    if s > 0:
        return "Adds Value"
    if s < 0:
        return "No Value"
    return "Inconclusive"


def plot_delta_bars(d: Dict[str, float], title: str):
    fig, ax = plt.subplots()
    names = list(d.keys())
    vals = [d[k] for k in names]
    ax.bar(names, vals)
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Δ (Keywords - Baseline)")
    st.pyplot(fig)


def plot_signals_step(df: pd.DataFrame, time_col: Optional[str] = None,
                      signal_col: Optional[str] = None,
                      title: str = "Signals (last 20)"):
    # Try to infer date/signal columns
    if time_col is None:
        for c in ["Date", "date", "timestamp", "time"]:
            if c in df.columns:
                time_col = c
                break
    if signal_col is None:
        for c in ["signal", "Signal", "trade", "position", "Position"]:
            if c in df.columns:
                signal_col = c
                break
    if time_col is None or signal_col is None:
        st.info("""Cannot infer time/signal column to plot step chart.
Hint: expected columns like 'Date' and 'signal'.""")
        return
    df2 = df.tail(20).copy()
    fig, ax = plt.subplots()
    ax.step(df2[time_col], df2[signal_col], where="post")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(signal_col)
    plt.xticks(rotation=25)
    st.pyplot(fig)


# -------------------------------------------------------------
# Pages
# -------------------------------------------------------------

def page_landing(all_assets: List[str]):
    st.header("Landing")
    if not all_assets:
        st.error(f"No artefacts found at: {ARTE_ROOT.resolve()}")
        st.stop()

    c1, c2 = st.columns(2)
    asset = c1.selectbox("Asset", options=all_assets, index=0)
    freq = c2.radio("Frequency", options=list(FREQ_LABELS.keys()),
                    format_func=lambda x: FREQ_LABELS[x], horizontal=True)

    st.session_state["sel_asset"] = asset
    st.session_state["sel_freq"] = freq

    mb, mk = load_metrics(asset, freq)
    st.subheader("Availability Check")
    a1, a2 = st.columns(2)
    with a1:
        st.write("Baseline Metrics:"); st.success("Found") if mb is not None else st.warning("Missing")
    with a2:
        st.write("Keyword Metrics:"); st.success("Found") if mk is not None else st.info("Not yet generated (Phase 2)")

    avail = discover_strategies(asset, freq)
    st.write("Detected strategies:", ", ".join(avail) if avail else "(none)")


def page_compare(all_assets: List[str]):
    st.header("Compare — Baseline vs Keywords")
    if not all_assets:
        st.error("No assets available.")
        st.stop()

    asset = st.selectbox("Asset", all_assets, index=0)
    freq = st.radio("Frequency", list(FREQ_LABELS.keys()), horizontal=True,
                    format_func=lambda x: FREQ_LABELS[x])

    mb, mk = load_metrics(asset, freq)
    if mb is None and mk is None:
        st.warning("Both metrics tables are missing.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Baseline Metrics")
        if mb is not None:
            st.dataframe(mb)
        else:
            st.info("metrics_baseline_* not found")

    with c2:
        st.subheader("Keyword Metrics")
        if mk is not None:
            st.dataframe(mk)
        else:
            st.info("metrics_keywords_* not found (generate Phase 2)")

    st.subheader("KPI Δ (Keywords - Baseline)")
    d_auc = d_acc = None
    if mb is not None and mk is not None:
        def as_scalar(df, col):
            if col in df.columns:
                return float(pd.to_numeric(df[col], errors="coerce").mean())
            return None
        for auc_name in ["AUC", "auc", "roc_auc"]:
            b = as_scalar(mb, auc_name); k = as_scalar(mk, auc_name)
            if b is not None and k is not None:
                d_auc = k - b; break
        for acc_name in ["Accuracy", "accuracy", "acc"]:
            b = as_scalar(mb, acc_name); k = as_scalar(mk, acc_name)
            if b is not None and k is not None:
                d_acc = k - b; break
        c3, c4, c5 = st.columns(3)
        with c3: kpi_badge("Δ AUC", d_auc)
        with c4: kpi_badge("Δ Accuracy", d_acc)
        with c5: st.metric("Verdict", verdict_from_deltas(d_auc, d_acc))
        deltas = {}
        if d_auc is not None and not np.isnan(d_auc): deltas["AUC"] = d_auc
        if d_acc is not None and not np.isnan(d_acc): deltas["Accuracy"] = d_acc
        if deltas: plot_delta_bars(deltas, f"{asset} {FREQ_LABELS[freq]}")

    st.subheader("Equity Overlay (figs)")
    fig_paths = find_equity_figs(asset, freq)
    if fig_paths:
        cols = st.columns(min(3, len(fig_paths)))
        for i, p in enumerate(fig_paths[:6]):
            with cols[i % len(cols)]:
                st.image(str(p), caption=p.name, use_container_width=True)
    else:
        st.caption("No equity figures found under figs/.")

    st.subheader("Leaderboard (optional)")
    lb = load_leaderboard(asset, freq)
    if lb is not None:
        st.dataframe(lb)
    else:
        st.caption("leaderboard_* not provided.")


def page_keyword_lab(all_assets: List[str]):
    st.header("Keyword Lab")
    sets = _load_keyword_sets()

    st.subheader("Existing Keyword Sets")
    if sets:
        view_df = pd.DataFrame([
            {"name": k, "asset": v.get("asset"), "freq": v.get("freq"),
             "keywords": ", ".join(v.get("keywords", []))}
            for k, v in sets.items()
        ])
        st.dataframe(view_df)
    else:
        st.info("No keyword sets yet.")

    st.subheader("Create or Update a Set")
    with st.form("kw_form", clear_on_submit=False):
        name = st.text_input("Set name (unique)")
        asset = st.selectbox("Asset", options=all_assets or ASSET_ORDER)
        freq = st.radio("Frequency", options=list(FREQ_LABELS.keys()), horizontal=True,
                        format_func=lambda x: FREQ_LABELS[x])
        raw = st.text_area("Keywords (comma-separated)")
        submitted = st.form_submit_button("Save Set")
        if submitted and name:
            kws = [s.strip() for s in raw.split(",") if s.strip()]
            sets[name] = {"asset": asset, "freq": freq, "keywords": kws}
            _save_keyword_sets(sets)
            st.success(f"Saved set '{name}' with {len(kws)} keywords.")
            st.cache_data.clear()

    st.subheader("Delete a Set")
    if sets:
        del_name = st.selectbox("Choose set to delete", options=["(none)"] + list(sets.keys()))
        if del_name != "(none)" and st.button("Delete", type="secondary"):
            sets.pop(del_name, None)
            _save_keyword_sets(sets)
            st.success(f"Deleted {del_name}.")
            st.cache_data.clear()


def page_signals_audit(all_assets: List[str]):
    st.header("Signals & Audit")
    if not all_assets:
        st.error("No assets available.")
        st.stop()

    c1, c2 = st.columns(2)
    asset = c1.selectbox("Asset", all_assets, index=0)
    freq = c2.radio("Frequency", list(FREQ_LABELS.keys()), horizontal=True,
                    format_func=lambda x: FREQ_LABELS[x])

    available = discover_strategies(asset, freq)
    if not available:
        st.info("No signals_* files found for this selection.")
        st.stop()

    strat = st.selectbox("Strategy", available)
    st.session_state["sel_strategy"] = strat

    df_sig = load_signals(asset, strat, freq)
    if df_sig is None or df_sig.empty:
        st.warning("Signals file not found or empty.")
    else:
        st.subheader("Signals Preview (last 20)")
        plot_signals_step(df_sig, title=f"{asset} {strat} {FREQ_LABELS[freq]}")
        st.dataframe(df_sig.tail(20))
        st.download_button(
            "Download signals CSV",
            data=df_sig.to_csv(index=False).encode("utf-8"),
            file_name=f"signals_{asset}_{strat}_{freq}.csv",
        )

    st.subheader("features_used.txt")
    txt = load_features_text(asset)
    if txt:
        st.code(txt)
    else:
        st.caption("No features_used.txt found (optional).")


def page_diagnostics(all_assets: List[str]):
    st.header("Diagnostics")
    st.write("ARTE_ROOT:", str(ARTE_ROOT.resolve()))
    if not ARTE_ROOT.exists():
        st.error("ARTE_ROOT path does not exist.")
        return
    assets = [p for p in ARTE_ROOT.iterdir() if p.is_dir()]
    st.write("Assets found:", [a.name for a in assets] or "(none)")
    for a in assets:
        st.subheader(a.name)
        missing = []
        for req in ["metrics_baseline_D.csv", "metrics_baseline_W.csv"]:
            if not (a / req).exists():
                missing.append(req)
        for f in ["D", "W"]:
            if not list(a.glob(f"signals_*_{f}.csv")):
                missing.append(f"signals_*_{f}.csv (at least one)")
        if missing:
            st.warning(f"Missing: {missing}")
        else:
            st.success("All baseline files present ✅")


# -------------------------------------------------------------
# App entry
# -------------------------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(f"Artefacts root: {ARTE_ROOT.resolve()}")

    assets = list_assets(ARTE_ROOT)

    page = st.sidebar.radio(
        "Navigate",
        ("Landing", "Compare", "Keyword Lab", "Signals & Audit", "Diagnostics"),
        index=0,
    )

    if page == "Landing":
        page_landing(assets)
    elif page == "Compare":
        page_compare(assets)
    elif page == "Keyword Lab":
        page_keyword_lab(assets)
    elif page == "Signals & Audit":
        page_signals_audit(assets)
    elif page == "Diagnostics":
        page_diagnostics(assets)


if __name__ == "__main__":
    main()

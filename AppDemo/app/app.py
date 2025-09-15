
# Google Trends Financial Modelling — Demo App (read-only)
# Loads precomputed artefacts from AppDemo/artefacts and supports
# frequency-suffixed filenames (e.g., metrics_baseline_D.csv).

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Resolve paths relative to this file (…/AppDemo)
BASE_DIR   = Path(__file__).resolve().parent.parent
ARTE_ROOT  = BASE_DIR / "artefacts"
KEYWORD_FILE = BASE_DIR / "keyword_sets" / "keyword_sets.json"

ASSETS = ["BTC", "Gold", "Oil", "USDCNY"]
FREQS  = ["D", "W"]
STRATEGIES = ["conservative", "balanced", "aggressive"]

# ---------- Helpers ----------
def safe_read_csv(p: Path, **kwargs):
    try:
        return pd.read_csv(p, **kwargs)
    except Exception:
        return None

def metrics_paths(asset_dir: Path, freq: str):
    # supports frequency suffix (_D/_W)
    return (
        asset_dir / f"metrics_baseline_{freq}.csv",
        asset_dir / f"metrics_keywords_{freq}.csv",
    )

def equity_paths(asset_dir: Path, freq: str):
    # optional, only if you exported them
    return (
        asset_dir / f"equity_baseline_{freq}.csv",
        asset_dir / f"equity_keywords_{freq}.csv",
    )

def signals_path(asset_dir: Path, strategy: str, freq: str):
    return asset_dir / f"signals_{strategy}_{freq}.csv"

def load_features(asset_dir: Path):
    p = asset_dir / "features_used.txt"
    return p.read_text() if p.exists() else ""

def list_keyword_sets():
    if KEYWORD_FILE.exists():
        try:
            return json.loads(KEYWORD_FILE.read_text())
        except Exception:
            return []
    return []

def list_sets_for(asset: str, freq: str, saved_sets):
    return [s for s in saved_sets if s.get("pair")==asset and s.get("freq")==freq]

def style_delta(a, b):
    if a is None or b is None:
        return "—"
    try:
        return f"{(a-b):+.3f}"
    except Exception:
        return "—"

# ---------- UI ----------
st.set_page_config(page_title="Google Trends Financial Modelling — Demo", layout="wide")
st.title("Google Trends Financial Modelling — Demo")

c1, c2, c3 = st.columns(3)
with c1: asset = st.selectbox("Asset", ASSETS, index=0)
with c2: freq  = st.selectbox("Frequency", FREQS, index=0)
with c3: strategy = st.radio("Strategy", STRATEGIES, index=1)

# NOTE: artefacts are directly under asset folder (no D/W subfolder)
asset_dir = ARTE_ROOT / asset
if not asset_dir.exists():
    st.error(f"No artefacts folder found: {asset_dir}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Baseline vs Keywords", "Keyword Sets", "Signals & Audit"])

# ---------- Tab 1: Baseline vs Keywords ----------
with tab1:
    st.subheader("Baseline vs Keywords (precomputed)")
    mb, mk = metrics_paths(asset_dir, freq)
    baseline = safe_read_csv(mb)
    keywords = safe_read_csv(mk)

    if baseline is None and keywords is None:
        st.warning(f"No metrics found. Expected files:\n- {mb.name}\n- {mk.name}")
    else:
        frames = []
        if baseline is not None:
            t = baseline.copy()
            t["variant"] = "baseline"
            frames.append(t)
        if keywords is not None:
            t = keywords.copy()
            t["variant"] = "keywords"
            frames.append(t)
        allm = pd.concat(frames, ignore_index=True) if frames else None

        if allm is not None:
            # narrow to chosen strategy if the field exists
            view = allm
            if "strategy" in view.columns:
                view = view[view["strategy"]==strategy]
            cols = [c for c in ["variant","model","dataset","strategy","auc","accuracy","trades","sharpe","max_dd","turnover","run_id"] if c in view.columns]
            st.dataframe(view[cols] if cols else view, use_container_width=True)

            # quick deltas
            auc_kw=acc_kw=auc_bl=acc_bl=None
            if set(["variant","auc"]).issubset(view.columns):
                try: auc_kw = float(view.loc[view["variant"]=="keywords","auc"].iloc[0])
                except: pass
                try: auc_bl = float(view.loc[view["variant"]=="baseline","auc"].iloc[0])
                except: pass
            if set(["variant","accuracy"]).issubset(view.columns):
                try: acc_kw = float(view.loc[view["variant"]=="keywords","accuracy"].iloc[0])
                except: pass
                try: acc_bl = float(view.loc[view["variant"]=="baseline","accuracy"].iloc[0])
                except: pass

            cA,cB,cC = st.columns(3)
            with cA: st.metric("Δ AUC (KW - Base)", style_delta(auc_kw, auc_bl))
            with cB: st.metric("Δ Accuracy (KW - Base)", style_delta(acc_kw, acc_bl))
            verdict = "Inconclusive"
            if (auc_kw is not None and auc_bl is not None):
                if (auc_kw - auc_bl) >= 0.02: verdict = "Adds Value"
                elif (auc_kw - auc_bl) <= -0.02: verdict = "No Value"
            with cC: st.metric("Verdict", verdict)

    # optional equity overlay
    eb, ek = equity_paths(asset_dir, freq)
    e_base = safe_read_csv(eb, parse_dates=["date"]) if eb.exists() else None
    e_kw   = safe_read_csv(ek, parse_dates=["date"]) if ek.exists() else None
    if e_base is not None or e_kw is not None:
        st.caption("Equity curve overlay (if available)")
        fig, ax = plt.subplots(figsize=(8,3))
        if e_base is not None: ax.plot(e_base["date"], e_base["equity"], label="baseline")
        if e_kw   is not None: ax.plot(e_kw["date"],   e_kw["equity"],   label="keywords")
        ax.set_xlabel("Date"); ax.set_ylabel("Equity"); ax.legend()
        st.pyplot(fig)

# ---------- Tab 2: Keyword Sets ----------
with tab2:
    st.subheader("Keyword Sets (precomputed variants)")
    saved_sets = list_keyword_sets()
    sets_for_sel = list_sets_for(asset, freq, saved_sets)
    if not sets_for_sel:
        st.info("No entries for this asset/frequency in keyword_sets.json.")
    else:
        names = [s["name"] for s in sets_for_sel]
        choice = st.selectbox("Select keyword set", names)
        set_dir = asset_dir / choice
        metrics_path = set_dir / f"metrics_{freq}.csv" if (set_dir / f"metrics_{freq}.csv").exists() else set_dir / "metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            if "strategy" in df.columns:
                st.dataframe(df[df["strategy"]==strategy], use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        else:
            st.warning(f"metrics not found in selected set: {metrics_path.name}")

        core_dir = asset_dir / "core"
        core_metrics = (core_dir / f"metrics_{freq}.csv") if (core_dir / f"metrics_{freq}.csv").exists() else (core_dir / "metrics.csv")
        if choice!="core" and core_metrics.exists() and metrics_path.exists():
            st.caption("Δ vs core (AUC / Accuracy)")
            df_sel  = pd.read_csv(metrics_path)
            df_core = pd.read_csv(core_metrics)
            a_sel   = df_sel.loc[df_sel.get("strategy","")==strategy, ["auc","accuracy"]].mean(numeric_only=True)
            a_core  = df_core.loc[df_core.get("strategy","")==strategy, ["auc","accuracy"]].mean(numeric_only=True)
            c1,c2 = st.columns(2)
            with c1: st.metric("Δ AUC (set - core)", style_delta(a_sel.get("auc"), a_core.get("auc")))
            with c2: st.metric("Δ Accuracy (set - core)", style_delta(a_sel.get("accuracy"), a_core.get("accuracy")))

# ---------- Tab 3: Signals & Audit ----------
with tab3:
    st.subheader("Signals")
    sig_path = signals_path(asset_dir, strategy, freq)
    sigs = safe_read_csv(sig_path, parse_dates=["date"])
    if sigs is not None:
        st.dataframe(sigs.tail(20), use_container_width=True)
        st.download_button("Download signals CSV", sigs.to_csv(index=False), file_name=sig_path.name)
    else:
        st.info(f"No signals file for this selection: {sig_path.name}")

    st.subheader("Features Used (audit)")
    text = load_features(asset_dir)
    st.text(text if text else "features_used.txt not found.")

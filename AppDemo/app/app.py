
# Google Trends Financial Modelling — Demo App (read-only)

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- Config ---
ARTE_ROOT = Path("artefacts")
KEYWORD_FILE = Path("keyword_sets/keyword_sets.json")
ASSETS = ["BTC", "Gold", "Oil", "USDCNY"]
FREQS = ["D", "W"]
STRATEGIES = ["conservative", "balanced", "aggressive"]

# --- Helpers ---
def safe_read_csv(p: Path, **kwargs):
    try: return pd.read_csv(p, **kwargs)
    except Exception: return None

def load_metrics_pair(base_dir: Path):
    return safe_read_csv(base_dir / "metrics_baseline.csv"), safe_read_csv(base_dir / "metrics_keywords.csv")

def load_equity_pair(base_dir: Path):
    e_base = safe_read_csv(base_dir / "equity_baseline.csv", parse_dates=["date"])
    e_kw   = safe_read_csv(base_dir / "equity_keywords.csv", parse_dates=["date"])
    return e_base, e_kw

def load_signals(base_dir: Path, strategy: str):
    return safe_read_csv(base_dir / f"signals_{strategy}.csv", parse_dates=["date"])

def load_features(base_dir: Path):
    p = base_dir / "features_used.txt"
    return p.read_text() if p.exists() else ""

def list_keyword_sets():
    if KEYWORD_FILE.exists():
        try: return json.loads(KEYWORD_FILE.read_text())
        except Exception: return []
    return []

def list_sets_for(asset: str, freq: str, saved_sets):
    return [s for s in saved_sets if s.get("pair")==asset and s.get("freq")==freq]

def style_delta(a, b):
    if a is None or b is None: return "—"
    try: return f"{(a-b):+.3f}"
    except Exception: return "—"

# --- UI ---
st.set_page_config(page_title="GT Financial Modelling Demo", layout="wide")
st.title("Google Trends Financial Modelling — Demo")

c1, c2, c3 = st.columns(3)
with c1: asset = st.selectbox("Asset", ASSETS, index=0)
with c2: freq = st.selectbox("Frequency", FREQS, index=0)
with c3: strategy = st.radio("Strategy", STRATEGIES, index=1)

base_dir = ARTE_ROOT / asset / freq
if not base_dir.exists():
    st.error("No artefacts found for this selection. Check artefacts folder structure.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Baseline vs Keywords", "Keyword Sets", "Signals & Audit"])

# --- Tab 1: Baseline vs Keywords ---
with tab1:
    st.subheader("Baseline vs Keywords (precomputed)")
    baseline, keywords = load_metrics_pair(base_dir)

    if baseline is None and keywords is None:
        st.warning("No metrics_baseline.csv / metrics_keywords.csv found.")
    else:
        frames = []
        if baseline is not None: 
            t = baseline.copy(); t["variant"]="baseline"; frames.append(t)
        if keywords is not None: 
            t = keywords.copy(); t["variant"]="keywords"; frames.append(t)
        allm = pd.concat(frames, ignore_index=True) if frames else None

        if allm is not None and "strategy" in allm.columns:
            spotlight = allm[allm["strategy"]==strategy].copy()
            cols = [c for c in ["variant","model","dataset","strategy","auc","accuracy","trades","sharpe","max_dd","turnover","run_id"] if c in spotlight.columns]
            st.dataframe(spotlight[cols], use_container_width=True)

            auc_kw=acc_kw=auc_bl=acc_bl=None
            if set(["variant","auc"]).issubset(spotlight.columns):
                try: auc_kw = float(spotlight.loc[spotlight["variant"]=="keywords","auc"].iloc[0])
                except: pass
                try: auc_bl = float(spotlight.loc[spotlight["variant"]=="baseline","auc"].iloc[0])
                except: pass
            if set(["variant","accuracy"]).issubset(spotlight.columns):
                try: acc_kw = float(spotlight.loc[spotlight["variant"]=="keywords","accuracy"].iloc[0])
                except: pass
                try: acc_bl = float(spotlight.loc[spotlight["variant"]=="baseline","accuracy"].iloc[0])
                except: pass

            cA,cB,cC = st.columns(3)
            with cA: st.metric("Δ AUC (KW - Base)", style_delta(auc_kw, auc_bl))
            with cB: st.metric("Δ Accuracy (KW - Base)", style_delta(acc_kw, acc_bl))
            verdict = "Inconclusive"
            if (auc_kw is not None and auc_bl is not None):
                if (auc_kw - auc_bl) >= 0.02: verdict = "Adds Value"
                elif (auc_kw - auc_bl) <= -0.02: verdict = "No Value"
            with cC: st.metric("Verdict", verdict)

    e_base, e_kw = load_equity_pair(base_dir)
    if e_base is not None or e_kw is not None:
        st.caption("Equity curve overlay (if available)")
        fig, ax = plt.subplots(figsize=(8,3))
        if e_base is not None: ax.plot(e_base["date"], e_base["equity"], label="baseline")
        if e_kw   is not None: ax.plot(e_kw["date"],   e_kw["equity"],   label="keywords")
        ax.set_xlabel("Date"); ax.set_ylabel("Equity"); ax.legend()
        st.pyplot(fig)

# --- Tab 2: Keyword Sets ---
with tab2:
    st.subheader("Keyword Sets (precomputed variants)")
    saved_sets = list_keyword_sets()
    sets_for_sel = list_sets_for(asset, freq, saved_sets)
    if not sets_for_sel:
        st.info("No entries for this asset/freq in keyword_sets.json.")
    else:
        names = [s["name"] for s in sets_for_sel]
        choice = st.selectbox("Select keyword set", names)
        set_dir = base_dir / choice
        metrics_path = set_dir / "metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            if "strategy" in df.columns:
                st.dataframe(df[df["strategy"]==strategy], use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        else:
            st.warning("metrics.csv not found in selected set.")

        core_dir = base_dir / "core"
        if choice!="core" and (core_dir / "metrics.csv").exists() and metrics_path.exists():
            st.caption("Δ vs core (AUC / Accuracy)")
            df_sel  = pd.read_csv(metrics_path)
            df_core = pd.read_csv(core_dir / "metrics.csv")
            a_sel   = df_sel.loc[df_sel.get("strategy","")==strategy, ["auc","accuracy"]].mean(numeric_only=True)
            a_core  = df_core.loc[df_core.get("strategy","")==strategy, ["auc","accuracy"]].mean(numeric_only=True)
            c1,c2 = st.columns(2)
            with c1: st.metric("Δ AUC (set - core)", style_delta(a_sel.get("auc"), a_core.get("auc")))
            with c2: st.metric("Δ Accuracy (set - core)", style_delta(a_sel.get("accuracy"), a_core.get("accuracy")))

# --- Tab 3: Signals & Audit ---
with tab3:
    st.subheader("Signals")
    sigs = load_signals(base_dir, strategy)
    if sigs is not None:
        st.dataframe(sigs.tail(20), use_container_width=True)
        st.download_button(
            "Download signals CSV",
            sigs.to_csv(index=False),
            file_name=f"signals_{asset}_{freq}_{strategy}.csv"
        )
    else:
        st.info("No signals file for this selection/strategy.")

    st.subheader("Features Used (audit)")
    text = load_features(base_dir)
    st.text(text if text else "features_used.txt not found.")

# AppDemo/app/app.py
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def _find_data_dir() -> Path:
    candidates = [
        Path(__file__).parent / "data",                  # AppDemo/app/data
        Path(__file__).parent.parent / "data",           # AppDemo/data
        Path.cwd() / "AppDemo" / "data",
        Path.cwd() / "data",
        Path("/content/gt-markets/AppDemo/data"),
        Path("/mount/src/gt-markets/AppDemo/data"),
        Path("/mnt/data"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path.cwd()

DATA_DIR = _find_data_dir()

def _latest(glob: str) -> Path | None:
    hits = sorted(DATA_DIR.glob(glob))
    return hits[-1] if hits else None

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mm = _latest("model_metrics*.csv")
    sm = _latest("strategy_metrics*.csv")
    sj = _latest("signals_snapshot*.json") or _latest("signals*.json") or (DATA_DIR / "signals_snapshot.json")

    if mm is None or sm is None:
        st.error("Missing data in AppDemo/data: model_metrics*.csv and/or strategy_metrics*.csv")
        return pd.DataFrame(), pd.DataFrame(), {}

    model_metrics = pd.read_csv(mm)
    strategy_metrics = pd.read_csv(sm)

    signals_map: Dict[str, Any] = {}
    if sj and sj.exists():
        try:
            signals_map = json.loads(sj.read_text())
        except Exception:
            try:
                signals_map = ast.literal_eval(sj.read_text())
            except Exception:
                signals_map = {}

    return model_metrics, strategy_metrics, signals_map

model_metrics, strategy_metrics, signals_map = load_inputs()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _dash_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.nan, None], "-")

def _f2(x) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "-"

def _pct(x) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "-"

def _parse_params_to_dict(p) -> dict:
    if p is None: return {}
    if isinstance(p, dict): return p
    if isinstance(p, str):
        s = p.strip()
        if not s: return {}
        for loader in (json.loads, ast.literal_eval):
            try:
                d = loader(s)
                return d if isinstance(d, dict) else {}
            except Exception:
                pass
    try:
        if pd.isna(p): return {}
    except Exception:
        pass
    return {}

def _model_short_dirret(name) -> str:
    """RF_cls -> 'RF (Direction)'; XGB_reg -> 'XGB (Return)'."""
    try:
        if name is None or (isinstance(name, float) and math.isnan(name)) or pd.isna(name):
            n = ""
        else:
            n = str(name)
    except Exception:
        n = ""
    n = n.strip()
    if not n:
        return "-"
    m = re.match(r"([A-Za-z0-9]+)[_\- ]?(cls|reg)?", n)
    base = m.group(1).upper() if m else n.upper()
    kind = (m.group(2) or "").lower() if m else ""
    tgt = "Direction" if kind == "cls" else ("Return" if kind == "reg" else "")
    return f"{base} ({tgt})" if tgt else base

def _detect_indicator_family(rule_text: str, params: dict):
    t = (rule_text or "").upper()
    if "MACD" in t: return "MACD"
    if "RSI" in t: return "RSI"
    if "CROSS" in t or " MA " in t or "MOVING AVERAGE" in t: return "MA"
    ind = str(params.get("ta") or params.get("indicator") or "").upper()
    if "MACD" in ind: return "MACD"
    if "RSI" in ind: return "RSI"
    if "MA" in ind or "SMA" in ind or "EMA" in ind: return "MA"
    return ""

def _explain_setup(rule_text: str, params: dict) -> Tuple[str, str]:
    fam = _detect_indicator_family(rule_text, params)
    if fam == "MA":
        fast = str(params.get("lo") or params.get("fast") or params.get("short") or "10")
        slow = str(params.get("hi") or params.get("slow") or params.get("long") or "50")
        return f"MA cross ({fast}/{slow})", "Short-term momentum crossing long-term. Bullish when fast > slow; bearish when fast < slow."
    if fam == "RSI":
        per = str(params.get("period") or params.get("window") or "14")
        th = ""
        if params.get("hi"): th = f" > {params['hi']}"
        elif params.get("lo"): th = f" < {params['lo']}"
        return f"RSI({per}){th}", "Momentum oscillator. >70 overbought; <30 oversold."
    if fam == "MACD":
        w = str(params.get("window") or params.get("macd") or params.get("ta_window") or "12-26-9")
        parts = w.replace(" ", "").replace("_","-").split("-")
        label = f"MACD cross ({parts[0]}/{parts[1]}, {parts[2]})" if len(parts) >= 3 else "MACD cross (12/26, 9)"
        return label, "Momentum turning point. Above signal bullish; below bearish."
    rt = (rule_text or "").strip() or "Custom rule"
    return rt, "Rule-based signal."

# ---------------------------------------------------------------------
# Simple Mode (kept minimal & safe)
# ---------------------------------------------------------------------
ASSETS = sorted(strategy_metrics["asset"].dropna().unique().tolist() if "asset" in strategy_metrics.columns else ["BTC","OIL"])
FREQS  = ["D","W"]

def simple_mode():
    st.header("Show me why")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        asset = st.selectbox("Asset", ASSETS, index=0)
    with c2:
        freq = st.radio("Frequency", FREQS, horizontal=True, index=1)
    with c3:
        dataset = st.radio("Dataset", ["Market only", "Market + Keywords"], horizontal=True, index=1)

    st.info("Simple Mode preview. Switch to **Advanced** for full analysis.")
    if not model_metrics.empty:
        mm = model_metrics.copy()
        if "asset" in mm.columns: mm = mm[mm["asset"] == asset]
        if "freq" in mm.columns: mm = mm[mm["freq"] == freq]
        st.dataframe(_dash_na(mm.head(12)), use_container_width=True)

# ---------------------------------------------------------------------
# Keyword Explorer (redesigned)
# ---------------------------------------------------------------------
def keyword_explorer_tab(models_df: pd.DataFrame, asset: str, freq: str):
    st.subheader("Keyword Explorer")

    req = {"asset","freq","dataset","metric","value"}
    if not req.issubset(set(models_df.columns)):
        st.info("Keyword Explorer expects tidy metrics with columns: 'asset','freq','dataset','metric','value'.")
        return

    df = models_df.copy()
    df = df[(df["asset"] == asset) & (df["freq"] == freq)]
    if df.empty:
        st.info("No metrics for this asset/frequency.")
        return

    pivot = df.pivot_table(index="metric", columns="dataset", values="value", aggfunc="mean")

    # identify dataset columns robustly
    def _find_base():
        for c in pivot.columns:
            lc = c.lower()
            if "market" in lc and "keyword" not in lc:
                return c
        return None
    def _find_kw():
        for c in pivot.columns:
            if "keyword" in c.lower():
                return c
        return None

    col_base = _find_base()
    col_kw   = _find_kw()

    if col_base is None or col_kw is None:
        st.info("Need both datasets (Market only, Market + Keywords) to compute deltas.")
        st.dataframe(_dash_na(pivot.reset_index()), use_container_width=True)
        return

    pivot["Δ"] = pivot[col_kw] - pivot[col_base]

    # KPI cards: Δ AUC and Δ MAE
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Δ AUC", _f2(pivot.loc["AUC","Δ"]) if "AUC" in pivot.index else "-")
    with c2:
        st.metric("Δ MAE", _f2(pivot.loc["MAE","Δ"]) if "MAE" in pivot.index else "-")

    # Two compact tables
    def _mk(metrics: List[str]) -> pd.DataFrame:
        rows=[]
        for m in metrics:
            if m in pivot.index:
                rows.append({
                    "Metric": m,
                    "Market only": _f2(pivot.loc[m, col_base]),
                    "Market + Keywords": _f2(pivot.loc[m, col_kw]),
                    "Δ": _f2(pivot.loc[m, "Δ"])
                })
        return pd.DataFrame(rows)

    st.markdown("### Direction metrics")
    st.dataframe(_dash_na(_mk(["AUC","Accuracy","F1"])), use_container_width=True)

    st.markdown("### Return metrics")
    st.dataframe(_dash_na(_mk(["MAE","RMSE","Spearman"])), use_container_width=True)

# ---------------------------------------------------------------------
# Strategy Insights (short model names, All option, Sharpe-desc)
# ---------------------------------------------------------------------
def strategy_insights_tab(strategies: pd.DataFrame, asset: str, freq: str, dataset_code: str):
    st.subheader("Strategy Insights")

    if strategies.empty:
        st.info("No strategies available.")
        return

    df = strategies.copy()

    # Normalize/derive columns
    if "model_label" in df.columns:
        df.rename(columns={"model_label": "Model"}, inplace=True)
    if "Model" not in df.columns and "family_id" in df.columns:
        df["Model"] = df["family_id"]
    if "Model" not in df.columns:
        df["Model"] = "-"

    df["Model"] = df["Model"].apply(_model_short_dirret)

    if "Setup" not in df.columns:
        if "rule" in df.columns:
            df["Setup"] = df.apply(lambda r: _explain_setup(str(r.get("rule","")), _parse_params_to_dict(r.get("params")))[0], axis=1)
        else:
            df["Setup"] = df["params"].apply(lambda s: _explain_setup("", _parse_params_to_dict(s))[0])

    for col, val in [("asset", asset), ("freq", freq), ("dataset", dataset_code)]:
        if col in df.columns:
            df = df[df[col] == val]

    # Controls: Model with "All"
    model_options = sorted(df["Model"].dropna().unique().tolist())
    model_choices = ["All"] + model_options
    sel_mod = st.multiselect("Model", model_choices, default=["All"])
    active_models = set(model_options) if (not sel_mod or "All" in sel_mod) else set([m for m in sel_mod if m != "All"])

    f = df[df["Model"].isin(list(active_models))].copy()

    # sort by Sharpe desc if exists
    if "sharpe" in f.columns:
        f = f.sort_values(by="sharpe", ascending=False)

    show_cols = [c for c in ["Model","Setup","sharpe","max_dd","ann_return"] if c in f.columns]
    show = f[show_cols].copy()
    show.rename(columns={"sharpe":"Sharpe","max_dd":"Max DD","ann_return":"Annual Return"}, inplace=True)
    st.dataframe(_dash_na(show), use_container_width=True)

# ---------------------------------------------------------------------
# Context (snapshot now; demo fallback; blue BUY/uptrend/positive)
# ---------------------------------------------------------------------
def context_tab(signals_map: dict, asset: str, freq: str, dataset: str, strategies: pd.DataFrame = None):
    import numpy as np

    st.subheader("Context")

    # Find snapshot for this asset/freq
    snap = None
    if isinstance(signals_map, dict):
        for k in (f"{asset}-{freq}", f"{asset}_{freq}", asset):
            if k in signals_map:
                snap = signals_map[k]
                break

    # Demo fallback if missing
    if not isinstance(snap, dict) or not snap:
        snap = {
            "signal": "BUY",
            "confidence": 72,
            "chg_d": 0.012,            # +1.2% today
            "chg_w": 0.035,            # +3.5% this week
            "rsi": 56,
            "macd_hist": 0.004,
            "vol_pctile": 0.55,
            "trend_state": "Uptrend",
            "top_keywords": ["ETF inflow", "halving", "institutional"],
        }

    # Helpers
    def fmt_pct(x):
        try: return f"{100*float(x):.1f}%"
        except: return "-"

    def bucket_vol(p):
        try:
            p = float(p)
            return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"
        except: return "-"

    def bucket_rsi(v):
        try:
            v = float(v)
            if v >= 70: return f"{int(v)} (Overbought)"
            if v <= 30: return f"{int(v)} (Oversold)"
            return f"{int(v)} (Neutral)"
        except: return "-"

    def macd_txt(h):
        try:
            h = float(h)
            return "Bullish" if h > 0 else "Bearish" if h < 0 else "Flat"
        except: return "-"

    def trend_label():
        t = str(snap.get("trend_state") or "").strip()
        if t: return t
        sig = str(snap.get("signal") or "").upper()
        if sig == "BUY":  return "Uptrend"
        if sig == "SELL": return "Downtrend"
        return "Sideways"

    # Styles and custom renderer (avoid metric truncation)
    st.markdown("""
    <style>
      .metric-label{font-size:.85rem;color:#9aa0a6;margin-bottom:4px}
      .metric-big{font-size:36px;font-weight:700;line-height:1.1;margin:0}
      .text-blue{color:#0B5FFF!important}
      .text-red{color:#ef4444!important}
    </style>
    """, unsafe_allow_html=True)

    def _metric(col, label, value, color=None):
        cls = f" text-{color}" if color else ""
        with col:
            st.markdown(
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-big{cls}">{value}</div>',
                unsafe_allow_html=True
            )

    # Signal & Confidence (no “As of”)
    signal_txt = str(snap.get("signal") or "-").upper()
    conf_val   = snap.get("confidence")
    sig_color  = "blue" if signal_txt == "BUY" else ("red" if signal_txt == "SELL" else None)

    c1, c2 = st.columns(2)
    _metric(c1, "Signal", signal_txt, sig_color)
    _metric(c2, "Confidence", f"{conf_val:.0f}%" if isinstance(conf_val, (int, float)) else "-", None)

    # Today's context tiles
    dchg      = snap.get("chg_d")
    wchg      = snap.get("chg_w")
    rsi       = snap.get("rsi")
    macd_hist = snap.get("macd_hist")
    vol_pctl  = snap.get("vol_pctile")
    tstate    = trend_label()

    today_color = "blue" if isinstance(dchg,(int,float)) and dchg > 0 else ("red" if isinstance(dchg,(int,float)) and dchg < 0 else None)
    week_color  = "blue" if isinstance(wchg,(int,float)) and wchg > 0 else ("red" if isinstance(wchg,(int,float)) and wchg < 0 else None)
    trend_color = "blue" if tstate.lower() == "uptrend" else ("red" if tstate.lower() == "downtrend" else None)

    c1, c2, c3, c4, c5 = st.columns(5)
    _metric(c1, "Today",   fmt_pct(dchg), today_color)
    _metric(c2, "1W",      fmt_pct(wchg), week_color)
    _metric(c3, "Trend",   tstate, trend_color)
    _metric(c4, "RSI(14)", bucket_rsi(rsi), None)
    _metric(c5, "MACD",    macd_txt(macd_hist), None)

    # Keywords & short narrative
    kws = snap.get("top_keywords") or snap.get("keywords") or []
    if isinstance(kws, (list, tuple)) and kws:
        st.caption("Notable keywords: " + ", ".join(map(str, kws)))

    parts = []
    if isinstance(dchg,(int,float)): parts.append(f"{fmt_pct(dchg)} today")
    if isinstance(wchg,(int,float)): parts.append(f"{fmt_pct(wchg)} this week")
    if parts:
        st.write(
            f"**Summary:** {asset} {', '.join(parts)}; trend **{tstate}**, "
            f"momentum **RSI {bucket_rsi(rsi)}**, volatility **{bucket_vol(vol_pctl)}**."
        )

# ---------------------------------------------------------------------
# Advanced Mode
# ---------------------------------------------------------------------
def advanced_mode(model_metrics: pd.DataFrame, strategy_metrics: pd.DataFrame, signals_map: Dict[str, Any]):
    st.header("Show me why")

    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        asset = st.selectbox("Asset", ASSETS, index=0)
    with c2:
        freq = st.radio("Frequency", FREQS, horizontal=True, index=1)
    with c3:
        dataset = st.radio("Dataset", ["Market only", "Market + Keywords"], horizontal=True, index=1)

    tabs = st.tabs(["Model Comparison","Keyword Explorer","Strategy Insights","Context"])

    with tabs[0]:
        st.write("Model Comparison (placeholder).")
        if not model_metrics.empty:
            mm = model_metrics.copy()
            if "asset" in mm.columns: mm = mm[mm["asset"]==asset]
            if "freq" in mm.columns: mm = mm[mm["freq"]==freq]
            st.dataframe(_dash_na(mm.head(20)), use_container_width=True)

    with tabs[1]:
        keyword_explorer_tab(model_metrics, asset, freq)

    with tabs[2]:
        strategy_insights_tab(strategy_metrics, asset, freq, dataset)

    with tabs[3]:
        context_tab(signals_map, asset, freq, dataset, strategies=strategy_metrics)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Markets Demo", layout="wide")
    mode = st.sidebar.radio("Mode", ["Simple","Advanced"], index=1)
    st.session_state["mode"] = mode

    if mode == "Simple":
        if "simple_mode" in globals() and callable(simple_mode):
            simple_mode()
        else:
            st.warning("Simple Mode unavailable in this build.")
    else:
        if "advanced_mode" in globals() and callable(advanced_mode):
            advanced_mode(model_metrics, strategy_metrics, signals_map)
        else:
            st.error("Advanced Mode unavailable in this build.")

if __name__ == "__main__":
    main()

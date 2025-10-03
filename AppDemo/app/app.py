# AppDemo/app/app.py  —  100% hard-coded demo, no external files

from __future__ import annotations
import json
import math
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Demo Data (hard-coded, plausible values)
# ============================================================

ASSETS = ["BTC", "OIL", "GOLD", "SPX"]
FREQS  = ["D", "W"]
DATASETS = ["base", "ext"]   # base = Market only, ext = Market + Keywords

# ---- Model metrics (per asset/freq/dataset/model/task) ----
# CLS: higher better (AUC/Acc/F1),  REG: lower better (MAE/RMSE); Spearman ~ [-0.2..0.7]
_DEMO_MODEL_ROWS: List[Dict[str, Any]] = []
rng = np.random.default_rng(42)

def _mk_cls_row(asset, freq, ds, model, auc, acc, f1):
    return dict(asset=asset, freq=freq, dataset=ds, task="CLS", model=model,
                auc=auc, accuracy=acc, f1=f1, mae=np.nan, rmse=np.nan, spearman=np.nan)

def _mk_reg_row(asset, freq, ds, model, mae, rmse, sp):
    return dict(asset=asset, freq=freq, dataset=ds, task="REG", model=model,
                auc=np.nan, accuracy=np.nan, f1=np.nan, mae=mae, rmse=rmse, spearman=sp)

# Seed sensible differentials: ext gives slight uplift for CLS; lower error for REG.
for asset in ASSETS:
    for freq in FREQS:
        # base (market only)
        _DEMO_MODEL_ROWS += [
            _mk_cls_row(asset, freq, "base", "XGB_cls", 0.71, 0.66, 0.63),
            _mk_cls_row(asset, freq, "base", "RF_cls",  0.69, 0.64, 0.61),
            _mk_cls_row(asset, freq, "base", "MLP_cls", 0.68, 0.63, 0.60),
            _mk_cls_row(asset, freq, "base", "LR_cls",  0.66, 0.61, 0.58),

            _mk_reg_row(asset, freq, "base", "LSTM_reg", 0.013, 0.020, 0.36),
            _mk_reg_row(asset, freq, "base", "GRU_reg",  0.014, 0.021, 0.33),
            _mk_reg_row(asset, freq, "base", "XGB_reg",  0.015, 0.022, 0.30),
        ]
        # ext (market + keywords) — small improvements
        _DEMO_MODEL_ROWS += [
            _mk_cls_row(asset, freq, "ext", "XGB_cls", 0.76, 0.70, 0.68),
            _mk_cls_row(asset, freq, "ext", "RF_cls",  0.73, 0.68, 0.65),
            _mk_cls_row(asset, freq, "ext", "MLP_cls", 0.71, 0.67, 0.64),
            _mk_cls_row(asset, freq, "ext", "LR_cls",  0.69, 0.65, 0.62),

            _mk_reg_row(asset, freq, "ext", "LSTM_reg", 0.011, 0.018, 0.43),
            _mk_reg_row(asset, freq, "ext", "GRU_reg",  0.012, 0.019, 0.41),
            _mk_reg_row(asset, freq, "ext", "XGB_reg",  0.013, 0.020, 0.39),
        ]

model_metrics = pd.DataFrame(_DEMO_MODEL_ROWS)

# ---- Strategy metrics (Sharpe/MaxDD/Annual Return + simple Setup) ----
def _setup(indicator, window=None, hi=None, lo=None):
    base = indicator + (f" ({window})" if window else "")
    if hi is not None and lo is not None:
        return base + f" · hi/lo {hi}/{lo}"
    return base

_DEMO_STRAT_ROWS: List[Dict[str, Any]] = []
for asset in ASSETS:
    for freq in FREQS:
        for ds in DATASETS:
            # Make ext a bit better
            uplift = 0.15 if ds == "ext" else 0.0
            _DEMO_STRAT_ROWS += [
                dict(asset=asset, freq=freq, dataset=ds, family="HYBRID_CONF",
                     family_id="xgb_cls", params=json.dumps({"model":"XGB_cls","hi":0.55,"lo":0.45,"ta":"TA_MAcross_10-50"}),
                     sharpe=1.10+uplift, max_dd=-0.12, ann_return=0.28+uplift,
                     Rule=_setup("MA cross", "10/50", 0.55, 0.45), Model="XGB_cls"),

                dict(asset=asset, freq=freq, dataset=ds, family="HYBRID_CONF",
                     family_id="rf_cls", params=json.dumps({"model":"RF_cls","hi":0.55,"lo":0.45,"ta":"TA_RSI14_70-30"}),
                     sharpe=0.95+uplift, max_dd=-0.15, ann_return=0.22+uplift,
                     Rule=_setup("RSI(14) > 70 / < 30"), Model="RF_cls"),

                dict(asset=asset, freq=freq, dataset=ds, family="HYBRID_CONF",
                     family_id="lstm_reg", params=json.dumps({"model":"LSTM_reg","ta":"TA_MACD_12-26-9"}),
                     sharpe=1.05+uplift, max_dd=-0.14, ann_return=0.26+uplift,
                     Rule=_setup("MACD cross", "12/26, 9"), Model="LSTM_reg"),

                dict(asset=asset, freq=freq, dataset=ds, family="HYBRID_CONF",
                     family_id="gru_reg", params=json.dumps({"model":"GRU_reg","ta":"TA_MAcross_20-100"}),
                     sharpe=0.98+uplift, max_dd=-0.16, ann_return=0.24+uplift,
                     Rule=_setup("MA cross", "20/100"), Model="GRU_reg"),
            ]

strategy_metrics = pd.DataFrame(_DEMO_STRAT_ROWS)

# ---- Signals (for Context + Simple cards) — deterministic per (asset,freq) ----
def seeded_snapshot(asset: str, freq: str) -> Dict[str, Any]:
    key = f"{asset}-{freq}".upper()
    seed = (abs(hash(key)) % (2**32 - 1)) or 12345
    r = np.random.default_rng(seed)
    sig = r.choice(["BUY","SELL","HOLD"], p=[0.5,0.3,0.2])
    trend = {"BUY":"Uptrend", "SELL":"Downtrend", "HOLD":"Sideways"}[sig]
    conf = int(r.integers(58, 86))
    dmove = r.normal(0.006, 0.012)
    wmove = r.normal(0.018, 0.025)
    if sig == "BUY":
        dmove = abs(dmove); wmove = abs(wmove)
    elif sig == "SELL":
        dmove = -abs(dmove); wmove = -abs(wmove)
    else:
        dmove *= 0.4; wmove *= 0.4
    rsi = int(np.clip(r.normal(54 if sig=="BUY" else 46 if sig=="SELL" else 50, 7), 15, 85))
    macd = float(r.normal(0.004 if sig=="BUY" else -0.004 if sig=="SELL" else 0.0, 0.002))
    volp = float(np.clip(r.uniform(0.2, 0.85), 0, 1))
    kw_pool = {
        "BTC":["ETF inflow","halving","institutional","on-chain"],
        "OIL":["OPEC supply","inventory draw","geopolitics","refinery"],
        "GOLD":["real yields","safe haven","ETF holdings","USD"],
        "SPX":["earnings","rate cuts","AI capex","buybacks"],
    }
    kws = r.choice(kw_pool.get(asset.upper(), ["momentum","macro","flows","positioning"]), size=3, replace=False).tolist()
    return dict(asset=asset, freq=freq, dataset="ext",
                signal=sig, confidence=conf, chg_d=dmove, chg_w=wmove,
                rsi=rsi, macd_hist=macd, vol_pctile=volp, trend_state=trend, top_keywords=kws)

signals_map: Dict[str, Any] = {f"{a}-{f}": seeded_snapshot(a,f) for a in ASSETS for f in FREQS}

# ============================================================
# Helpers / Formatting
# ============================================================
def _dash_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.nan, None], "–")

def _f2(x) -> str:
    try: return f"{float(x):.2f}"
    except: return "–"

def _pct(x) -> str:
    try: return f"{100*float(x):.2f}%"
    except: return "–"

def _model_short_dirret(name) -> str:
    s = str(name or "").strip()
    m = re.match(r"([A-Za-z0-9]+)[_\- ]?(cls|reg)?", s)
    base = m.group(1).upper() if m else s.upper()
    kind = (m.group(2) or "").lower() if m else ""
    tgt = "Direction" if kind == "cls" else ("Return" if kind == "reg" else "")
    return f"{base} ({tgt})" if tgt else base

# ============================================================
# Simple Mode
# ============================================================
def reason_text(sig: Dict[str, Any]) -> str:
    decision = (sig.get("signal") or "").upper()
    freq = "Weekly" if str(sig.get("freq")).upper() == "W" else "Daily"
    core = f"Model suggests {decision.title()}"
    return f"{core}. ({freq} · Market + Keywords)"

def _pill(text: str, tone: str) -> str:
    colors = {"good": ("#1e9e49", "#e7f6ec"),
              "bad": ("#d9534f", "#fdeaea"),
              "neutral": ("#666", "#efefef")}
    fg, bg = colors.get(tone, colors["neutral"])
    return f"<span style='background:{bg};color:{fg};padding:3px 10px;border-radius:999px;font-weight:600;margin-left:6px;'>{text}</span>"

def simple_card(asset: str, sig: Dict[str, Any]):
    decision = (sig.get("signal") or "").upper()
    idx = int(sig.get("confidence", 0))
    border = "2px solid #1e9e49" if decision == "BUY" else "2px solid #d9534f" if decision == "SELL" else "1px solid #999"
    dec_p = _pill(decision, "good" if decision=="BUY" else "bad" if decision=="SELL" else "neutral")
    str_p = _pill(f"Strength {idx}", "neutral")
    reason = reason_text(sig)
    st.markdown(f"""
    <div style="border:{border};border-radius:12px;padding:12px 14px;margin-bottom:10px;background:#121212;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:1.05rem;font-weight:700;color:#ffffff;">{asset}</div>
        <div>{dec_p}{str_p}</div>
      </div>
      <div style="color:#cfcfcf;margin-top:6px;">{reason}.</div>
    </div>
    """, unsafe_allow_html=True)

def simple_mode():
    st.title("What should I trade now?")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        min_strength = st.slider("Min strength", 0, 100, 60)
    with c2:
        st.caption("Click an asset to switch to Advanced (use top navbar).")

    buy, sell = [], []
    for a in ASSETS:
        for f in FREQS:
            s = signals_map[f"{a}-{f}"]
            if int(s["confidence"]) < min_strength:
                continue
            if s["signal"] == "BUY": buy.append((a, s))
            elif s["signal"] == "SELL": sell.append((a, s))

    col_b, col_s = st.columns(2)
    with col_b:
        st.header("BUY")
        if not buy: st.caption("No BUY signals under current filters.")
        for a, s in sorted(buy, key=lambda t: -t[1]["confidence"])[:4]:
            simple_card(a, s)
    with col_s:
        st.header("SELL")
        if not sell: st.caption("No SELL signals under current filters.")
        for a, s in sorted(sell, key=lambda t: -t[1]["confidence"])[:4]:
            simple_card(a, s)

# ============================================================
# Tabs — Advanced
# ============================================================
def model_comparison_tab(models: pd.DataFrame, asset: str, freq: str, dataset_code: str):
    st.markdown("### Model Comparison")
    md = models[(models["asset"]==asset) & (models["freq"]==freq) & (models["dataset"]==dataset_code)]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Direction (AUC ↑)**")
        cls = md[md["task"]=="CLS"].copy()
        if cls.empty:
            st.caption("No direction models for this selection.")
        else:
            cls["Model"] = cls["model"].apply(_model_short_dirret)
            tbl = (cls.groupby("Model", as_index=False)[["auc","accuracy","f1"]].max()
                     .sort_values("auc", ascending=False)
                     .reset_index(drop=True))
            st.dataframe(_dash_na(tbl.round(3)), use_container_width=True)

    with c2:
        st.markdown("**Return (MAE ↓)**")
        reg = md[md["task"]=="REG"].copy()
        if reg.empty:
            st.caption("No return models for this selection.")
        else:
            reg["Model"] = reg["model"].apply(_model_short_dirret)
            tbl = (reg.groupby("Model", as_index=False)[["mae","rmse","spearman"]].mean()
                     .sort_values("mae", ascending=True)
                     .reset_index(drop=True))
            st.dataframe(_dash_na(tbl.round(4)), use_container_width=True)

def keyword_explorer_tab(models: pd.DataFrame, asset: str, freq: str):
    st.subheader("Keyword Explorer")

    df = models[(models["asset"]==asset) & (models["freq"]==freq)].copy()
    if df.empty:
        st.info("No metrics for this asset/frequency.")
        return

    # collapse across models by dataset (best for CLS, best-for-lower for REG)
    def agg_cls(m):
        return pd.Series({"AUC": m["auc"].max(), "Accuracy": m["accuracy"].max(), "F1": m["f1"].max()})
    def agg_reg(m):
        return pd.Series({"MAE": m["mae"].min(), "RMSE": m["rmse"].min(), "Spearman": m["spearman"].max()})

    cls = df[df["task"]=="CLS"].groupby("dataset").apply(agg_cls).reset_index()
    reg = df[df["task"]=="REG"].groupby("dataset").apply(agg_reg).reset_index()

    # KPI deltas (ext - base)
    def kget(df_, metric):
        try:
            row = df_.set_index("dataset").loc[["base","ext"], metric]
            return float(row["ext"] - row["base"])
        except Exception:
            return np.nan

    c1, c2 = st.columns(2)
    st.metric("Δ AUC", _f2(kget(cls, "AUC")), help="Market+Keywords minus Market only")
    st.metric("Δ MAE", _f2(-kget(reg, "MAE")), help="Lower MAE is better, so sign flipped")

    st.markdown("#### Direction metrics")
    st.dataframe(_dash_na(cls.round(3)), use_container_width=True)

    st.markdown("#### Return metrics")
    st.dataframe(_dash_na(reg.round(4)), use_container_width=True)

def strategy_insights_tab(strategies: pd.DataFrame, asset: str, freq: str, dataset_code: str):
    st.subheader("Strategy Insights")
    df = strategies[(strategies["asset"]==asset) & (strategies["freq"]==freq) & (strategies["dataset"]==dataset_code)].copy()

    if df.empty:
        st.info("No strategies for this selection.")
        return

    # Short model names + simple Setup (from Rule)
    df["Model"] = df["Model"].apply(_model_short_dirret) if "Model" in df.columns else df["family_id"].apply(_model_short_dirret)
    if "Setup" not in df.columns:
        df["Setup"] = df["Rule"]

    # Model selector with "All"
    models_avail = sorted(df["Model"].dropna().unique().tolist())
    choices = ["All"] + models_avail
    sel = st.multiselect("Model", choices, default=["All"])
    active = set(models_avail) if ("All" in sel or not sel) else set([s for s in sel if s!="All"])
    f = df[df["Model"].isin(active)].copy()

    # Sort by Sharpe desc
    f = f.sort_values("sharpe", ascending=False)

    show = f[["Model","Setup","sharpe","max_dd","ann_return"]].rename(
        columns={"sharpe":"Sharpe","max_dd":"Max DD","ann_return":"Annual Return"}
    )
    st.dataframe(_dash_na(show.round(2)), use_container_width=True)

def context_tab(signals: Dict[str, Any], asset: str, freq: str, dataset_code: str):
    """Demo-only snapshot that varies by (asset, freq)."""
    st.subheader("Context")
    snap = signals.get(f"{asset}-{freq}") or {}

    def fmt_pct(x): 
        try: return f"{100*float(x):.1f}%"
        except: return "–"
    def bucket_vol(p):
        try:
            p=float(p); return "Low" if p<0.33 else "Moderate" if p<0.66 else "High"
        except: return "–"
    def bucket_rsi(v):
        try:
            v=float(v); 
            return f"{int(v)} (Overbought)" if v>=70 else f"{int(v)} (Oversold)" if v<=30 else f"{int(v)} (Neutral)"
        except: return "–"
    def macd_txt(h):
        try:
            h=float(h); return "Bullish" if h>0 else "Bearish" if h<0 else "Flat"
        except: return "–"

    # style (high contrast, minimal fill; black default numbers; blue/red highlights)
    st.markdown("""
    <style>
      .kpi{border:1px solid rgba(255,255,255,.18); border-radius:14px; padding:12px 14px;
           background: rgba(255,255,255,0.02); height:100%;
           display:flex; flex-direction:column; gap:6px}
      .kpi .label{font-size:.85rem; color:#b9c3cf}
      .kpi .value{font-size:26px; font-weight:800; line-height:1.15; color:#ffffff}
      .blue{color:#2D7CFF!important}
      .red{color:#FF5C5C!important}
      .pill{display:inline-block; padding:4px 10px; border:1px solid rgba(255,255,255,.18);
            border-radius:999px; margin-right:6px; margin-top:6px; font-size:.85rem; color:#d5dbe3}
    </style>
    """, unsafe_allow_html=True)

    def card(col, label, value, color_cls=""):
        with col:
            st.markdown(
                f'<div class="kpi"><div class="label">{label}</div>'
                f'<div class="value {color_cls}">{value}</div></div>',
                unsafe_allow_html=True
            )

    sig = (snap.get("signal") or "-").upper()
    conf = snap.get("confidence")
    dchg = snap.get("chg_d"); wchg = snap.get("chg_w")
    rsi  = snap.get("rsi");  macd = snap.get("macd_hist"); volp = snap.get("vol_pctile")
    trend = snap.get("trend_state") or ("Uptrend" if sig=="BUY" else "Downtrend" if sig=="SELL" else "Sideways")
    sig_color   = "blue" if sig=="BUY" else "red" if sig=="SELL" else ""
    today_color = "blue" if isinstance(dchg,(int,float)) and dchg>0 else "red" if isinstance(dchg,(int,float)) and dchg<0 else ""
    week_color  = "blue" if isinstance(wchg,(int,float)) and wchg>0 else "red" if isinstance(wchg,(int,float)) and wchg<0 else ""
    trend_color = "blue" if trend=="Uptrend" else "red" if trend=="Downtrend" else ""

    r1 = st.columns(4)
    card(r1[0], "Signal", sig, sig_color)
    card(r1[1], "Confidence", f"{int(conf)}%" if isinstance(conf,(int,float)) else "–")
    card(r1[2], "Trend", trend, trend_color)
    card(r1[3], "RSI(14)", bucket_rsi(rsi))

    r2 = st.columns(4)
    card(r2[0], "Today", fmt_pct(dchg), today_color)
    card(r2[1], "1W",    fmt_pct(wchg), week_color)
    card(r2[2], "MACD",  macd_txt(macd))
    card(r2[3], "Volatility", bucket_vol(volp))

    kws = snap.get("top_keywords") or []
    if kws:
        st.markdown("".join([f'<span class="pill">{str(k)}</span>' for k in kws]), unsafe_allow_html=True)

    st.write(
        f"**Summary:** {asset} {fmt_pct(dchg)} today, {fmt_pct(wchg)} this week; "
        f"trend **{trend}**, RSI **{bucket_rsi(rsi)}**, vol **{bucket_vol(volp)}**."
    )

# ============================================================
# Advanced Mode (selectors + tabs)
# ============================================================
def advanced_mode(models: pd.DataFrame, strategies: pd.DataFrame, signals: Dict[str,Any]):
    st.title("Show me why")

    # Build adaptive choices so we never land on empty slices
    assets = sorted(models["asset"].unique().tolist())
    asset = st.selectbox("Asset", assets, index=0)

    avail_freqs = sorted(models.loc[models["asset"]==asset,"freq"].unique().tolist())
    freq = st.radio("Frequency", avail_freqs, horizontal=True, index=0)

    label_to_code = {"Market only":"base", "Market + Keywords":"ext"}
    code_to_label = {v:k for k,v in label_to_code.items()}
    avail_ds_codes = sorted(models.loc[(models["asset"]==asset)&(models["freq"]==freq),"dataset"].unique().tolist())
    ds_options = [code_to_label.get(c,c) for c in avail_ds_codes]
    ds_label = st.radio("Dataset", ds_options, horizontal=True, index=0)
    dataset_code = label_to_code.get(ds_label, ds_label)

    tabs = st.tabs(["Model Comparison","Keyword Explorer","Strategy Insights","Context"])
    with tabs[0]:
        model_comparison_tab(models, asset, freq, dataset_code)
    with tabs[1]:
        keyword_explorer_tab(models, asset, freq)
    with tabs[2]:
        strategy_insights_tab(strategies, asset, freq, dataset_code)
    with tabs[3]:
        context_tab(signals, asset, freq, dataset_code)

# ============================================================
# App entry
# ============================================================
def main():
    st.set_page_config(page_title="Markets Demo (Hard-coded)", layout="wide")
    mode = st.sidebar.radio("Mode", ["Simple","Advanced"], index=1)
    if mode == "Simple":
        simple_mode()
    else:
        advanced_mode(model_metrics, strategy_metrics, signals_map)

if __name__ == "__main__":
    main()

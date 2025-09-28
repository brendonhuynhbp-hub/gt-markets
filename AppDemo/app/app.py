
# app.py — Demo-first app: Simple (cards) + Advanced (4 tabs, streamlined + deduped Model Comparison)
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Financial Trends App Demo", layout="wide")

# =========================================================
# File discovery & loading (prefers signals_demo.json)
# =========================================================
CANDIDATE_DIRS = [
    Path("."), Path("./AppDemo/data"), Path("./app/data"), Path("./data"),
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent / "AppDemo" / "data",
    Path(__file__).resolve().parent.parent / "data",
]

def find_file(filename: str) -> Path | None:
    for d in CANDIDATE_DIRS:
        p = d / filename
        if p.exists():
            return p
    for p in Path(".").resolve().rglob(filename):
        return p
    return None

def load_snapshot() -> dict:
    demo = find_file("signals_demo.json")
    snap = find_file("signals_snapshot.json")
    path = demo or snap
    if not path:
        st.error("Missing signals file: add AppDemo/data/signals_demo.json (preferred) "
                 "or AppDemo/data/signals_snapshot.json.")
        st.stop()
    with open(path, "r") as f:
        data = json.load(f)
    return data

@st.cache_data(show_spinner=True)
def load_all():
    models_p = find_file("model_metrics.csv")
    strategies_p = find_file("strategy_metrics.csv")
    if not models_p or not strategies_p:
        st.error("Missing: model_metrics.csv and/or strategy_metrics.csv "
                 "(expected under AppDemo/data/).")
        st.stop()
    models = pd.read_csv(models_p)
    strategies = pd.read_csv(strategies_p)
    # normalize headers
    models.columns = [c.strip().lower() for c in models.columns]
    strategies.columns = [c.strip().lower() for c in strategies.columns]
    return models, strategies

signals_snapshot = load_snapshot()
model_metrics, strategy_metrics = load_all()

# =========================================================
# Friendly names & helpers
# =========================================================
MODEL_MAP = {
    "XGB": "XGBoost", "XGB_cls": "XGBoost", "XGB_reg": "XGBoost",
    "GRU": "GRU", "GRU_cls": "GRU", "GRU_reg": "GRU",
    "LSTM": "LSTM", "LSTM_cls": "LSTM", "LSTM_reg": "LSTM",
    "RF": "Random Forest", "RF_cls": "Random Forest", "RF_reg": "Random Forest",
    "LR_cls": "Logistic Regression", "LR_reg": "Linear Regression",
    "MLP": "MLP", "MLP_cls": "MLP", "MLP_reg": "MLP",
}
def friendly_model_name(x: str) -> str:
    if not isinstance(x, str): return ""
    base = x.split("_")[0]
    return MODEL_MAP.get(x, MODEL_MAP.get(base, x))

def ta_friendly_label(ta: str) -> str:
    """TA_RSI21_70-30 -> 'RSI(21) 70/30'; TA_MAcross_50-200 -> 'MA 50/200'"""
    if not isinstance(ta, str) or not ta:
        return ""
    if ta.startswith("TA_RSI"):
        try:
            core = ta.replace("TA_RSI", "")
            period, bands = core.split("_")
            hi, lo = bands.split("-")
            return f"RSI({int(period)}) {hi}/{lo}"
        except Exception:
            return "RSI confirmation"
    if ta.startswith("TA_MAcross"):
        try:
            core = ta.replace("TA_MAcross_", "")
            fast, slow = core.split("-")
            return f"MA {fast}/{slow}"
        except Exception:
            return "MA crossover"
    return ta

@st.cache_data
def tidy_strategy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Expand JSON params and attach friendly labels."""
    out = df.copy()
    cols = {"model_param": [], "hi": [], "lo": [], "ta": []}
    for s in out.get("params", pd.Series([None]*len(out))).fillna(""):
        try:
            d = json.loads(s) if isinstance(s, str) else {}
        except Exception:
            d = {}
        cols["model_param"].append(d.get("model", ""))
        cols["hi"].append(d.get("hi", ""))
        cols["lo"].append(d.get("lo", ""))
        cols["ta"].append(d.get("ta", ""))

    for k,v in cols.items():
        out[k] = v

    out["model_label"] = out["model_param"].apply(friendly_model_name)
    out["ta_label"] = out["ta"].apply(ta_friendly_label)

    def rule_label(r):
        bits = []
        if r.get("ta_label"): bits.append(r["ta_label"])
        if r.get("hi") != "" and r.get("lo") != "":
            bits.append(f"hi/lo {r['hi']}/{r['lo']}")
        return " · ".join(bits)
    out["rule_label"] = out[["ta_label","hi","lo"]].apply(lambda r: rule_label(r), axis=1)
    return out

def clamp(v, lo=0, hi=100): 
    try:
        return max(lo, min(hi, int(v)))
    except Exception:
        return lo

def dash_na(df: pd.DataFrame) -> pd.DataFrame:
    """Render NaN/None as '–' for display."""
    return df.replace([np.nan, None], "–")

# =========================================================
# Gauges (compact)
# =========================================================
def _bands(decision: str):
    d = (decision or "").upper()
    if d == "SELL":
        return [{"range":[0,40],"color":"green"},
                {"range":[40,60],"color":"yellow"},
                {"range":[60,100],"color":"red"}]
    return [{"range":[0,40],"color":"red"},
            {"range":[40,60],"color":"yellow"},
            {"range":[60,100],"color":"green"}]

def gauge(value: int, decision: str, title: str, show_number: bool, height: int = 150):
    v = clamp(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number" if show_number else "gauge",
        value=v,
        number={'font': {'size': 28}} if show_number else None,
        title={"text": title, "font": {"size": 12}},
        gauge={
            "axis": {"range": [0,100]},
            "bar": {"color": "black", "thickness": 0.25},
            "steps": _bands(decision),
            "threshold": {"line":{"color":"black","width":3}, "thickness":0.7, "value": v},
        },
    ))
    fig.update_layout(height=height, margin=dict(l=2,r=2,t=8,b=2))
    return fig

# =========================================================
# Reason generation for Simple Mode (clear confidence)
# =========================================================
def best_ta_confirmation(strategies_df: pd.DataFrame, asset: str, freq: str, dataset: str):
    sd = strategies_df[(strategies_df["asset"]==asset) &
                       (strategies_df["freq"]==freq) &
                       (strategies_df["dataset"]==dataset) &
                       (strategies_df["family"].str.upper()=="HYBRID_CONF")]
    if sd.empty: return None
    row = sd.sort_values("sharpe", ascending=False).iloc[0]
    try:
        params = json.loads(row["params"]) if isinstance(row["params"], str) else {}
    except Exception:
        params = {}
    return ta_friendly_label(params.get("ta",""))

def keywords_uplift_label(models_df: pd.DataFrame, asset: str, freq: str):
    m = models_df[(models_df["asset"]==asset) & (models_df["freq"]==freq)]
    if m.empty: return None
    cls_b = m[(m["dataset"]=="base") & (m["task"]=="CLS")]
    cls_e = m[(m["dataset"]=="ext")  & (m["task"]=="CLS")]
    reg_b = m[(m["dataset"]=="base") & (m["task"]=="REG")]
    reg_e = m[(m["dataset"]=="ext")  & (m["task"]=="REG")]
    delta, metric = None, None
    if not cls_b.empty and not cls_e.empty:
        delta = float(cls_e["auc"].max() - cls_b["auc"].max()); metric = "AUC"
    elif not reg_b.empty and not reg_e.empty:
        delta = float(reg_b["mae"].min() - reg_e["mae"].min()); metric = "MAE"
    else:
        return None
    if delta is None: return None
    if metric == "AUC":
        if delta >= 0.01: return f"Keywords help (+{delta:.2f} AUC)"
        if delta <= -0.01: return f"Keywords hurt ({delta:.2f} AUC)"
        return "Keywords neutral"
    # MAE (lower better)
    if delta >= 0.001: return f"Keywords help (−{delta:.3f} MAE)"
    if delta <= -0.001: return f"Keywords hurt (+{abs(delta):.3f} MAE)"
    return "Keywords neutral"

def reason_text(sig: dict) -> str:
    """
    Plain-English reason for Simple Mode.
    CLS: 'Model predicts ↑/↓ with XX% confidence — strong chance of up/downside.'
    REG: 'Model projects ±X.X% expected return — up/downside bias.'
    """
    decision = (sig.get("decision") or "").upper()
    fam = (sig.get("family") or "").upper()
    freq = "Weekly" if sig.get("freq") == "W" else "Daily"
    ds = (sig.get("dataset") or "").upper()

    if fam == "CLS" and sig.get("prob") is not None and sig.get("thr") is not None:
        prob_up = float(sig["prob"]); thr = float(sig["thr"]); prob_dn = 1.0 - prob_up
        if prob_up >= thr:
            core = f"Model predicts ↑ with {prob_up:.0%} confidence — strong chance of upside"
        elif prob_dn >= thr:
            core = f"Model predicts ↓ with {prob_dn:.0%} confidence — strong chance of downside"
        else:
            core = f"Model confidence {prob_up:.0%} — no clear signal"
    elif fam == "REG" and sig.get("pred_ret") is not None:
        pr = float(sig["pred_ret"]); bias = "upside" if pr >= 0 else "downside"
        core = f"Model projects {pr:+.2%} expected return — {bias} bias"
    else:
        core = f"Model suggests {decision.title()}"

    return f"{core}. ({freq} · {ds})"

# =========================================================
# Simple Mode — cards + gauges + "View details"
# =========================================================
def _pill(text: str, tone: str):
    colors = {
        "good":   ("#1e9e49","#e7f6ec"),
        "bad":    ("#d9534f","#fdeaea"),
        "neutral":("#666","#efefef")
    }
    fg, bg = colors.get(tone, colors["neutral"])
    return f"<span style='background:{bg};color:{fg};padding:3px 10px;border-radius:999px;font-weight:600;margin-left:6px;'>{text}</span>"

def card(asset: str, sig: dict, models_df: pd.DataFrame, strategies_df: pd.DataFrame, show_gauges: bool):
    decision = (sig.get("decision") or "").upper()
    idx = clamp(sig.get("index", 0))
    ta = best_ta_confirmation(strategies_df, asset, sig.get("freq",""), sig.get("dataset",""))
    kw = keywords_uplift_label(models_df, asset, sig.get("freq",""))
    reason_core = reason_text(sig)

    dec_pill = _pill("BUY" if decision=="BUY" else "SELL", "good" if decision=="BUY" else "bad")
    str_pill = _pill(f"Strength {idx}", "neutral")
    ta_txt = f"; {ta} confirmation" if ta else ""
    kw_txt = f"; {kw}" if kw else ""
    border = "2px solid #1e9e49" if decision=="BUY" else "2px solid #d9534f"

    st.markdown(
        f"""
        <div style="border:{border};border-radius:12px;padding:12px 14px;margin-bottom:10px;background:#121212;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="font-size:1.05rem;font-weight:700;">{asset}</div>
            <div>{dec_pill}{str_pill}</div>
          </div>
          <div style="color:#cfcfcf;margin-top:6px;">{reason_core}{ta_txt}{kw_txt}.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if show_gauges:
        st.plotly_chart(gauge(idx, decision, f"{asset} · {sig.get('freq','')} · {sig.get('dataset','')}", False),
                        use_container_width=True)
    if st.button(f"View details — {asset} ({sig.get('freq','')}/{sig.get('dataset','').upper()})", key=f"view_{asset}_{sig.get('freq','')}_{sig.get('dataset','')}"):
        st.session_state["mode"] = "Advanced"
        st.session_state["asset"] = asset
        st.session_state["freq"] = sig.get("freq","D")
        st.session_state["dataset"] = (sig.get("dataset") or "ext").lower()
        st.session_state["tab_index"] = 0  # Model Comparison

def simple_mode(signals: dict):
    st.title("What should I trade now?")
    c1,c2,c3 = st.columns([1.1,1,1.3])
    with c1: min_strength = st.slider("Min strength", 0, 100, 0)
    with c2: show_gauges = st.toggle("Show gauges", value=True)
    with c3:
        st.caption("Click a card to drill into Advanced.")

    buy, sell = [], []
    for key, s in signals.items():
        asset = s.get("asset") or key.split("-")[0]
        if not asset: continue
        if s.get("index") is None: continue
        if clamp(s["index"]) < min_strength: continue
        dec = (s.get("decision") or "").upper()
        if dec == "BUY": buy.append((asset, s))
        elif dec == "SELL": sell.append((asset, s))

    col_b, col_s = st.columns(2)
    with col_b:
        st.header("BUY")
        if not buy: st.caption("No BUY signals under current filters.")
        for a,s in sorted(buy, key=lambda t: -clamp(t[1]["index"]))[:4]:
            card(a, s, model_metrics, strategy_metrics, show_gauges)

    with col_s:
        st.header("SELL")
        if not sell: st.caption("No SELL signals under current filters.")
        for a,s in sorted(sell, key=lambda t: -clamp(t[1]["index"]))[:4]:
            card(a, s, model_metrics, strategy_metrics, show_gauges)

# =========================================================
# Advanced Mode — 4 tabs (Model Comparison streamlined + deduped)
# =========================================================
def model_comparison_tab(models: pd.DataFrame, asset: str, freq: str, dataset_code: str):
    md = models[(models["asset"]==asset) & (models["freq"]==freq) & (models["dataset"]==dataset_code)]

    # ---- Summary strip: Best AUC / Best MAE (unchanged logic) ----
    c1,c2 = st.columns([1,1])
    with c1:
        cls = md[md["task"]=="CLS"]
        if cls.empty:
            st.markdown("**Best AUC:** –")
        else:
            best_row = cls.loc[cls["auc"].idxmax()]
            st.markdown(f"**Best AUC:** `{best_row['auc']:.3f}` ({friendly_model_name(best_row['model'])})")
    with c2:
        reg = md[md["task"]=="REG"]
        if reg.empty:
            st.markdown("**Best MAE:** –")
        else:
            best_row = reg.loc[reg["mae"].idxmin()]
            st.markdown(f"**Best MAE:** `{best_row['mae']:.3f}` ({friendly_model_name(best_row['model'])})")

    st.divider()

    # ---- Direction Prediction (AUC) ----
    cls_tbl = md[md["task"]=="CLS"].copy()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### Direction Prediction")
        if cls_tbl.empty:
            st.caption("No direction models for this selection.")
        else:
            cls_tbl["Model"] = cls_tbl["model"].apply(friendly_model_name)
            # One row per model → keep BEST (max) AUC
            cls_tbl = (cls_tbl.groupby("Model", as_index=False)["auc"]
                              .max()
                              .rename(columns={"auc":"AUC"})
                              .sort_values("AUC", ascending=False)
                              .reset_index(drop=True))
            st.dataframe(dash_na(cls_tbl[["Model","AUC"]].round(3)), use_container_width=True)

    # ---- Return Prediction (MAE lower is better) ----
    reg_tbl = md[md["task"]=="REG"].copy()
    with c4:
        st.markdown("### Return Prediction")
        if reg_tbl.empty:
            st.caption("No return models for this selection.")
        else:
            reg_tbl["Model"] = reg_tbl["model"].apply(friendly_model_name)
            # One row per model → keep BEST (min) MAE
            reg_tbl = (reg_tbl.groupby("Model", as_index=False)["mae"]
                               .min()
                               .rename(columns={"mae":"MAE (lower is better)"})
                               .sort_values("MAE (lower is better)", ascending=True)
                               .reset_index(drop=True))
            st.dataframe(dash_na(reg_tbl[["Model","MAE (lower is better)"]].round(3)), use_container_width=True)

def keyword_explorer_tab(models: pd.DataFrame, asset: str, freq: str):
    st.subheader("Keyword effect (Market only vs Market + Keywords)")
    cls_base = models[(models["asset"]==asset)&(models["freq"]==freq)&(models["dataset"]=="base")&(models["task"]=="CLS")]
    cls_ext  = models[(models["asset"]==asset)&(models["freq"]==freq)&(models["dataset"]=="ext") &(models["task"]=="CLS")]
    reg_base = models[(models["asset"]==asset)&(models["freq"]==freq)&(models["dataset"]=="base")&(models["task"]=="REG")]
    reg_ext  = models[(models["asset"]==asset)&(models["freq"]==freq)&(models["dataset"]=="ext") &(models["task"]=="REG")]
    rows=[]
    if not cls_base.empty and not cls_ext.empty:
        rows.append({"Metric":"AUC (direction)","Market only":cls_base["auc"].max(),"Market + Keywords":cls_ext["auc"].max(),
                     "Uplift":cls_ext["auc"].max()-cls_base["auc"].max()})
    if not reg_base.empty and not reg_ext.empty:
        rows.append({"Metric":"MAE (return)","Market only":reg_base["mae"].min(),"Market + Keywords":reg_ext["mae"].min(),
                     "Uplift":reg_base["mae"].min()-reg_ext["mae"].min()})
    if rows:
        df = pd.DataFrame(rows).round(3)
        st.dataframe(dash_na(df), use_container_width=True)
    else:
        st.info("No comparison available for this selection.")

def strategy_insights_tab(strategies: pd.DataFrame, asset: str, freq: str, dataset_code: str):
    sd = strategies[(strategies["asset"]==asset) & (strategies["freq"]==freq) & (strategies["dataset"]==dataset_code)]
    if sd.empty:
        st.warning("No strategy metrics for this selection."); return
    sd_tidy = tidy_strategy_metrics(sd)

    fam_order = ["HYBRID_CONF", "ML", "TA", "HYBRID"]
    sd_tidy["fam_sort"] = sd_tidy["family"].apply(lambda x: fam_order.index(x) if x in fam_order else 99)

    cols = ["family","model_label","rule_label","sharpe","maxdd","ann_return"]
    df = (sd_tidy
          .sort_values(["fam_sort","sharpe"], ascending=[True, False])
          [cols].rename(columns={
              "family":"Family",
              "model_label":"Model",
              "rule_label":"Rule",
              "sharpe":"Sharpe",
              "maxdd":"Max DD",
              "ann_return":"Annual Return"
          })
          .round({"Sharpe":3, "Max DD":3, "Annual Return":3})
          .reset_index(drop=True))

    st.dataframe(dash_na(df), use_container_width=True)

def context_tab(signals: dict, asset: str, freq: str, dataset_code: str):
    sig = None
    for k,v in signals.items():
        a = v.get("asset") or k.split("-")[0]
        f = v.get("freq") or (k.split("-")[1] if "-" in k else "")
        d = (v.get("dataset") or "").lower()
        if a==asset and f==freq and (d==dataset_code.lower()):
            sig = v; break
    if not sig:
        st.info("No snapshot entry for this selection."); return
    st.markdown("#### Sentiment gauge")
    st.caption("Index reflects model confidence / predicted return strength (0–100).")
    st.plotly_chart(gauge(sig.get("index",0), sig.get("decision","SELL"), f"{asset} · {freq} · {dataset_code.upper()}", True),
                    use_container_width=True)

def advanced_mode(models: pd.DataFrame, strategies: pd.DataFrame, signals: dict):
    st.title("Show me why")

    # Sticky selectors
    assets = sorted(models["asset"].dropna().unique().tolist())
    asset_default = st.session_state.get("asset", assets[0] if assets else "")
    freq_default = st.session_state.get("freq","W")
    dataset_default = st.session_state.get("dataset","ext")

    # Dataset labels: Market only (base) / Market + Keywords (ext)
    label_to_code = {"Market only":"base", "Market + Keywords":"ext"}
    code_to_label = {v:k for k,v in label_to_code.items()}
    initial_label = code_to_label.get(dataset_default, "Market + Keywords")

    colA,colB,colC = st.columns([1.2,0.9,1.3])
    with colA:
        asset = st.selectbox("Asset", assets, index=(assets.index(asset_default) if asset_default in assets else 0))
    with colB:
        freq = st.radio("Frequency", ["D","W"], horizontal=True, index=(["D","W"].index(freq_default) if freq_default in ["D","W"] else 1))
    with colC:
        ds_label = st.radio("Dataset", ["Market only","Market + Keywords"], horizontal=True,
                            index=(["Market only","Market + Keywords"].index(initial_label)))
        dataset_code = label_to_code[ds_label]

    tabs = st.tabs(["Model Comparison","Keyword Explorer","Strategy Insights","Context"])
    with tabs[0]: model_comparison_tab(models, asset, freq, dataset_code)
    with tabs[1]: keyword_explorer_tab(models, asset, freq)
    with tabs[2]: strategy_insights_tab(strategies, asset, freq, dataset_code)
    with tabs[3]: context_tab(signals, asset, freq, dataset_code)

# =========================================================
# MAIN ROUTER
# =========================================================
mode = st.sidebar.radio("Mode", ["Simple","Advanced"], index=(0 if st.session_state.get("mode","Simple")=="Simple" else 1))
if mode == "Simple":
    st.session_state["mode"] = "Simple"
    simple_mode(signals_snapshot)
else:
    st.session_state["mode"] = "Advanced"
    advanced_mode(model_metrics, strategy_metrics, signals_snapshot)

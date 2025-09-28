import re
# AppDemo/app/app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DATA_DIRS = [
    Path(__file__).resolve().parent.parent / "data",  # AppDemo/data relative to this file
    Path("AppDemo/data"),                              # when running from repo root
    Path("./data"),                                    # fallback
]

# ------------------------------------------------------------
# Helpers: safe file discovery & data loading
# ------------------------------------------------------------
def _find_data_file(fname: str) -> Path:
    for d in DATA_DIRS:
        p = d / fname
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {fname} in any of {DATA_DIRS}")

@st.cache_data(show_spinner=False)
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    # model & strategy metrics
    mm = pd.read_csv(_find_data_file("model_metrics.csv"))
    sm = pd.read_csv(_find_data_file("strategy_metrics.csv"))

    # normalize headers
    mm.columns = mm.columns.str.strip().str.lower()
    sm.columns = sm.columns.str.strip().str.lower()

    # prefer demo signals if available
    try:
        sig_path = _find_data_file("signals_demo.json")
    except FileNotFoundError:
        sig_path = _find_data_file("signals_snapshot.json")

    with open(sig_path, "r") as f:
        sig = json.load(f)

    # light cleanup
    if "acc" in mm.columns and "accuracy" not in mm.columns:
        mm = mm.rename(columns={"acc": "accuracy"})
    if "maxdd" in sm.columns and "max_dd" not in sm.columns:
        sm = sm.rename(columns={"maxdd": "max_dd"})

    # force canonical dtypes where sensible
    for col in ["asset", "freq", "dataset", "model", "task"]:
        if col in mm.columns:
            mm[col] = mm[col].astype(str)
    for col in ["asset", "freq", "dataset", "family"]:
        if col in sm.columns:
            sm[col] = sm[col].astype(str)

    return mm, sm, sig

model_metrics, strategy_metrics, signals_map = load_all()

# ------------------------------------------------------------
# Labeling & parsing helpers
# ------------------------------------------------------------
MODEL_LABELS = {
    "XGB": "XGBoost", "XGB_cls": "XGBoost", "XGB_reg": "XGBoost",
    "GRU": "GRU", "GRU_cls": "GRU", "GRU_reg": "GRU",
    "LSTM": "LSTM", "LSTM_cls": "LSTM", "LSTM_reg": "LSTM",
    "RF": "Random Forest", "RF_cls": "Random Forest", "RF_reg": "Random Forest",
    "LR_cls": "Logistic Regression", "LR_reg": "Linear Regression",
    "MLP": "MLP", "MLP_cls": "MLP", "MLP_reg": "MLP",
}

def model_to_label(m: str) -> str:
    if not isinstance(m, str): return ""
    base = m.split("_")[0]
    return MODEL_LABELS.get(m, MODEL_LABELS.get(base, m))

def ta_to_label(ta: str) -> str:
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

def parse_strategy_params(params: Any) -> dict:
    """Return dict(model_param, hi, lo, ta, model_label, ta_label, rule_label)."""
    res = {"model_param": "", "hi": "", "lo": "", "ta": "", "model_label": "", "ta_label": "", "rule_label": ""}
    if not isinstance(params, str) or not params:
        return res
    try:
        # params in CSV are JSON-like
        d = json.loads(params.replace("'", "\""))
    except Exception:
        return res
    res["model_param"] = d.get("model", "")
    res["hi"] = d.get("hi", "")
    res["lo"] = d.get("lo", "")
    res["ta"] = d.get("ta", "")
    res["model_label"] = model_to_label(res["model_param"])
    res["ta_label"] = ta_to_label(res["ta"])
    bits = []
    if res["ta_label"]:
        bits.append(res["ta_label"])
    if res["hi"] != "" and res["lo"] != "":
        bits.append(f"hi/lo {res['hi']}/{res['lo']}")
    res["rule_label"] = " · ".join(bits)
    return res

def clamp(v: Any, lo: int = 0, hi: int = 100) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except Exception:
        return lo

def dash_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.nan, None], "–")

# ------------------------------------------------------------
# Gauges
# ------------------------------------------------------------
def _bands(decision: str) -> List[Dict[str, Any]]:
    d = (decision or "").upper()
    if d == "SELL":
        return [{"range": [0, 40], "color": "green"},
                {"range": [40, 60], "color": "yellow"},
                {"range": [60, 100], "color": "red"}]
    return [{"range": [0, 40], "color": "red"},
            {"range": [40, 60], "color": "yellow"},
            {"range": [60, 100], "color": "green"}]

def gauge(val: int, decision: str, title: str, show_number: bool, height: int = 150) -> go.Figure:
    v = clamp(val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number" if show_number else "gauge",
        value=v,
        number={'font': {'size': 28}} if show_number else None,
        title={"text": title, "font": {"size": 12}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "black", "thickness": 0.25},
               "steps": _bands(decision),
               "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.7, "value": v}}
    ))
    fig.update_layout(height=height, margin=dict(l=2, r=2, t=8, b=2))
    return fig

# ------------------------------------------------------------
# Reasons for Simple Mode
# ------------------------------------------------------------
def reason_text(sig: Dict[str, Any]) -> str:
    decision = (sig.get("decision") or "").upper()
    fam = (sig.get("family") or "").upper()
    freq = "Weekly" if str(sig.get("freq", "")).upper() == "W" else "Daily"
    ds = (sig.get("dataset") or "").upper()
    # Classification: prob/thr
    if fam == "CLS" and pd.notna(sig.get("prob")):
        p = float(sig["prob"])
        thr = float(sig.get("thr", 0.6))
        pdn = 1.0 - p
        if p >= thr:
            core = f"Model predicts ↑ with {p:.0%} confidence — upside edge"
        elif pdn >= thr:
            core = f"Model predicts ↓ with {pdn:.0%} confidence — downside edge"
        else:
            core = f"Model confidence {p:.0%} — no clear edge"
    # Regression: pred_ret
    elif fam == "REG" and pd.notna(sig.get("pred_ret")):
        pr = float(sig["pred_ret"])
        bias = "upside" if pr >= 0 else "downside"
        core = f"Model projects {pr:+.2%} expected return — {bias} bias"
    else:
        core = f"Model suggests {decision.title()}"
    return f"{core}. ({freq} · {ds})"

# ------------------------------------------------------------
# Best TA (for a quick confirm line in Simple Mode)
# ------------------------------------------------------------
def best_ta(asset: str, freq: str, dataset: str) -> str | None:
    sm = strategy_metrics
    sub = sm[(sm.get("asset") == asset) &
             (sm.get("freq") == freq) &
             (sm.get("dataset") == dataset) &
             (sm.get("family").str.upper() == "HYBRID_CONF")]
    if sub.empty:
        return None
    row = sub.sort_values("sharpe", ascending=False).iloc[0]
    info = parse_strategy_params(row.get("params", ""))
    return info["ta_label"] or None

# ------------------------------------------------------------
# Simple Mode card
# ------------------------------------------------------------
def _pill(text: str, tone: str) -> str:
    colors = {"good": ("#1e9e49", "#e7f6ec"),
              "bad": ("#d9534f", "#fdeaea"),
              "neutral": ("#666", "#efefef")}
    fg, bg = colors.get(tone, colors["neutral"])
    return f"<span style='background:{bg};color:{fg};padding:3px 10px;border-radius:999px;font-weight:600;margin-left:6px;'>{text}</span>"

def simple_card(asset: str, sig: Dict[str, Any], show_gauge: bool):
    decision = (sig.get("decision") or "").upper()
    idx = clamp(sig.get("index", 0))
    ta = best_ta(asset, str(sig.get("freq", "")), str(sig.get("dataset", "")))
    reason = reason_text(sig)
    dec_p = _pill("BUY" if decision == "BUY" else "SELL", "good" if decision == "BUY" else "bad")
    str_p = _pill(f"Strength {idx}", "neutral")
    ta_txt = f"; {ta} confirmation" if ta else ""
    border = "2px solid #1e9e49" if decision == "BUY" else "2px solid #d9534f"

    st.markdown(f"""
    <div style="border:{border};border-radius:12px;padding:12px 14px;margin-bottom:10px;background:#121212;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:1.05rem;font-weight:700;">{asset}</div>
        <div>{dec_p}{str_p}</div>
      </div>
      <div style="color:#cfcfcf;margin-top:6px;">{reason}{ta_txt}.</div>
    </div>
    """, unsafe_allow_html=True)

    if show_gauge:
        st.plotly_chart(gauge(idx, decision, f"{asset}", False), use_container_width=True)

    if st.button(f"View details — {asset} ({sig.get('freq','')}/{str(sig.get('dataset','')).upper()})",
                 key=f"view_{asset}_{sig.get('freq','')}_{sig.get('dataset','')}"):
        st.session_state.update({
            "mode": "Advanced",
            "asset": asset,
            "freq": str(sig.get("freq", "W")),
            "dataset": str(sig.get("dataset", "ext")).lower(),
            "tab_index": 0
        })

# ------------------------------------------------------------
# Tabs (Advanced)
# ------------------------------------------------------------
def model_comparison_tab(models: pd.DataFrame, asset: str, freq: str, dataset_code: str):
    """Model Comparison (minimal): AUC for CLS, MAE for REG."""
    md = models[(models.get("asset") == asset) &
                (models.get("freq") == freq) &
                (models.get("dataset") == dataset_code)].copy()

    c1, c2 = st.columns(2)

    # ---- Direction (Classification) ----
    with c1:
        st.markdown("### Direction Prediction (AUC ↑ better)")
        cls = md[md.get("task") == "CLS"].copy()
        if cls.empty or "auc" not in cls.columns:
            st.caption("No direction models for this selection.")
        else:
            # friendly label + keep max AUC per model label to avoid duplicates
            cls["Model"] = cls["model"].apply(model_to_label)
            tbl = (cls.groupby("Model", as_index=False)["auc"].max()
                      .rename(columns={"auc": "AUC"})
                      .sort_values("AUC", ascending=False)
                      .reset_index(drop=True))
            # best highlight
            best_row = tbl.iloc[0]
            st.markdown(f"**Best AUC:** `{best_row['AUC']:.3f}` — {best_row['Model']}")
            st.dataframe(dash_na(tbl.round(3)), use_container_width=True)

    # ---- Return (Regression) ----
    with c2:
        st.markdown("### Return Prediction (MAE ↓ better)")
        reg = md[md.get("task") == "REG"].copy()
        if reg.empty or "mae" not in reg.columns:
            st.caption("No return models for this selection.")
        else:
            reg["Model"] = reg["model"].apply(model_to_label)
            tbl = (reg.groupby("Model", as_index=False)["mae"].min()
                      .rename(columns={"mae": "MAE (lower is better)"})
                      .sort_values("MAE (lower is better)", ascending=True)
                      .reset_index(drop=True))
            best_row = tbl.iloc[0]
            st.markdown(f"**Best MAE:** `{best_row['MAE (lower is better)']:.3f}` — {best_row['Model']}")
            st.dataframe(dash_na(tbl.round(3)), use_container_width=True)


def _uplift_color(v, thr=0.002):
    try:
        if pd.isna(v): return "color: inherit"
        if v > thr:    return "color: #1e9e49; font-weight: 600"
        if v < -thr:   return "color: #d9534f; font-weight: 600"
        return "color: #999999"
    except Exception:
        return "color: inherit"

def keyword_explorer_tab(models: pd.DataFrame, asset: str, freq: str):
    st.subheader("Keyword effect (Market only vs Market + Keywords)")
    m = models[(models.get("asset") == asset) & (models.get("freq") == freq)]

    # prepare base vs ext splits
    cls_b = m[(m.get("dataset") == "base") & (m.get("task") == "CLS")]
    cls_e = m[(m.get("dataset") == "ext")  & (m.get("task") == "CLS")]
    reg_b = m[(m.get("dataset") == "base") & (m.get("task") == "REG")]
    reg_e = m[(m.get("dataset") == "ext")  & (m.get("task") == "REG")]

    def best_max(df, col): return float(df[col].max()) if (not df.empty and col in df) else float("nan")
    def best_min(df, col): return float(df[col].min()) if (not df.empty and col in df) else float("nan")

    rows = []
    for name, col in [("AUC (trend prediction)", "auc"),
                      ("Accuracy (trend prediction)", "accuracy"),
                      ("F1 (trend prediction)", "f1")]:
        b, e = best_max(cls_b, col), best_max(cls_e, col)
        if not (np.isnan(b) and np.isnan(e)):
            rows.append({"Metric": name, "Market only": b, "Market + Keywords": e, "Uplift": e - b})

    b, e = best_min(reg_b, "mae"), best_min(reg_e, "mae")
    if not (np.isnan(b) and np.isnan(e)):
        rows.append({"Metric": "MAE (return error, lower is better)", "Market only": b, "Market + Keywords": e, "Uplift": b - e})

    if "rmse" in m.columns:
        b, e = best_min(reg_b, "rmse"), best_min(reg_e, "rmse")
        if not (np.isnan(b) and np.isnan(e)):
            rows.append({"Metric": "RMSE (return error, lower is better)", "Market only": b, "Market + Keywords": e, "Uplift": b - e})

    if "spearman" in m.columns:
        b, e = best_max(reg_b, "spearman"), best_max(reg_e, "spearman")
        if not (np.isnan(b) and np.isnan(e)):
            rows.append({"Metric": "Spearman (return correlation)", "Market only": b, "Market + Keywords": e, "Uplift": e - b})

    if not rows:
        st.info("No comparison available for this selection.")
        return

    df = pd.DataFrame(rows)
    styled = (df.style
              .format({"Market only": "{:.3f}", "Market + Keywords": "{:.3f}", "Uplift": "{:+.3f}"})
              .applymap(_uplift_color, subset=["Uplift"]))
    st.dataframe(styled, use_container_width=True)

def strategy_insights_tab(strategies: pd.DataFrame, asset: str, freq: str, dataset: str):
    st.subheader("Strategy Insights")

    df = strategies.copy()
    df = df[(df["asset"] == asset) & (df["freq"] == freq) & (df["dataset"] == dataset)]

    # Parse pretty labels if not present
    if "model_label" not in df or "rule_label" not in df:
        df["model_label"] = df["params"].apply(lambda s: parse_label(s, "model") if isinstance(s, str) else "")
        df["rule_label"]  = df["params"].apply(lambda s: parse_label(s, "rule")  if isinstance(s, str) else "")
    df.rename(columns={"family":"Family","model_label":"Model","rule_label":"Rule",
                       "sharpe":"Sharpe","max_dd":"Max DD","ann_return":"Annual Return"}, inplace=True)

    # Controls
    families = sorted(df["Family"].dropna().unique().tolist())
    models   = sorted(df["Model"].dropna().unique().tolist())

    c1,c2,c3,c4 = st.columns([1.2,1.2,1.2,1])
    with c1:
        sel_fam = st.multiselect("Family", families, default=families)
    with c2:
        sel_mod = st.multiselect("Model", models, default=models)
    with c3:
        sort_by = st.selectbox("Sort by", ["Sharpe","Annual Return","Max DD"], index=0)
    with c4:
        topn = st.number_input("Top-N / family", 1, 20, 5, 1)

    q = st.text_input("Search in Rule (regex ok)", "")

    # Filter
    f = df[df["Family"].isin(sel_fam) & df["Model"].isin(sel_mod)]
    if q.strip():
        try:
            f = f[f["Rule"].str.contains(q, case=False, regex=True, na=False)]
        except re.error:
            st.warning("Invalid regex in search; showing all.")

    f = (f
         .drop_duplicates(subset=["Family","Model","Rule","Sharpe","Max DD","Annual Return"])
         .reset_index(drop=True))

    # Top-N per family by chosen metric
    asc = (sort_by == "Max DD")  # more negative is worse, so ascending
    f = (f.sort_values(by=[sort_by], ascending=asc)
           .groupby("Family", group_keys=False)
           .head(int(topn)))

    # Summary
    st.caption(f"Showing **{len(f):,}** strategies across **{f['Family'].nunique()}** family(ies).")

    # Format & style
    show = f[["Family","Model","Rule","Sharpe","Max DD","Annual Return"]].copy()
    show["Sharpe"] = show["Sharpe"].astype(float)
    show["Annual Return"] = show["Annual Return"].astype(float)
    show["Max DD"] = show["Max DD"].astype(float)

    def fmt_dec(x): 
        return f"{x:.2f}" if pd.notna(x) else "-"

    styled = (show.style
        .format({"Sharpe": fmt_dec, "Max DD": fmt_dec, "Annual Return": fmt_dec})
        .apply(lambda s: ["color:#e74c3c" if v < 0 else "" for v in s] if s.name=="Max DD" else [""]*len(s))
        .bar(subset=["Sharpe"], align="zero")
        .background_gradient(subset=["Annual Return"], cmap="Greens")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Download button
    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, file_name=f"strategies_{asset}_{freq}_{dataset}.csv", mime="text/csv")


def context_tab(signals: Dict[str, Any], asset: str, freq: str, dataset_code: str):
    # try to find a matching signal
    target = None
    for k, v in signals.items():
        # support both compact keys and full records
        a = v.get("asset") or (k.split("-")[0] if "-" in k else "")
        f = v.get("freq")  or (k.split("-")[1] if "-" in k and len(k.split("-")) >= 2 else "")
        d = (v.get("dataset") or (k.split("-")[2] if "-" in k and len(k.split("-")) >= 3 else "")).lower()
        if a == asset and str(f).upper() == freq and d == dataset_code.lower():
            target = v; break

    st.subheader("Context")
    if not target:
        st.caption("No snapshot entry for this selection.")
        return
    st.markdown("#### Sentiment gauge")
    st.caption("Index reflects model confidence / predicted return strength (0–100).")
    st.plotly_chart(
        gauge(target.get("index", 0), target.get("decision", "SELL"), f"{asset} · {freq} · {dataset_code.upper()}", True),
        use_container_width=True
    )

# ------------------------------------------------------------
# Modes
# ------------------------------------------------------------
def simple_mode():
    st.title("What should I trade now?")
    c1, c2, c3 = st.columns([1.1, 1, 1.3])

    with c1:
        min_strength = st.slider("Min strength", 0, 100, 50)
    with c2:
        show_gauges = st.toggle("Show gauges", value=True)
    with c3:
        st.caption("Click a card to drill into Advanced.")

    # build BUY/SELL lists
    buy, sell = [], []

    # signals_map supports two formats:
    #  1) compact demo keys like "BTC-W" or objects with no 'asset'
    #  2) full records with 'asset','freq','dataset'
    for k, s in signals_map.items():
        asset = s.get("asset") or (k.split("-")[0] if "-" in k else str(k))
        freq  = str(s.get("freq") or (k.split("-")[1] if "-" in k and len(k.split("-")) >= 2 else "W")).upper()
        # normalize dataset; ignore custom ones like 'eng' in simple mode
        dataset = str(s.get("dataset") or (k.split("-")[2] if "-" in k and len(k.split("-")) >= 3 else "ext")).lower()
        if dataset not in {"base", "ext"}:
            dataset = "ext"

        s = {**s, "asset": asset, "freq": freq, "dataset": dataset}

        strength = clamp(s.get("index", 0))
        if strength < min_strength:
            continue
        dec = (s.get("decision") or "").upper()
        if dec == "BUY":
            buy.append((asset, s))
        elif dec == "SELL":
            sell.append((asset, s))

    col_b, col_s = st.columns(2)
    with col_b:
        st.header("BUY")
        if not buy:
            st.caption("No BUY signals under current filters.")
        for a, s in sorted(buy, key=lambda t: -clamp(t[1]["index"]))[:4]:
            simple_card(a, s, show_gauge=show_gauges)

    with col_s:
        st.header("SELL")
        if not sell:
            st.caption("No SELL signals under current filters.")
        for a, s in sorted(sell, key=lambda t: -clamp(t[1]["index"]))[:4]:
            simple_card(a, s, show_gauge=show_gauges)

def advanced_mode(models: pd.DataFrame, strategies: pd.DataFrame, signals: Dict[str, Any]):
    st.title("Show me why")

    assets = sorted(models.get("asset").dropna().astype(str).unique().tolist())
    asset_def = st.session_state.get("asset", assets[0] if assets else "")
    freq_def = st.session_state.get("freq", "W")
    dataset_def = st.session_state.get("dataset", "ext")

    label_to_code = {"Market only": "base", "Market + Keywords": "ext"}
    code_to_label = {v: k for k, v in label_to_code.items()}
    init_label = code_to_label.get(dataset_def, "Market + Keywords")

    colA, colB, colC = st.columns([1.2, 0.9, 1.3])
    with colA:
        asset = st.selectbox("Asset", assets, index=(assets.index(asset_def) if asset_def in assets else 0))
    with colB:
        freq = st.radio("Frequency", ["D", "W"], horizontal=True,
                        index=(["D", "W"].index(freq_def) if freq_def in ["D", "W"] else 1))
    with colC:
        ds_label = st.radio("Dataset", ["Market only", "Market + Keywords"], horizontal=True,
                            index=(["Market only", "Market + Keywords"].index(init_label)))
        dataset_code = label_to_code[ds_label]

    tabs = st.tabs(["Model Comparison", "Keyword Explorer", "Strategy Insights", "Context"])
    with tabs[0]:
        model_comparison_tab(models, asset, freq, dataset_code)
    with tabs[1]:
        keyword_explorer_tab(models, asset, freq)
    with tabs[2]:
        strategy_insights_tab(strategies, asset, freq, dataset_code)
    with tabs[3]:
        context_tab(signals, asset, freq, dataset_code)

# ------------------------------------------------------------
# App entry
# ------------------------------------------------------------
st.set_page_config(page_title="Markets Demo", layout="wide")

mode = st.sidebar.radio("Mode", ["Simple", "Advanced"],
                        index=(0 if st.session_state.get("mode", "Simple") == "Simple" else 1))

if mode == "Simple":
    st.session_state["mode"] = "Simple"
    simple_mode()
else:
    st.session_state["mode"] = "Advanced"
    advanced_mode(model_metrics, strategy_metrics, signals_map)


def parse_label(params: str, part: str) -> str:
    if not isinstance(params, str):
        return ""
    if part == "model":
        m = re.search(r"(model|mdl)\s*=\s*([^;|]+)", params, re.I)
    else:
        m = re.search(r"(rule|ta)\s*=\s*(.+)", params, re.I)
    return m.group(2).strip() if m else params

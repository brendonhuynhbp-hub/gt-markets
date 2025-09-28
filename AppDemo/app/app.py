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
    """
    Keyword Explorer (redesigned):
      - No dataset toggle (only Asset + Frequency)
      - KPI cards: Δ AUC (ext − base), Δ MAE (base − ext)
      - Split tables: Direction (AUC, Accuracy, F1) & Return (MAE, RMSE, Spearman)
      - Collapsible bar chart of NORMALISED uplifts at the bottom
    """
    st.subheader("Keyword Explorer")

    # Filter to asset & freq
    m = models[(models.get("asset") == asset) & (models.get("freq") == freq)]
    if m.empty:
        st.info("No data for this selection.")
        return

    # Split by dataset & task
    cls_b = m[(m.get("dataset") == "base") & (m.get("task") == "CLS")]
    cls_e = m[(m.get("dataset") == "ext")  & (m.get("task") == "CLS")]
    reg_b = m[(m.get("dataset") == "base") & (m.get("task") == "REG")]
    reg_e = m[(m.get("dataset") == "ext")  & (m.get("task") == "REG")]

    def best_max(df, col): return float(df[col].max()) if (not df.empty and col in df) else float("nan")
    def best_min(df, col): return float(df[col].min()) if (not df.empty and col in df) else float("nan")

    # Best scores by dataset
    best_auc_b, best_auc_e = best_max(cls_b, "auc"), best_max(cls_e, "auc")
    best_acc_b, best_acc_e = best_max(cls_b, "accuracy"), best_max(cls_e, "accuracy")
    best_f1_b,  best_f1_e  = best_max(cls_b, "f1"), best_max(cls_e, "f1")
    best_mae_b, best_mae_e = best_min(reg_b, "mae"), best_min(reg_e, "mae")
    best_rmse_b, best_rmse_e = best_min(reg_b, "rmse"), best_min(reg_e, "rmse")
    best_spear_b, best_spear_e = best_max(reg_b, "spearman"), best_max(reg_e, "spearman")

    # Uplifts (Direction: ext - base; Return: base - ext)
    d_auc = (best_auc_e - best_auc_b) if not (np.isnan(best_auc_b) or np.isnan(best_auc_e)) else np.nan
    d_acc = (best_acc_e - best_acc_b) if not (np.isnan(best_acc_b) or np.isnan(best_acc_e)) else np.nan
    d_f1  = (best_f1_e  - best_f1_b ) if not (np.isnan(best_f1_b ) or np.isnan(best_f1_e )) else np.nan
    d_mae = (best_mae_b - best_mae_e) if not (np.isnan(best_mae_b) or np.isnan(best_mae_e)) else np.nan
    d_rmse= (best_rmse_b - best_rmse_e) if not (np.isnan(best_rmse_b) or np.isnan(best_rmse_e)) else np.nan
    d_spear=(best_spear_e - best_spear_b) if not (np.isnan(best_spear_b) or np.isnan(best_spear_e)) else np.nan

    # KPI cards
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Δ AUC (ext − base)", value=("+" if not np.isnan(d_auc) and d_auc>=0 else "") + (f"{d_auc:.3f}" if not np.isnan(d_auc) else "—"))
    with c2:
        st.metric("Δ MAE (base − ext)", value=("+" if not np.isnan(d_mae) and d_mae>=0 else "") + (f"{d_mae:.3f}" if not np.isnan(d_mae) else "—"))

    # Direction metrics table
    dir_rows = []
    if not (np.isnan(best_auc_b) and np.isnan(best_auc_e)):
        dir_rows.append({"Metric": "AUC", "Market only": best_auc_b, "Market + Keywords": best_auc_e, "Δ": d_auc})
    if not (np.isnan(best_acc_b) and np.isnan(best_acc_e)):
        dir_rows.append({"Metric": "Accuracy", "Market only": best_acc_b, "Market + Keywords": best_acc_e, "Δ": d_acc})
    if not (np.isnan(best_f1_b) and np.isnan(best_f1_e)):
        dir_rows.append({"Metric": "F1", "Market only": best_f1_b, "Market + Keywords": best_f1_e, "Δ": d_f1})

    # Return metrics table
    ret_rows = []
    if not (np.isnan(best_mae_b) and np.isnan(best_mae_e)):
        ret_rows.append({"Metric": "MAE", "Market only": best_mae_b, "Market + Keywords": best_mae_e, "Δ": d_mae})
    if not (np.isnan(best_rmse_b) and np.isnan(best_rmse_e)):
        ret_rows.append({"Metric": "RMSE", "Market only": best_rmse_b, "Market + Keywords": best_rmse_e, "Δ": d_rmse})
    if not (np.isnan(best_spear_b) and np.isnan(best_spear_e)):
        ret_rows.append({"Metric": "Spearman", "Market only": best_spear_b, "Market + Keywords": best_spear_e, "Δ": d_spear})

    st.markdown("### Direction metrics")
    if dir_rows:
        df_dir = pd.DataFrame(dir_rows)
        styled_dir = (df_dir.style
                      .format({"Market only": "{:.3f}", "Market + Keywords": "{:.3f}", "Δ": "{:+.3f}"})
                      .applymap(_uplift_color, subset=["Δ"]))
        st.dataframe(styled_dir, use_container_width=True, hide_index=True)
    else:
        st.caption("No direction metrics available.")

    st.markdown("### Return metrics")
    if ret_rows:
        df_ret = pd.DataFrame(ret_rows)
        styled_ret = (df_ret.style
                      .format({"Market only": "{:.3f}", "Market + Keywords": "{:.3f}", "Δ": "{:+.3f}"})
                      .applymap(_uplift_color, subset=["Δ"]))
        st.dataframe(styled_ret, use_container_width=True, hide_index=True)
    else:
        st.caption("No return metrics available.")

    # Normalised uplift chart (single combined)
    with st.expander("Show normalised uplifts (Δ% vs base)", expanded=False):
        rows = []
        if not (np.isnan(best_auc_b) and np.isnan(best_auc_e)):
            rows.append(("AUC", best_auc_b, best_auc_e, "direction"))
        if not (np.isnan(best_acc_b) and np.isnan(best_acc_e)):
            rows.append(("Accuracy", best_acc_b, best_acc_e, "direction"))
        if not (np.isnan(best_f1_b) and np.isnan(best_f1_e)):
            rows.append(("F1", best_f1_b, best_f1_e, "direction"))
        if not (np.isnan(best_mae_b) and np.isnan(best_mae_e)):
            rows.append(("MAE", best_mae_b, best_mae_e, "return"))
        if not (np.isnan(best_rmse_b) and np.isnan(best_rmse_e)):
            rows.append(("RMSE", best_rmse_b, best_rmse_e, "return"))
        if not (np.isnan(best_spear_b) and np.isnan(best_spear_e)):
            rows.append(("Spearman", best_spear_b, best_spear_e, "return"))
        if rows:
            import plotly.graph_objects as go
            metrics, pct, hover = [], [], []
            for name, base_v, ext_v, family in rows:
                if np.isnan(base_v) or base_v == 0:
                    change = np.nan
                else:
                    if family == "direction":
                        change = (ext_v - base_v) / base_v
                    else:
                        change = (base_v - ext_v) / base_v
                metrics.append(name)
                pct.append(change if not np.isnan(change) else 0.0)
                hover.append(f"{name}: base={base_v:.4g}, ext={ext_v:.4g}, Δ%={(change*100 if not np.isnan(change) else 0):.2f}%")
            colors = ["#2ca02c" if val >= 0 else "#d62728" for val in pct]
            fig = go.Figure(data=[go.Bar(x=metrics, y=[v*100 for v in pct], text=[f"{v*100:.2f}%" for v in pct],
                                         textposition="outside", marker=dict(color=colors), hovertext=hover, hoverinfo="text")])
            fig.update_layout(yaxis_title="Δ vs base (%)", xaxis_title="Metric",
                              margin=dict(l=0, r=10, t=10, b=0), height=420)
            fig.update_yaxes(zeroline=True, zerolinewidth=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No data to chart.")


def strategy_insights_tab(strategies: pd.DataFrame, asset: str, freq: str, dataset_code: str):
    sd = strategies[(strategies.get("asset") == asset) &
                    (strategies.get("freq") == freq) &
                    (strategies.get("dataset") == dataset_code)]
    if sd.empty:
        st.warning("No strategy metrics for this selection.")
        return

    # parse params into readable fields
    parsed = sd["params"].fillna("").apply(parse_strategy_params).tolist() if "params" in sd.columns else []
    if parsed:
        parsed_df = pd.DataFrame(parsed, index=sd.index)
        sd = pd.concat([sd, parsed_df], axis=1)
    else:
        sd["model_label"] = ""
        sd["rule_label"] = ""

    # family ordering to keep hybrids together
    fam_order = ["HYBRID_CONF", "ML", "TA", "HYBRID"]
    sd["fam_sort"] = sd["family"].apply(lambda x: fam_order.index(x) if x in fam_order else 99)

    cols_wanted = ["family", "model_label", "rule_label", "sharpe", "max_dd", "ann_return"]
    cols_avail  = [c for c in cols_wanted if c in sd.columns]

    df = (
        sd.sort_values(["fam_sort", "sharpe"], ascending=[True, False])[cols_avail]
          .rename(columns={
              "family": "Family",
              "model_label": "Model",
              "rule_label": "Rule",
              "sharpe": "Sharpe",
              "max_dd": "Max DD",
              "ann_return": "Annual Return"
          })
          .round({"Sharpe": 3, "Max DD": 3, "Annual Return": 3})
          .reset_index(drop=True)
    )
    st.dataframe(dash_na(df), use_container_width=True)

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

from __future__ import annotations
import re
import json
import ast
import numpy as np

# ======================= Helpers for Strategy Insights =======================
def _params_to_dict(v):
    """Parse a params cell that may be a dict, JSON string, or Python-literal string."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                d = loader(v)
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    return {}

def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Populate model_label / rule_label from params or existing columns."""
    import pandas as pd
    if "model_label" not in df.columns:
        df["model_label"] = pd.NA
    if "rule_label" not in df.columns:
        df["rule_label"] = pd.NA

    need_model = df["model_label"].fillna("").eq("").any()
    need_rule  = df["rule_label"].fillna("").eq("").any()

    if need_model or need_rule:
        p = df["params"] if "params" in df.columns else pd.Series([""]*len(df), index=df.index)
        parsed = p.apply(_params_to_dict)

        if need_model:
            df.loc[df["model_label"].fillna("").eq(""), "model_label"] = parsed.apply(
                lambda d: str(d.get("model","")) if isinstance(d, dict) else ""
            )

        if need_rule:
            # Prefer explicit ta_label column if present
            if "ta_label" in df.columns:
                ta_col = df["ta_label"].astype(str)
                mask = df["rule_label"].fillna("").eq("") & ta_col.ne("")
                df.loc[mask, "rule_label"] = ta_col

            # Otherwise build from params keys
            def build_rule(d: dict) -> str:
                if not isinstance(d, dict):
                    return ""
                return str(d.get("ta_label") or d.get("ta") or d.get("rule") or "")
            fallback = parsed.apply(build_rule)
            df.loc[df["rule_label"].fillna("").eq(""), "rule_label"] = fallback

    for c in ["model_label","rule_label"]:
        df[c] = df[c].replace("", pd.NA)
    return df

def _model_friendly(name) -> str:
    """Return a concise model label with target type. Robust to None/NaN/<NA>."""
    import pandas as pd, math, re as _re
    try:
        if name is None or (isinstance(name, float) and math.isnan(name)) or (hasattr(pd, "isna") and pd.isna(name)):
            n = ""
        else:
            n = str(name)
    except Exception:
        n = ""
    n = n.strip()
    base = n if n else ""
    target = ""
    m = _re.match(r'([A-Za-z0-9]+)[_\- ]?(cls|reg)?', n or "")
    if m:
        base = m.group(1).upper()
        kind = (m.group(2) or "").lower()
        if kind == "cls":
            target = "Direction"
        elif kind == "reg":
            target = "Return"
    aliases = {"XGB": "XGB", "RF": "RF", "MLP": "MLP", "LSTM": "LSTM", "LR": "Logistic"}
    pretty = aliases.get(base, base) if base else ""
    return f"{pretty} ({target})" if (pretty and target) else (pretty or "-")

_INDICATOR_NAMES = {
    "TA_MAcross": "MA cross",
    "TA_MACD": "MACD",
    "TA_RSI": "RSI",
    "TA_BBands": "Bollinger Bands",
}

def split_rule_columns(rule_text: str) -> tuple[str, str]:
    """Extract (Indicator, Window) from a rule string like 'TA_MAcross_10-50 ...'"""
    indicator = ""
    window = ""
    if isinstance(rule_text, str) and rule_text:
        base = rule_text.split("·")[0].strip()
        parts = base.split("_")
        if len(parts) >= 2:
            code = "_".join(parts[:2])
            indicator = _INDICATOR_NAMES.get(code, code.replace("TA_", "").replace("_", " "))
            if len(parts) >= 3:
                window = parts[2]
        else:
            indicator = base
    return indicator, window

def thresholds_to_policy(rule_text: str, params: dict | str | None) -> str:
    """
    Map thresholds to Risk Appetite labels (Set B):
      Aggressive (≈0.50), Moderate (≈0.52), Conservative (≈0.55), Ultra-Conservative (≈0.60).
    If no thresholds found -> Aggressive.
    """
    d = _params_to_dict(params or {})
    hi = d.get("hi"); lo = d.get("lo")

    if hi is None and isinstance(rule_text, str):
        m = re.search(r"hi/lo\s+([0-9.]+)\s*/\s*([0-9.]+)", rule_text or "")
        if m:
            try:
                hi = float(m.group(1)); lo = float(m.group(2))
            except Exception:
                hi = lo = None

    if hi is None or lo is None:
        return "Aggressive"
    if hi >= 0.595:
        return "Ultra-Conservative"
    if hi >= 0.545:
        return "Conservative"
    if hi >= 0.515:
        return "Moderate"
    return "Aggressive"

def _infer_model_code(row: dict) -> str:
    """
    Infer canonical model code from row:
      - Prefer params["model"] if available.
      - Else parse family_id / family for tokens like XGB, RF, LSTM, GRU, MLP, LR and optional cls/reg.
    Returns standardized codes like "XGB_cls", "RF_reg", etc., or empty string if unknown.
    """
    import re as _re, json as _json, ast as _ast

    # 1) params.model
    params = row.get("params")
    if params:
        if isinstance(params, dict):
            m = params.get("model")
            if m: return str(m)
        elif isinstance(params, str):
            for loader in (_json.loads, _ast.literal_eval):
                try:
                    d = loader(params)
                    if isinstance(d, dict):
                        m = d.get("model")
                        if m: return str(m)
                except Exception:
                    pass

    # 2) family_id or family
    for key in ("family_id", "family"):
        val = row.get(key)
        if not isinstance(val, str):
            continue
        # Look for patterns like XGB_cls, RF-reg, LSTM, GRU, MLP, LR with optional suffix cls/reg
        m = _re.search(r'\b(XGB|RF|LSTM|GRU|MLP|LR)\b[-_ ]?(cls|reg)?', val, flags=_re.I)
        if m:
            base = m.group(1).upper()
            kind = (m.group(2) or "").lower()
            if kind in ("cls","reg"):
                return f"{base}_{kind}"
            # if kind missing, try to infer from other hints in string
            if _re.search(r'\breg(ress|ression)?\b', val, _re.I):
                return f"{base}_reg"
            if _re.search(r'\bcls(class|ification)?\b', val, _re.I):
                return f"{base}_cls"
            # default to classification if unknown
            return f"{base}_cls"

    return ""

# ===================== End Helpers for Strategy Insights =====================

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

    # >>> CHANGED: asset name color set to white (#ffffff)
    st.markdown(f"""
    <div style="border:{border};border-radius:12px;padding:12px 14px;margin-bottom:10px;background:#121212;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:1.05rem;font-weight:700;color:#ffffff;">{asset}</div>
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
    for col in ["asset","freq","dataset","family","sharpe","max_dd","ann_return"]:
        if col not in df.columns:
            st.warning(f"Missing column '{col}' in strategies data.")
            return

    df = df[(df["asset"] == asset) & (df["freq"] == freq) & (df["dataset"] == dataset)].copy()

    # robust labels and renames
    df = derive_labels(df)
    df.rename(columns={"family":"Family","model_label":"Model","rule_label":"Rule",
                       "sharpe":"Sharpe","max_dd":"Max DD","ann_return":"Annual Return"}, inplace=True)

    # Infer Model from params/family_id/family when missing
    if "Model" not in df.columns or df["Model"].fillna("").eq("").all():
        df["Model"] = df.apply(lambda r: _infer_model_code(r.to_dict()), axis=1)
    # If partially missing, fill only blanks
    mask_empty = df["Model"].fillna("").eq("")
    if mask_empty.any():
        df.loc[mask_empty, "Model"] = df[mask_empty].apply(lambda r: _infer_model_code(r.to_dict()), axis=1)

    if "params" not in df.columns:
        df["params"] = None

    # Build Setup and Policy, and friendly model names
    parts = df.apply(lambda r: split_rule_columns(str(r.get("Rule", ""))), axis=1)
    df["Indicator"], df["Window"] = zip(*parts)
    df["Setup"] = df.apply(
        lambda r: (
            r["Indicator"]
            + (f" ({str(r['Window']).replace('-', '/')})" if pd.notna(r["Window"]) and str(r["Window"]).strip() not in ["", "nan"] else "")
        ),
        axis=1,
    )
    df["Model"] = df["Model"].apply(_model_friendly)
    df["Confidence Policy"] = df.apply(lambda r: thresholds_to_policy(str(r.get("Rule","")), r.get("params")), axis=1)

    # Controls (models inferred from strategies only, so the list always matches available data)
    df["Model"] = df["Model"].apply(_model_friendly)
    options = sorted(df["Model"].dropna().unique().tolist())
    c1,c2 = st.columns([1.3,1.3])
    with c1:
        sel_mod = st.multiselect("Model", options, default=options)
    with c2:
        sort_by = st.selectbox("Sort by", ["Sharpe","Annual Return","Max DD"], index=0)
    q = st.text_input("Search (Setup / Model / Policy)", "")

    # Filter
    f = df[df["Model"].isin(sel_mod)].copy()
    if q.strip():
        qre = re.compile(re.escape(q), re.I)
        mask = (
            f["Setup"].fillna("").str.contains(qre) |
            f["Model"].fillna("").str.contains(qre) |
            f["Confidence Policy"].fillna("").str.contains(qre) |
            f["Rule"].fillna("").str.contains(qre)
        )
        f = f[mask]

    # Dedupe + sort
    f = (
        f.drop_duplicates(subset=["Model","Setup","Confidence Policy","Sharpe","Max DD","Annual Return"])
         .reset_index(drop=True)
    )
    asc = (sort_by == "Max DD")
    f = f.sort_values(by=[sort_by], ascending=asc)

    st.caption(f"Showing **{len(f):,}** strategies across **{f['Model'].nunique()}** model(s).")

    # Display
    show = f[["Model","Setup","Confidence Policy","Sharpe","Max DD","Annual Return"]].copy()
    for col in ["Sharpe","Max DD","Annual Return"]:
        show[col] = pd.to_numeric(show[col], errors="coerce")

    def fmt_dec(x):
        return f"{x:.2f}" if pd.notna(x) else "-"

    styled = (
        show.style
            .format({"Sharpe": fmt_dec, "Max DD": fmt_dec, "Annual Return": fmt_dec})
            .apply(lambda s: ["color:#e74c3c" if (pd.notna(v) and v < 0) else "" for v in s] if s.name=="Max DD" else [""]*len(s))
            .bar(subset=["Sharpe"], align="zero")
    )
    try:
        import matplotlib  # optional gradient
        styled = styled.background_gradient(subset=["Annual Return"], cmap="Greens")
    except Exception:
        pass

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Download
    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, file_name=f"strategies_{asset}_{freq}_{dataset}.csv", mime="text/csv")


def context_tab(signals: Dict[str, Any], asset: str, freq: str, dataset_code: str):
    """Demo-only snapshot that varies by (asset, freq). No backtest. High-contrast cards."""
    import numpy as np
    import streamlit as st

    # ---------- deterministic demo generator (varies by asset+freq) ----------
    key = f"{asset}-{freq}".upper()
    seed = (abs(hash(key)) % (2**32 - 1)) or 42
    rng = np.random.default_rng(seed)

    # choose signal & trend consistently
    sig = rng.choice(["BUY", "SELL", "HOLD"], p=[0.48, 0.32, 0.20])
    trend = {"BUY": "Uptrend", "SELL": "Downtrend", "HOLD": "Sideways"}[sig]

    # confidence & moves (skew to match signal)
    base_conf = rng.integers(58, 86)  # 58–85%
    dmove = rng.normal(0.006, 0.012)  # ~0.6% avg, +/- 1.2% std
    wmove = rng.normal(0.018, 0.025)  # ~1.8% avg week
    if sig == "BUY":
        dmove = abs(dmove)
        wmove = abs(wmove)
    elif sig == "SELL":
        dmove = -abs(dmove)
        wmove = -abs(wmove)
    else:
        dmove *= 0.4
        wmove *= 0.4

    # momentum & vol
    rsi = int(np.clip(rng.normal(54 if sig=="BUY" else 46 if sig=="SELL" else 50, 7), 15, 85))
    macd_hist = float(rng.normal(0.004 if sig=="BUY" else -0.004 if sig=="SELL" else 0.0, 0.002))
    vol_pctile = float(np.clip(rng.uniform(0.2, 0.85), 0, 1))

    # keywords (vary by asset a bit)
    kw_pool = {
        "BTC": ["ETF inflow", "halving", "institutional", "on-chain"],
        "OIL": ["OPEC supply", "inventory draw", "geopolitics", "refinery"],
        "GOLD": ["real yields", "safe haven", "ETF holdings", "USD"],
        "SPX": ["earnings", "rate cuts", "AI capex", "buybacks"],
    }
    kws = rng.choice(kw_pool.get(asset.upper(), ["momentum", "macro", "flows", "positioning"]),
                     size=3, replace=False).tolist()

    # ---------- helpers ----------
    def fmt_pct(x):
        try: return f"{100*float(x):.1f}%"
        except: return "–"

    def bucket_vol(p):
        try:
            p = float(p);  return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"
        except: return "–"

    def bucket_rsi(v):
        try:
            v = float(v)
            return f"{int(v)} (Overbought)" if v >= 70 else f"{int(v)} (Oversold)" if v <= 30 else f"{int(v)} (Neutral)"
        except: return "–"

    def macd_txt(h):
        try:
            h = float(h);  return "Bullish" if h > 0 else "Bearish" if h < 0 else "Flat"
        except: return "–"

    # ---------- styles (high contrast, minimal fill) ----------
    st.subheader("Context")
    st.markdown("""
    <style>
      .kpi{border:1px solid rgba(255,255,255,.18); border-radius:14px; padding:12px 14px;
           background: rgba(255,255,255,0.02); height:100%;
           display:flex; flex-direction:column; gap:6px}
      .kpi .label{font-size:.85rem; color:#b9c3cf}
      .kpi .value{font-size:26px; font-weight:800; line-height:1.15; color:#000000}
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

    # colors
    sig_color   = "blue" if sig == "BUY" else "red" if sig == "SELL" else ""
    today_color = "blue" if dmove > 0 else "red" if dmove < 0 else ""
    week_color  = "blue" if wmove > 0 else "red" if wmove < 0 else ""
    trend_color = "blue" if trend == "Uptrend" else "red" if trend == "Downtrend" else ""

    # ---------- layout ----------
    r1 = st.columns(4)
    card(r1[0], "Signal", sig, sig_color)
    card(r1[1], "Confidence", f"{int(base_conf)}%")
    card(r1[2], "Trend", trend, trend_color)
    card(r1[3], "RSI(14)", bucket_rsi(rsi))

    r2 = st.columns(4)
    card(r2[0], "Today", fmt_pct(dmove), today_color)
    card(r2[1], "1W", fmt_pct(wmove), week_color)
    card(r2[2], "MACD", macd_txt(macd_hist))
    card(r2[3], "Volatility", bucket_vol(vol_pctile))

    if kws:
        st.markdown("".join([f'<span class="pill">{str(k)}</span>' for k in kws]), unsafe_allow_html=True)

    st.write(
        f"**Summary:** {asset} {fmt_pct(dmove)} today, {fmt_pct(wmove)} this week; "
        f"trend **{trend}**, RSI **{bucket_rsi(rsi)}**, vol **{bucket_vol(vol_pctile)}**."
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

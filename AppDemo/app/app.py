# AppDemo/app/app.py
from __future__ import annotations

import json, ast, math, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Paths & data loading
# ---------------------------------------------------------------------
def _find_data_dir() -> Path:
    candidates = [
        Path(__file__).parent / "data",                       # AppDemo/app/data
        Path(__file__).parent.parent / "data",                # AppDemo/data
        Path.cwd() / "AppDemo" / "data",
        Path.cwd() / "data",
        Path("/content/gt-markets/AppDemo/data"),
        Path("/mount/src/gt-markets/AppDemo/data"),
        Path("/mnt/data"),  # local dev
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path.cwd()

DATA_DIR = _find_data_dir()

def _latest(glob: str) -> Path | None:
    hits = sorted(DATA_DIR.glob(glob))
    return hits[-1] if hits else None

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    mm_path = _latest("model_metrics*.csv")
    sm_path = _latest("strategy_metrics*.csv")
    sig_path = _latest("signals_snapshot*.json") or _latest("signals*.json") or (DATA_DIR / "signals_snapshot.json")

    if mm_path is None or sm_path is None:
        st.error("model_metrics.csv or strategy_metrics.csv not found in AppDemo/data.")
        return pd.DataFrame(), pd.DataFrame(), {}

    model_metrics = pd.read_csv(mm_path)
    strategy_metrics = pd.read_csv(sm_path)

    # signals snapshot (optional)
    signals_map: Dict[str, Any] = {}
    if sig_path and sig_path.exists():
        try:
            signals_map = json.loads(sig_path.read_text())
        except Exception:
            try:
                signals_map = ast.literal_eval(sig_path.read_text())
            except Exception:
                signals_map = {}

    return model_metrics, strategy_metrics, signals_map

model_metrics, strategy_metrics, signals_map = load_inputs()

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _dash_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.nan, None], "-")

def _pct(x) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "-"

def _f2(x) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "-"

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def _model_short_dirret(name) -> str:
    """
    "RF_cls" -> "RF (Direction)"
    "XGB_reg" -> "XGB (Return)"
    """
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

def _parse_params_to_dict(p):
    if p is None:
        return {}
    if isinstance(p, dict):
        return p
    if isinstance(p, str):
        s = p.strip()
        if not s:
            return {}
        for loader in (json.loads, ast.literal_eval):
            try:
                d = loader(s)
                return d if isinstance(d, dict) else {}
            except Exception:
                continue
    try:
        if pd.isna(p):
            return {}
    except Exception:
        pass
    return {}

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
        return f"RSI({per}){th}", "Momentum oscillator. >70 overbought (pullback risk); <30 oversold (bounce potential)."
    if fam == "MACD":
        w = str(params.get("window") or params.get("macd") or params.get("ta_window") or "12-26-9")
        parts = w.replace(" ", "").replace("_","-").split("-")
        label = f"MACD cross ({parts[0]}/{parts[1]}, {parts[2]})" if len(parts) >= 3 else "MACD cross (12/26, 9)"
        return label, "Momentum turning point. Cross above signal bullish; below bearish."
    rt = (rule_text or "").strip() or "Custom rule"
    return rt, "Rule-based signal from the selected strategy."

def _format_perf(sharpe, max_dd, ann_ret):
    return _f2(sharpe), _f2(max_dd), _pct(ann_ret)

# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------
ASSETS = sorted(strategy_metrics["asset"].dropna().unique().tolist() if "asset" in strategy_metrics.columns else ["BTC","OIL"])
FREQS  = ["D","W"]

# ---------------------------------------------------------------------
# Simple Mode (minimal, safe)
# ---------------------------------------------------------------------
def simple_mode():
    st.header("Show me why")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        asset = st.selectbox("Asset", ASSETS, index=0)
    with c2:
        freq = st.radio("Frequency", FREQS, horizontal=True, index=1)
    with c3:
        dataset = st.radio("Dataset", ["Market only", "Market + Keywords"], horizontal=True, index=1)

    st.write("This is a lightweight Simple Mode placeholder. Switch to **Advanced** for full analysis.")
    # A tiny summary
    if not model_metrics.empty:
        mm = model_metrics.copy()
        if "asset" in mm.columns:
            mm = mm[mm["asset"] == asset]
        st.dataframe(_dash_na(mm.head(10)), use_container_width=True)

# ---------------------------------------------------------------------
# Keyword Explorer
# ---------------------------------------------------------------------
def keyword_explorer_tab(models_df: pd.DataFrame, asset: str, freq: str):
    st.subheader("Keyword Explorer")

    # Compute deltas between Market only vs Market+Keywords (if available)
    df = models_df.copy()
    if not {"asset","freq","dataset","metric","value"}.issubset(df.columns):
        st.info("Keyword Explorer requires tidy model metrics with ['asset','freq','dataset','metric','value'].")
        return

    base = df[(df["asset"]==asset) & (df["freq"]==freq)]
    if base.empty:
        st.info("No metrics for this asset/frequency.")
        return

    # Pivot so we can compute deltas
    pivot = base.pivot_table(index="metric", columns="dataset", values="value", aggfunc="mean")
    # Map to expected dataset names if needed
    cols = {c.lower():c for c in pivot.columns}
    def _get(colnames):
        # try both exact and casefold
        for want in ["market only","market_only","market"]:
            if want in cols: return cols[want]
        for c in pivot.columns:
            if "market" in c.lower() and "keyword" not in c.lower():
                return c
        return None
    def _get_kw(colnames):
        for want in ["market + keywords","market+keywords","market_keywords","market and keywords","keywords"]:
            if want in cols: return cols[want]
        for c in pivot.columns:
            if "keyword" in c.lower():
                return c
        return None

    col_base = _get(pivot.columns)
    col_kw   = _get_kw(pivot.columns)

    if col_base is None or col_kw is None:
        st.info("Need both datasets (Market only, Market + Keywords) to compute deltas.")
        show = pivot.reset_index()
        st.dataframe(_dash_na(show), use_container_width=True)
        return

    pivot["Δ"] = pivot[col_kw] - pivot[col_base]

    # KPI cards: Δ AUC and Δ MAE
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Δ AUC", _f2(pivot.loc["AUC","Δ"]) if "AUC" in pivot.index else "-")
    with c2:
        st.metric("Δ MAE", _f2(pivot.loc["MAE","Δ"]) if "MAE" in pivot.index else "-")

    # Direction metrics
    dir_metrics = ["AUC","Accuracy","F1"]
    ret_metrics = ["MAE","RMSE","Spearman"]

    def _make_table(metrics: List[str]) -> pd.DataFrame:
        rows = []
        for m in metrics:
            if m in pivot.index:
                rows.append({
                    "Metric": m,
                    "Market only": _f2(pivot.loc[m, col_base]),
                    "Market + Keywords": _f2(pivot.loc[m, col_kw]),
                    "Δ": _f2(pivot.loc[m,"Δ"])
                })
        return pd.DataFrame(rows)

    st.markdown("### Direction metrics")
    st.dataframe(_dash_na(_make_table(dir_metrics)), use_container_width=True)

    st.markdown("### Return metrics")
    st.dataframe(_dash_na(_make_table(ret_metrics)), use_container_width=True)

# ---------------------------------------------------------------------
# Strategy Insights
# ---------------------------------------------------------------------
def strategy_insights_tab(strategies: pd.DataFrame, asset: str, freq: str, dataset_code: str):
    st.subheader("Strategy Insights")

    if strategies.empty:
        st.info("No strategies available.")
        return

    df = strategies.copy()
    # Robust columns
    if "model_label" in df.columns:
        df.rename(columns={"model_label":"Model"}, inplace=True)
    if "rule_label" in df.columns:
        df.rename(columns={"rule_label":"Setup"}, inplace=True)
    if "rule" in df.columns and "Setup" not in df.columns:
        df.rename(columns={"rule":"Setup"}, inplace=True)

    # Short model label with target
    if "Model" in df.columns:
        df["Model"] = df["Model"].apply(_model_short_dirret)
    elif "family_id" in df.columns:
        df["Model"] = df["family_id"].apply(_model_short_dirret)
    else:
        df["Model"] = "-"

    # Build Setup if missing from params
    if "Setup" not in df.columns:
        df["Setup"] = df["params"].apply(lambda s: _explain_setup("", _parse_params_to_dict(s))[0] if isinstance(s,str) else "-")
    else:
        # enrich if too raw
        df["Setup"] = df.apply(lambda r: _explain_setup(str(r.get("Setup","")), _parse_params_to_dict(r.get("params"))) [0], axis=1)

    # Filter (asset/freq/dataset) if present
    for col, val in [("asset", asset), ("freq", freq), ("dataset", dataset_code)]:
        if col in df.columns:
            df = df[df[col] == val]

    # Controls: Model with "All"
    model_options = sorted(df["Model"].dropna().unique().tolist())
    all_option = "All"
    model_choices = [all_option] + model_options
    sel_mod = st.multiselect("Model", model_choices, default=[all_option])
    active_models = set(model_options) if (not sel_mod or all_option in sel_mod) else set([m for m in sel_mod if m != all_option])

    f = df[df["Model"].isin(list(active_models))].copy()

    # Sort by Sharpe desc
    if "sharpe" in f.columns:
        f = f.sort_values(by=["sharpe"], ascending=False)

    # Pick visible columns
    cols = []
    for c in ["Model","Setup","sharpe","max_dd","ann_return"]:
        if c in f.columns: cols.append(c)
    show = f[cols].copy()
    # Nicify
    if "sharpe" in show.columns: show.rename(columns={"sharpe":"Sharpe"}, inplace=True)
    if "max_dd" in show.columns: show.rename(columns={"max_dd":"Max DD"}, inplace=True)
    if "ann_return" in show.columns: show.rename(columns={"ann_return":"Annual Return"}, inplace=True)

    st.dataframe(_dash_na(show), use_container_width=True)

# ---------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------
def context_tab(signals_map: dict, asset: str, freq: str, dataset: str, strategies: pd.DataFrame = None):
    def _fmt(x, pct=False):
        try:
            x = float(x)
            return (f"{x:.2f}" if not pct else f"{x*100:.1f}%")
        except Exception:
            return "-"

    # ---------- SIGNAL SNAPSHOT ----------
    snap = None
    keys = list((signals_map or {}).keys())
    # exacts first
    for k in [f"{asset}-{freq}", f"{asset}_{freq}", asset]:
        if isinstance(signals_map, dict) and k in signals_map:
            snap = signals_map[k]; break
    # fuzzy
    if snap is None and isinstance(signals_map, dict):
        aset = asset.lower(); fr = freq.lower()
        for k in keys:
            lk = str(k).lower()
            if aset in lk and fr[0] in lk:  # 'w' in 'weekly'
                snap = signals_map[k]; break

    sig_txt, conf, ts = "-", None, "-"
    if isinstance(snap, dict):
        sig_txt = str(snap.get("signal") or snap.get("action") or "-").upper()
        conf    = snap.get("confidence") or snap.get("score")
        ts      = str(snap.get("timestamp") or snap.get("time") or "-")

    st.subheader("Context")
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Signal", sig_txt)
    with c2: st.metric("Confidence", f"{conf:.0f}%" if isinstance(conf,(int,float)) else "-")
    with c3: st.metric("As of", ts)

    # ---------- STRATEGY ROW (progressive fallback) ----------
    expl_label, expl_why = None, None
    sh, dd, ar = None, None, None

    if isinstance(strategies, pd.DataFrame) and not strategies.empty:
        df = strategies.copy()
        for c in ["asset","freq","dataset","rule","Rule","params"]:
            if c not in df.columns: df[c] = np.nan

        def pick(d):
            if d.empty: return None
            if "sharpe" in d.columns:
                d = d.sort_values("sharpe", ascending=False)
            return d.iloc[0].to_dict()

        cand = None
        for subset in [
            df[(df["asset"]==asset) & (df["freq"]==freq) & (df["dataset"]==dataset)],
            df[(df["asset"]==asset) & (df["freq"]==freq)],
            df[(df["asset"]==asset)],
            df
        ]:
            cand = pick(subset)
            if cand: break

        if cand:
            p = _parse_params_to_dict(cand.get("params"))
            expl_label, expl_why = _explain_setup(cand.get("rule") or cand.get("Rule"), p)
            sh, dd, ar = cand.get("sharpe"), cand.get("max_dd"), cand.get("ann_return")

    if expl_label is None:
        if sig_txt == "BUY":
            expl_label, expl_why = "MA cross (10/50)", "Short-term momentum crossed above long-term; trend continuation likely."
        elif sig_txt == "SELL":
            expl_label, expl_why = "RSI(14) > 70", "Overbought conditions; risk of pullback."
        else:
            expl_label, expl_why = "MACD cross (12/26, 9)", "Momentum turning; watch for confirmation."

    st.markdown(f"**Setup:** {expl_label}")
    st.caption(expl_why or "Derived from the best historical strategy for this market and timeframe.")

    k1,k2,k3 = st.columns(3)
    with k1: st.metric("Sharpe", _fmt(sh))
    with k2: st.metric("Max Drawdown", _fmt(dd))
    with k3: st.metric("Annual Return", _fmt(ar, pct=True))

    with st.expander("Debug (data wiring)"):
        st.write("Signal keys:", list((signals_map or {}).keys())[:20])
        if isinstance(strategies, pd.DataFrame):
            st.write("Rows by filters:",
                     {"asset+freq+dataset": int(((strategies.get("asset")==asset) & (strategies.get("freq")==freq) & (strategies.get("dataset")==dataset)).sum() if "asset" in strategies else 0),
                      "asset+freq": int(((strategies.get("asset")==asset) & (strategies.get("freq")==freq)).sum() if "asset" in strategies else 0),
                      "asset": int((strategies.get("asset")==asset).sum() if "asset" in strategies else 0),
                      "total": len(strategies)})

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
# Main router (robust)
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Markets Demo", layout="wide")
    mode = st.sidebar.radio("Mode", ["Simple","Advanced"], index=1)
    st.session_state["mode"] = mode

    if mode == "Simple":
        if "simple_mode" in globals() and callable(simple_mode):
            simple_mode()
        else:
            st.warning("Simple Mode is unavailable in this build.")
    else:
        if "advanced_mode" in globals() and callable(advanced_mode):
            advanced_mode(model_metrics, strategy_metrics, signals_map)
        else:
            st.error("Advanced Mode is unavailable in this build.")

if __name__ == "__main__":
    main()

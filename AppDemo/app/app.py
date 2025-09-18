# app.py
# GT Markets – Demo App (Streamlit)
# - Auto-detects artefacts folder (or set ARTEFACTS_ROOT env var)
# - Landing page shows Baseline & Keyword metrics
#   • Reads root-level CSVs if present
#   • Otherwise aggregates from per-asset files on the fly
# - Explorer page per asset (leaderboard, metrics, discovered keyword sets)

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

APP_TITLE = "📊 GT Markets – Demo App"
ENV_ARTEFACTS = os.environ.get("ARTEFACTS_ROOT", "").strip()

# ---------------- Path helpers ----------------

def _candidate_roots() -> List[Path]:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        Path(ENV_ARTEFACTS) if ENV_ARTEFACTS else None,
        here / "artefacts",
        here.parent / "AppDemo" / "artefacts",
        cwd / "AppDemo" / "artefacts",
        cwd / "artefacts",
        Path("/mount/src/gt-markets/AppDemo/artefacts"),
        Path("/workspace/gt-markets/AppDemo/artefacts"),
    ]
    return [p for p in candidates if p is not None]

def _likely_valid(root: Path) -> bool:
    if not root.exists():
        return False
    expected_any = [
        root / "metrics_summary_D.csv",
        root / "metrics_summary_W.csv",
        root / "metrics_keywords_D.csv",
        root / "metrics_keywords_W.csv",
    ]
    return any(p.exists() for p in expected_any) or any(p.is_dir() for p in root.iterdir())

@st.cache_data(show_spinner=False)
def detect_artefacts_root() -> Tuple[Path | None, List[str]]:
    checked = []
    for c in _candidate_roots():
        checked.append(str(c))
        if _likely_valid(c):
            return c, checked
    for c in _candidate_roots():
        checked.append(str(c))
        if c.exists():
            return c, checked
    return None, checked

# ---------------- I/O helpers ----------------

@st.cache_data(show_spinner=False)
def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path or not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Failed to read {path.name}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def safe_read_json(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        try:
            return pd.read_json(path).to_dict()
        except Exception:
            return {}

def list_assets(root: Path) -> List[str]:
    if not root or not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def files_in(root: Path, pattern: str) -> List[str]:
    if not root or not root.exists():
        return []
    return sorted([p.name for p in root.glob(pattern)])

# ---------------- Aggregation (root fallbacks) ----------------

@st.cache_data(show_spinner=False)
def aggregate_baseline(root: Path, freq: str) -> pd.DataFrame:
    """Concat all per-asset metrics_baseline_<freq>.csv into a root-style summary."""
    frames = []
    for asset in list_assets(root):
        f = root / asset / f"metrics_baseline_{freq}.csv"
        df = safe_read_csv(f)
        if not df.empty:
            # enforce columns
            must = ["type", "asset", "freq", "strategy", "Return_Ann", "Sharpe", "MaxDD", "WinRate", "N_Trades"]
            for m in ["asset", "freq"]:
                if m not in df.columns:
                    df[m] = asset if m == "asset" else freq
            for m in ["type", "strategy"]:
                if m not in df.columns:
                    df[m] = "baseline" if m == "type" else "BASE_EMA"
            frames.append(df[[c for c in must if c in df.columns] + [c for c in df.columns if c not in must]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(show_spinner=False)
def aggregate_keywords(root: Path, freq: str) -> pd.DataFrame:
    """Concat all per-asset metrics_keywords_<freq>.csv into a root keyword table."""
    frames = []
    for asset in list_assets(root):
        f = root / asset / f"metrics_keywords_{freq}.csv"
        df = safe_read_csv(f)
        if not df.empty:
            if "asset" not in df.columns:
                df["asset"] = asset
            if "freq" not in df.columns:
                df["freq"] = freq
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ---------------- Keyword discovery ----------------

KEYWORD_COL_CANDIDATES = ["keywords","keyword","kw","kw_list","kwset","kw_set","keyword_set","keywordset","kw_group","kwgroup"]
SET_COL_CANDIDATES     = ["set","kw_set","keyword_set","feature_set","featureset","group","kw_group"]
MODEL_COL_CANDIDATES   = ["model","clf","algo","classifier"]
WIN_COL_CANDIDATES     = ["window","w","win","lookback","horizon"]

def _parse_keywords_cell(val) -> List[str]:
    if pd.isna(val): return []
    s = str(val).strip()
    if "|" in s: return [x.strip() for x in s.split("|") if x.strip()]
    if "," in s and not s.startswith("{}"):
        cand = [x.strip() for x in s.split(",") if x.strip()]
        if len(cand) > 1: return cand
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)): return [str(x).strip() for x in obj if str(x).strip()]
        except Exception: pass
    return [s] if s else []

def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None

def _extract_keywords_from_metadata(meta: dict) -> Dict[str, List[str]]:
    if not meta or not isinstance(meta, dict): return {}
    for key in ["keyword_sets","keywords","kw_sets","kwsets","kw_groups"]:
        if key in meta and isinstance(meta[key], dict):
            out = {}
            for set_name, v in meta[key].items():
                if isinstance(v, list):
                    out[str(set_name).upper()] = [str(x).strip() for x in v if str(x).strip()]
                elif isinstance(v, dict) and "values" in v and isinstance(v["values"], list):
                    out[str(set_name).upper()] = [str(x).strip() for x in v["values"] if str(x).strip()]
            if out: return out
    flat = {}
    for k, v in meta.items():
        if re.search(r"(kw|keyword).*(base|ext)", k, flags=re.I) and isinstance(v, list):
            set_name = "BASE" if re.search(r"base", k, flags=re.I) else "EXT"
            flat.setdefault(set_name, []).extend([str(x).strip() for x in v if str(x).strip()])
    if flat: return flat
    out = {}
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if re.search(r"(kw|keyword).*set", k, flags=re.I) and isinstance(v, (list, dict)):
                    if isinstance(v, list): out[k.upper()] = [str(x).strip() for x in v if str(x).strip()]
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(vv, list): out[str(kk).upper()] = [str(x).strip() for x in vv if str(x).strip()]
                else: walk(v)
        elif isinstance(obj, list):
            for it in obj: walk(it)
    walk(meta)
    return out

def _discover_from_metrics_csv(df: pd.DataFrame) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if df.empty: return out
    kw_col   = _first_existing_col(df, KEYWORD_COL_CANDIDATES)
    set_col  = _first_existing_col(df, SET_COL_CANDIDATES)
    model_c  = _first_existing_col(df, MODEL_COL_CANDIDATES)
    win_c    = _first_existing_col(df, WIN_COL_CANDIDATES)

    if kw_col:
        if set_col and set_col in df.columns:
            for set_name, sub in df.groupby(set_col):
                s = str(set_name).upper()
                kws = []
                for v in sub[kw_col].dropna().unique().tolist(): kws.extend(_parse_keywords_cell(v))
                if not kws: continue
                out.setdefault(s, {"keywords": [], "models": set(), "windows": set()})
                out[s]["keywords"] = sorted(set(out[s]["keywords"] + kws))
                if model_c: out[s]["models"].update([str(x) for x in sub[model_c].dropna().unique().tolist()])
                if win_c:   out[s]["windows"].update([str(x) for x in sub[win_c].dropna().unique().tolist()])
        else:
            kws = []
            for v in df[kw_col].dropna().unique().tolist(): kws.extend(_parse_keywords_cell(v))
            if kws:
                out.setdefault("ALL", {"keywords": [], "models": set(), "windows": set()})
                out["ALL"]["keywords"] = sorted(set(kws))
                if model_c: out["ALL"]["models"] = set([str(x) for x in df[model_c].dropna().unique().tolist()])
                if win_c:   out["ALL"]["windows"] = set([str(x) for x in df[win_c].dropna().unique().tolist()])
    else:
        if set_col:
            for set_name, sub in df.groupby(set_col):
                s = str(set_name).upper()
                out.setdefault(s, {"keywords": [], "models": set(), "windows": set()})
                if model_c: out[s]["models"].update([str(x) for x in sub[model_c].dropna().unique().tolist()])
                if win_c:   out[s]["windows"].update([str(x) for x in sub[win_c].dropna().unique().tolist()])
    return out

def _discover_sets_from_signal_names(asset_dir: Path, freq: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for f in asset_dir.glob(f"signals_KW_*_{freq}.csv"):
        tokens = f.stem.replace("signals_", "").split("_")
        model = None; set_name = None; win = None
        for t in tokens:
            if t.upper() in {"BASE","EXT"}: set_name = t.upper()
            elif re.fullmatch(r"w?\d+", t, flags=re.I): win = t
            elif t.upper() in {"KW","D","W"}: pass
            else: model = t.upper()
        if set_name:
            out.setdefault(set_name, {"keywords": [], "models": set(), "windows": set()})
            if model: out[set_name]["models"].add(model)
            if win:   out[set_name]["windows"].add(win)
    return out

@st.cache_data(show_spinner=False)
def discover_keyword_sets(root: Path, asset: str) -> pd.DataFrame:
    asset_dir = root / asset
    rows = []

    meta = safe_read_json(asset_dir / "metadata.json")
    meta_map = _extract_keywords_from_metadata(meta) if meta else {}

    for freq in ["D","W"]:
        df_kw = safe_read_csv(asset_dir / f"metrics_keywords_{freq}.csv")
        via_metrics = _discover_from_metrics_csv(df_kw) if not df_kw.empty else {}
        via_signals = _discover_sets_from_signal_names(asset_dir, freq)

        sets = sorted(set(via_metrics.keys()) | set(via_signals.keys()) | set(meta_map.keys()))
        for s in sets:
            models  = set()
            wins    = set()
            keywords = []
            if s in via_metrics:
                models |= via_metrics[s].get("models", set())
                wins   |= via_metrics[s].get("windows", set())
                keywords = via_metrics[s].get("keywords", []) or keywords
            if s in via_signals:
                models |= via_signals[s].get("models", set())
                wins   |= via_signals[s].get("windows", set())
            if not keywords and s in meta_map:
                keywords = meta_map[s]
            src = []
            if s in via_metrics: src.append("metrics")
            if s in via_signals: src.append("signals")
            if s in meta_map:    src.append("metadata")

            rows.append({
                "set": s,
                "freq": freq,
                "models": ", ".join(sorted(models)) if models else "—",
                "windows": ", ".join(sorted(wins)) if wins else "—",
                "keywords": ", ".join(keywords) if keywords else "—",
                "source": "+".join(src) if src else "—",
            })
    return pd.DataFrame(rows).sort_values(["set","freq"]).reset_index(drop=True)

# ---------------- UI – Landing ----------------

def landing(root: Path):
    st.title(APP_TITLE)
    st.caption(f"Artefacts root: `{root}`")

    # Baseline
    st.markdown("## 🔹 Baseline Metrics")
    tab_d, tab_w = st.tabs(["Daily (D)", "Weekly (W)"])
    with tab_d:
        df_d = safe_read_csv(root / "metrics_summary_D.csv")
        if df_d.empty:
            df_d = aggregate_baseline(root, "D")
        st.dataframe(df_d, use_container_width=True, hide_index=True) if not df_d.empty else st.info("No baseline metrics for Daily.")
    with tab_w:
        df_w = safe_read_csv(root / "metrics_summary_W.csv")
        if df_w.empty:
            df_w = aggregate_baseline(root, "W")
        st.dataframe(df_w, use_container_width=True, hide_index=True) if not df_w.empty else st.info("No baseline metrics for Weekly.")

    st.divider()

    # Keyword
    st.markdown("## 🔹 Keyword Model Metrics")
    tab_kd, tab_kw = st.tabs(["Daily (D)", "Weekly (W)"])
    with tab_kd:
        kw_d = safe_read_csv(root / "metrics_keywords_D.csv")
        if kw_d.empty:
            kw_d = aggregate_keywords(root, "D")
        st.dataframe(kw_d, use_container_width=True, hide_index=True) if not kw_d.empty else st.info("No keyword metrics for Daily.")
    with tab_kw:
        kw_w = safe_read_csv(root / "metrics_keywords_W.csv")
        if kw_w.empty:
            kw_w = aggregate_keywords(root, "W")
        st.dataframe(kw_w, use_container_width=True, hide_index=True) if not kw_w.empty else st.info("No keyword metrics for Weekly.")

    # Diagnostics
    with st.expander("🔎 Diagnostics"):
        assets = list_assets(root)
        st.write("**Detected asset folders:**", ", ".join(assets) if assets else "—")
        cols = st.columns(2)
        with cols[0]:
            st.write("- metrics_summary_D.csv:", (root / "metrics_summary_D.csv").exists())
            st.write("- metrics_keywords_D.csv:", (root / "metrics_keywords_D.csv").exists())
        with cols[1]:
            st.write("- metrics_summary_W.csv:", (root / "metrics_summary_W.csv").exists())
            st.write("- metrics_keywords_W.csv:", (root / "metrics_keywords_W.csv").exists())
        st.write("**Root CSVs present:**")
        st.code("\n".join(files_in(root, "*.csv")) or "—", language="text")

# ---------------- UI – Explorer ----------------

def explorer(root: Path):
    st.title("🔭 Explorer (Per-Asset)")
    assets = list_assets(root)
    if not assets:
        st.info("No asset folders detected in artefacts root.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        asset = st.selectbox("Asset", assets, index=0)
        freq = st.radio("Frequency", ["D", "W"], horizontal=True)

    asset_dir = root / asset
    st.caption(f"Folder: `{asset_dir}`")

    st.markdown("### Leaderboard")
    lb = safe_read_csv(asset_dir / f"leaderboard_{freq}.csv")
    st.dataframe(lb, use_container_width=True, hide_index=True) if not lb.empty else st.info("No leaderboard available.")

    st.markdown("### Baseline Metrics (asset)")
    base = safe_read_csv(asset_dir / f"metrics_baseline_{freq}.csv")
    st.dataframe(base, use_container_width=True, hide_index=True) if not base.empty else st.info("No per-asset baseline metrics.")

    st.markdown("### Keyword Metrics (asset)")
    kw_asset = safe_read_csv(asset_dir / f"metrics_keywords_{freq}.csv")
    st.dataframe(kw_asset, use_container_width=True, hide_index=True) if not kw_asset.empty else st.info("No per-asset keyword metrics.")

    st.markdown("### 🧪 Keyword Sets (discovered)")
    kw_sets = discover_keyword_sets(root, asset)
    if kw_sets.empty:
        st.info("No keyword sets discovered from output files.")
    else:
        st.dataframe(kw_sets[kw_sets["freq"] == freq], use_container_width=True, hide_index=True)

    with st.expander("📁 Signals available"):
        sigs = files_in(asset_dir, f"signals_*_{freq}.csv")
        st.code("\n".join(sigs) or "—", language="text")

    with st.expander("🧾 Raw files"):
        present = [
            f"leaderboard_{freq}.csv",
            f"metrics_baseline_{freq}.csv",
            f"metrics_keywords_{freq}.csv",
            "metadata.json",
        ]
        st.write("**Expected files:**")
        cols = st.columns(2)
        for i, name in enumerate(present):
            p = asset_dir / name
            (cols[i % 2]).write(f"- {name}: {'✅' if p.exists() else '❌'}")

# ---------------- Entry ----------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    root, checked = detect_artefacts_root()
    if root is None:
        st.error("Could not locate an artefacts folder.")
        st.write("Paths checked:")
        st.code("\n".join(checked) or "—")
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Landing", "Explorer"])
    st.sidebar.caption("Paths checked:")
    st.sidebar.code("\n".join(checked), language="text")

    if page == "Landing":
        landing(root)
    else:
        explorer(root)

if __name__ == "__main__":
    main()

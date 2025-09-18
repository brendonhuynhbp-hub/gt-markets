# app.py
# GT Markets – Demo App (Streamlit)
# - No Google Drive dependency
# - Auto-detects artefacts folder (or set ARTEFACTS_ROOT env var)
# - Baseline & Keyword metrics on Landing
# - Explorer page with per-asset Leaderboard, Metrics, and Keyword Sets discovered
# - Keyword discovery scans: metadata.json, metrics_keywords_*.csv, signals_KW_* files

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


# ============================================================
# Config
# ============================================================

APP_TITLE = "📊 Market Model Dashboard"
ENV_ARTEFACTS = os.environ.get("ARTEFACTS_ROOT", "").strip()  # optional override


# ============================================================
# Path helpers
# ============================================================

def _candidate_roots() -> List[Path]:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        Path(ENV_ARTEFACTS) if ENV_ARTEFACTS else None,
        here / "artefacts",
        here / "AppDemo" / "artefacts",
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
    return any(p.exists() for p in expected_any)


@st.cache_data(show_spinner=False)
def detect_artefacts_root() -> Tuple[Path | None, List[str]]:
    checked = []
    for c in _candidate_roots():
        checked.append(str(c))
        if _likely_valid(c):
            return c, checked
    # fallback: first existing 'artefacts' dir (even if empty)
    for c in _candidate_roots():
        if c.exists():
            return c, checked
    return None, checked


# ============================================================
# I/O helpers
# ============================================================

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
            # try pandas fallback (handles comments sometimes)
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


# ============================================================
# Keyword discovery logic
# ============================================================

KEYWORD_COL_CANDIDATES = [
    "keywords", "keyword", "kw", "kw_list", "kwset", "kw_set",
    "keyword_set", "keywordset", "kw_group", "kwgroup",
]

SET_COL_CANDIDATES = [
    "set", "kw_set", "keyword_set", "feature_set", "featureset",
    "group", "kw_group",
]

MODEL_COL_CANDIDATES = ["model", "clf", "algo", "classifier"]
WIN_COL_CANDIDATES = ["window", "w", "win", "lookback", "horizon"]


def _parse_keywords_cell(val) -> List[str]:
    """
    Parse a cell that might contain keywords in various formats:
    - "gold|usd|inflation"
    - "['gold', 'usd']"
    - '["gold","usd"]'
    - single word "gold"
    """
    if pd.isna(val):
        return []
    s = str(val).strip()

    # pipe or comma separated
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    if "," in s and not s.startswith("{"):
        # might be csv style
        cand = [x.strip() for x in s.split(",") if x.strip()]
        if len(cand) > 1:
            return cand

    # python/json literal list
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass

    # fallback: single token
    return [s] if s else []


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    # softer match contains
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None


def _extract_keywords_from_metadata(meta: dict) -> Dict[str, List[str]]:
    """
    Try several likely shapes for metadata.json to pull keyword sets.
    Expected mapping like {"BASE": [...], "EXT": [...]} or similar.
    """
    if not meta or not isinstance(meta, dict):
        return {}

    # direct hits
    for key in ["keyword_sets", "keywords", "kw_sets", "kwsets", "kw_groups"]:
        if key in meta and isinstance(meta[key], dict):
            # values can be list[str] or nested dicts
            out = {}
            for set_name, v in meta[key].items():
                if isinstance(v, list):
                    out[str(set_name).upper()] = [str(x).strip() for x in v if str(x).strip()]
                elif isinstance(v, dict) and "values" in v and isinstance(v["values"], list):
                    out[str(set_name).upper()] = [str(x).strip() for x in v["values"] if str(x).strip()]
            if out:
                return out

    # flattened possibilities (kw_base / kw_ext)
    flat = {}
    for k, v in meta.items():
        if re.search(r"(kw|keyword).*(base|ext)", k, flags=re.I) and isinstance(v, list):
            set_name = "BASE" if re.search(r"base", k, flags=re.I) else "EXT"
            flat.setdefault(set_name, [])
            flat[set_name].extend([str(x).strip() for x in v if str(x).strip()])
    if flat:
        return flat

    # deep search for any dicts named like keyword sets
    out = {}
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if re.search(r"(kw|keyword).*set", k, flags=re.I) and isinstance(v, (list, dict)):
                    if isinstance(v, list):
                        out[k.upper()] = [str(x).strip() for x in v if str(x).strip()]
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(vv, list):
                                out[str(kk).upper()] = [str(x).strip() for x in vv if str(x).strip()]
                else:
                    walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)
    walk(meta)
    return out


def _discover_from_metrics_csv(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    From a metrics_keywords_*.csv DataFrame, try to assemble sets.
    Returns mapping: set_name -> {"keywords": [...], "models": set([...]), "windows": set([...])}
    """
    out: Dict[str, Dict] = {}
    if df.empty:
        return out

    # candidate columns
    kw_col = _first_existing_col(df, KEYWORD_COL_CANDIDATES)  # may directly contain keywords
    set_col = _first_existing_col(df, SET_COL_CANDIDATES)
    model_col = _first_existing_col(df, MODEL_COL_CANDIDATES)
    win_col = _first_existing_col(df, WIN_COL_CANDIDATES)

    if kw_col:
        # there are explicit keywords in rows → group by set (or single group)
        if set_col and set_col in df.columns:
            for set_name, sub in df.groupby(set_col):
                set_key = str(set_name).upper()
                kws = []
                for v in sub[kw_col].dropna().unique().tolist():
                    kws.extend(_parse_keywords_cell(v))
                if not kws:
                    continue
                out.setdefault(set_key, {"keywords": [], "models": set(), "windows": set()})
                out[set_key]["keywords"] = sorted(set(out[set_key]["keywords"] + kws))
                if model_col:
                    out[set_key]["models"].update([str(x) for x in sub[model_col].dropna().unique().tolist()])
                if win_col:
                    out[set_key]["windows"].update([str(x) for x in sub[win_col].dropna().unique().tolist()])
        else:
            # no set column → aggregate all as "ALL"
            kws = []
            for v in df[kw_col].dropna().unique().tolist():
                kws.extend(_parse_keywords_cell(v))
            if kws:
                out.setdefault("ALL", {"keywords": [], "models": set(), "windows": set()})
                out["ALL"]["keywords"] = sorted(set(kws))
                if model_col:
                    out["ALL"]["models"] = set([str(x) for x in df[model_col].dropna().unique().tolist()])
                if win_col:
                    out["ALL"]["windows"] = set([str(x) for x in df[win_col].dropna().unique().tolist()])
    else:
        # no explicit keywords column → still capture set/model/window labels if present
        if set_col:
            for set_name, sub in df.groupby(set_col):
                set_key = str(set_name).upper()
                out.setdefault(set_key, {"keywords": [], "models": set(), "windows": set()})
                if model_col:
                    out[set_key]["models"].update([str(x) for x in sub[model_col].dropna().unique().tolist()])
                if win_col:
                    out[set_key]["windows"].update([str(x) for x in sub[win_col].dropna().unique().tolist()])
        else:
            # nothing usable in df
            pass

    return out


def _discover_sets_from_signal_names(asset_dir: Path, freq: str) -> Dict[str, Dict]:
    """
    Use KW signal filenames to infer set names (e.g., BASE/EXT) and models/windows.
    signals_KW_LR_BASE_D.csv → set=BASE, model=LR, win=w30 (if present)
    """
    out: Dict[str, Dict] = {}
    for f in asset_dir.glob(f"signals_KW_*_{freq}.csv"):
        name = f.name  # e.g., signals_KW_LR_w30_BASE_D.csv or signals_KW_LR_BASE_D.csv
        # capture tokens
        tokens = name.replace("signals_", "").replace(".csv", "").split("_")
        # tokens like ["KW", "LR", "w30", "BASE", "D"] or ["KW","LR","BASE","D"]
        model = None
        set_name = None
        win = None
        for t in tokens:
            if t.upper() in {"BASE", "EXT"}:
                set_name = t.upper()
            elif re.fullmatch(r"w?\d+", t, flags=re.I):
                win = t
            elif t.upper() in {"KW", "D", "W"}:
                pass
            else:
                # likely model token: LR / RF / XGB / GRU / LSTM / MLP
                model = t.upper()
        if set_name:
            out.setdefault(set_name, {"keywords": [], "models": set(), "windows": set()})
            if model:
                out[set_name]["models"].add(model)
            if win:
                out[set_name]["windows"].add(win)
    return out


@st.cache_data(show_spinner=False)
def discover_keyword_sets(root: Path, asset: str) -> pd.DataFrame:
    """
    Build a table of discovered keyword sets for an asset by combining:
    - metadata.json (preferred for the actual keyword lists)
    - metrics_keywords_D/W.csv (for explicit keywords, sets, models, windows)
    - signals_KW_* files (for set/model/window inference)
    Output columns: set, freq, models, windows, keywords, source
    """
    asset_dir = root / asset
    rows = []

    # 1) metadata.json (asset-level)
    meta_path = asset_dir / "metadata.json"
    meta = safe_read_json(meta_path)
    meta_map = _extract_keywords_from_metadata(meta) if meta else {}

    # 2) metrics_keywords (per freq)
    for freq in ["D", "W"]:
        df_kw = safe_read_csv(asset_dir / f"metrics_keywords_{freq}.csv")
        via_metrics = _discover_from_metrics_csv(df_kw) if not df_kw.empty else {}

        # 3) signals filenames (per freq)
        via_signals = _discover_sets_from_signal_names(asset_dir, freq)

        # merge sources: metrics → signals → metadata (for keywords fill)
        sets = sorted(set(via_metrics.keys()) | set(via_signals.keys()) | set(meta_map.keys()))
        for s in sets:
            models = set()
            wins = set()
            keywords = []

            if s in via_metrics:
                models |= via_metrics[s].get("models", set())
                wins |= via_metrics[s].get("windows", set())
                keywords = via_metrics[s].get("keywords", []) or keywords

            if s in via_signals:
                models |= via_signals[s].get("models", set())
                wins |= via_signals[s].get("windows", set())

            if not keywords and s in meta_map:
                keywords = meta_map[s]

            # Pick source label
            src = []
            if s in via_metrics:
                src.append("metrics")
            if s in via_signals:
                src.append("signals")
            if s in meta_map:
                src.append("metadata")
            source = "+".join(src) if src else "—"

            rows.append({
                "set": s,
                "freq": freq,
                "models": ", ".join(sorted(models)) if models else "—",
                "windows": ", ".join(sorted(wins)) if wins else "—",
                "keywords": ", ".join(keywords) if keywords else "—",
                "source": source,
            })

    df_out = pd.DataFrame(rows).sort_values(["set", "freq"]).reset_index(drop=True)
    return df_out


# ============================================================
# UI – Landing
# ============================================================

def landing(root: Path):
    st.title(APP_TITLE)
    st.caption(f"Artefacts root: `{root}`")

    # -------- Baseline --------
    st.markdown("## 🔹 Baseline Metrics")
    tab_d, tab_w = st.tabs(["Daily (D)", "Weekly (W)"])
    with tab_d:
        df_d = safe_read_csv(root / "metrics_summary_D.csv")
        st.dataframe(df_d, use_container_width=True, hide_index=True) if not df_d.empty else st.info("No baseline metrics for Daily.")
    with tab_w:
        df_w = safe_read_csv(root / "metrics_summary_W.csv")
        st.dataframe(df_w, use_container_width=True, hide_index=True) if not df_w.empty else st.info("No baseline metrics for Weekly.")

    st.divider()

    # -------- Keyword --------
    st.markdown("## 🔹 Keyword Model Metrics")
    tab_kd, tab_kw = st.tabs(["Daily (D)", "Weekly (W)"])
    with tab_kd:
        kw_d = safe_read_csv(root / "metrics_keywords_D.csv")
        st.dataframe(kw_d, use_container_width=True, hide_index=True) if not kw_d.empty else st.info("No keyword metrics for Daily.")
    with tab_kw:
        kw_w = safe_read_csv(root / "metrics_keywords_W.csv")
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
        st.write("**Root CSVs:**")
        st.code("\n".join(files_in(root, "*.csv")) or "—", language="text")


# ============================================================
# UI – Explorer (per asset)
# ============================================================

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

    # Leaderboard
    st.markdown("### Leaderboard")
    lb = safe_read_csv(asset_dir / f"leaderboard_{freq}.csv")
    st.dataframe(lb, use_container_width=True, hide_index=True) if not lb.empty else st.info("No leaderboard available.")

    # Baseline metrics
    st.markdown("### Baseline Metrics (asset)")
    base = safe_read_csv(asset_dir / f"metrics_baseline_{freq}.csv")
    st.dataframe(base, use_container_width=True, hide_index=True) if not base.empty else st.info("No per-asset baseline metrics.")

    # Keyword metrics for the asset
    st.markdown("### Keyword Metrics (asset)")
    kw_asset = safe_read_csv(asset_dir / f"metrics_keywords_{freq}.csv")
    st.dataframe(kw_asset, use_container_width=True, hide_index=True) if not kw_asset.empty else st.info("No per-asset keyword metrics.")

    # Keyword sets discovered
    st.markdown("### 🧪 Keyword Sets (discovered)")
    kw_sets = discover_keyword_sets(root, asset)
    if kw_sets.empty:
        st.info("No keyword sets discovered from output files.")
    else:
        # show only current freq by default
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


# ============================================================
# Entry
# ============================================================

def main():
    st.set_page_config(page_title="GT Markets – Demo App", layout="wide")

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

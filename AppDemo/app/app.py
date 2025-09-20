# app.py — GT Markets · Interactive Metrics Table + Comparison
import streamlit as st
import pandas as pd
from pathlib import Path

# =========================
# Config
# =========================
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

# =========================
# Helpers
# =========================
def load_metrics(kind: str, freq: str) -> pd.DataFrame:
    """
    Load metrics CSV by kind ('baseline' or 'keywords') and freq ('D' or 'W').
    Normalizes columns for clean comparison.
    """
    fname = f"metrics_{kind}_{freq}.csv"
    fpath = ARTEFACTS_DIR / fname
    if not fpath.exists():
        return pd.DataFrame()

    df = pd.read_csv(fpath)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure required cols exist
    expected = {"market", "freq", "strategy", "sharpe", "return", "hitrate", "maxdd"}
    missing = expected - set(df.columns)
    if missing:
        st.warning(f"{fname} missing columns: {missing}")
        return pd.DataFrame()

    # Clean core identity cols
    df["market"] = df["market"].str.upper()
    df["freq"] = freq
    df["strategy"] = df["strategy"].str.upper()

    # Keywords: may have source + model
    if kind == "keywords":
        if "source" not in df.columns:
            df["source"] = "—"
        else:
            df["source"] = df["source"].astype(str).str.upper().replace({"BASE": "BASE", "EXT": "EXT"})
        if "model" not in df.columns:
            df["model"] = "—"
        else:
            df["model"] = df["model"].astype(str).str.upper()
    else:
        df["source"] = "—"
        df["model"] = "—"

    # Numeric cols
    for col in ["sharpe", "return", "hitrate", "maxdd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def combine_with_deltas(df_base: pd.DataFrame, df_kw: pd.DataFrame) -> pd.DataFrame:
    """
    Merge baseline + keywords and compute deltas.
    """
    all_rows = []

    for _, b in df_base.iterrows():
        anchor = (b["market"], b["freq"], b["strategy"])
        all_rows.append(
            {
                "Market": b["market"],
                "Freq": b["freq"],
                "Strategy": b["strategy"],
                "Source": "—",
                "Model": "—",
                "Sharpe": b["sharpe"],
                "Return %": b["return"],
                "HitRate %": b["hitrate"],
                "MaxDD %": b["maxdd"],
                "ΔSharpe": None,
                "ΔReturn pp": None,
                "ΔHitRate pp": None,
                "ΔMaxDD pp": None,
            }
        )

        # Matching keyword rows
        match = df_kw[
            (df_kw["market"] == b["market"])
            & (df_kw["freq"] == b["freq"])
            & (df_kw["strategy"] == b["strategy"])
        ]
        for _, k in match.iterrows():
            all_rows.append(
                {
                    "Market": k["market"],
                    "Freq": k["freq"],
                    "Strategy": k["strategy"],
                    "Source": k["source"],
                    "Model": k["model"],
                    "Sharpe": k["sharpe"],
                    "Return %": k["return"],
                    "HitRate %": k["hitrate"],
                    "MaxDD %": k["maxdd"],
                    "ΔSharpe": k["sharpe"] - b["sharpe"] if pd.notna(k["sharpe"]) and pd.notna(b["sharpe"]) else None,
                    "ΔReturn pp": k["return"] - b["return"] if pd.notna(k["return"]) and pd.notna(b["return"]) else None,
                    "ΔHitRate pp": k["hitrate"] - b["hitrate"] if pd.notna(k["hitrate"]) and pd.notna(b["hitrate"]) else None,
                    "ΔMaxDD pp": k["maxdd"] - b["maxdd"] if pd.notna(k["maxdd"]) and pd.notna(b["maxdd"]) else None,
                }
            )

    return pd.DataFrame(all_rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply number formatting and arrows/colors for deltas.
    """
    def fmt_num(x, digits=2, pct=False):
        if pd.isna(x):
            return "—"
        return f"{x:.{digits}f}{'%' if pct else ''}"

    def fmt_delta(x, inverse=False):
        if pd.isna(x):
            return "—"
        arrow = "↑" if (x > 0 and not inverse) or (x < 0 and inverse) else "↓" if x != 0 else "→"
        color = "green" if (x > 0 and not inverse) or (x < 0 and inverse) else "red" if x != 0 else "grey"
        return f"<span style='color:{color}'>{arrow} {x:.2f}</span>"

    out = df.copy()
    out["Sharpe"] = out["Sharpe"].map(lambda v: fmt_num(v))
    out["Return %"] = out["Return %"].map(lambda v: fmt_num(v, pct=True))
    out["HitRate %"] = out["HitRate %"].map(lambda v: fmt_num(v, pct=True))
    out["MaxDD %"] = out["MaxDD %"].map(lambda v: fmt_num(v, pct=True))

    out["ΔSharpe"] = out["ΔSharpe"].map(lambda v: fmt_delta(v))
    out["ΔReturn pp"] = out["ΔReturn pp"].map(lambda v: fmt_delta(v))
    out["ΔHitRate pp"] = out["ΔHitRate pp"].map(lambda v: fmt_delta(v))
    out["ΔMaxDD pp"] = out["ΔMaxDD pp"].map(lambda v: fmt_delta(v, inverse=True))

    return out


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="GT Markets · Metrics Comparison", layout="wide")
st.title("GT Markets · Model & Strategy Metrics")

kind = st.radio("Metrics type", ["baseline + keywords"], index=0)
freq = st.radio("Frequency", ["D", "W"], index=0)

df_base = load_metrics("baseline", freq)
df_kw = load_metrics("keywords", freq)

if df_base.empty and df_kw.empty:
    st.error("No metrics files found in artefacts.")
    st.stop()

df_all = combine_with_deltas(df_base, df_kw)

# Sidebar filters
markets = st.sidebar.multiselect("Market", sorted(df_all["Market"].unique()), default=sorted(df_all["Market"].unique()))
strategies = st.sidebar.multiselect("Strategy", sorted(df_all["Strategy"].unique()), default=sorted(df_all["Strategy"].unique()))
models = st.sidebar.multiselect("Model", sorted(df_all["Model"].unique()), default=sorted(df_all["Model"].unique()))

df_filt = df_all[
    df_all["Market"].isin(markets) & df_all["Strategy"].isin(strategies) & df_all["Model"].isin(models)
]

# Sort by ΔSharpe then ΔReturn
df_filt = df_filt.sort_values(by=["ΔSharpe", "ΔReturn pp"], ascending=[False, False])

# Format for display
df_disp = format_table(df_filt)

st.markdown("### Metrics Table with Baseline vs Keywords Comparison")
st.write("Baseline rows show raw metrics. Keyword rows show raw metrics + deltas vs the matching baseline.")
st.write(df_disp.to_html(escape=False, index=False), unsafe_allow_html=True)

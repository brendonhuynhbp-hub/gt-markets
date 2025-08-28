import os
import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq

from src.preprocess.config import (
    ASSETS, KEYWORDS, START_DATE, END_DATE,
    DRIVE_PROJECT_DIR, LOCAL_PROJECT_DIR,
    TRENDS_GEO, TRENDS_CAT
)

# ----------------- Drive mount & project dir -----------------
def get_project_root():
    """Mount Google Drive in Colab (if available) and return the project root path."""
    in_colab = False
    try:
        import google.colab  # type: ignore
        in_colab = True
    except Exception:
        pass

    if in_colab:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        root = DRIVE_PROJECT_DIR
    else:
        root = LOCAL_PROJECT_DIR

    # Ensure subfolders exist
    os.makedirs(os.path.join(root, "data", "raw"), exist_ok=True)
    os.makedirs(os.path.join(root, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(root, "reports", "figures"), exist_ok=True)
    return root

# ----------------- Trends helpers (5-year stitch) -----------------
def daterange_chunks(start, end, max_days=1825):  # ~5 years
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    while s < e:
        chunk_end = min(s + pd.Timedelta(days=max_days), e)
        yield s.date().isoformat(), chunk_end.date().isoformat()
        # keep a 30-day overlap
        s = chunk_end - pd.Timedelta(days=30)

def scale_join(a: pd.Series, b: pd.Series) -> pd.Series:
    overlap = a.index.intersection(b.index)
    if len(overlap) >= 7 and a.loc[overlap].mean() > 0 and b.loc[overlap].mean() > 0:
        factor = (a.loc[overlap].median() / b.loc[overlap].median())
        b = b * factor
    out = pd.concat([a[~a.index.isin(overlap)], b]).sort_index()
    return out[~out.index.duplicated(keep="last")]

def get_trends_series(keyword: str, start: str, end: str, geo: str, cat: int) -> pd.Series:
    py = TrendReq(hl="en-US", tz=0)
    chunks = []
    for s, e in daterange_chunks(start, end):
        py.build_payload([keyword], timeframe=f"{s} {e}", geo=geo, cat=cat)
        part = py.interest_over_time().drop(columns=["isPartial"], errors="ignore")
        chunks.append(part[keyword])
    series = chunks[0]
    for nxt in chunks[1:]:
        series = scale_join(series, nxt)
    series.name = keyword
    return series.asfreq("D").ffill()  # daily

def get_trends_matrix(keywords, start, end, geo, cat) -> pd.DataFrame:
    cols = []
    for kw in keywords:
        try:
            s = get_trends_series(kw, start, end, geo, cat)
            cols.append(s)
        except Exception as e:
            print(f"[warn] trends failed for '{kw}': {e}")
    return pd.concat(cols, axis=1) if cols else pd.DataFrame()

# ----------------- Prices -----------------
def get_prices(assets: dict, start: str, end: str) -> pd.DataFrame:
    df = yf.download(list(assets.keys()), start=start, end=end, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.rename(columns=assets).dropna(how="all")
    return df.asfreq("D").ffill()

# ----------------- Technical features -----------------
def add_technicals(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[f"{c}_ret1"] = out[c].pct_change()
        out[f"{c}_sma20"] = out[c].rolling(20).mean()
        # basic RSI
        gains = out[c].pct_change().clip(lower=0).rolling(14).mean()
        losses = out[c].pct_change().clip(upper=0).abs().rolling(14).mean()
        rs = gains / (losses.replace(0, np.nan))
        out[f"{c}_rsi14"] = 100 - (100 / (1 + rs))
    return out

# ----------------- Main -----------------
def main():
    root = get_project_root()
    raw_dir = os.path.join(root, "data", "raw")
    proc_dir = os.path.join(root, "data", "processed")

    # Fetch data
    prices = get_prices(ASSETS, START_DATE, END_DATE)
    trends = get_trends_matrix(KEYWORDS, START_DATE, END_DATE, TRENDS_GEO, TRENDS_CAT)

    # Save raw data
    prices.to_csv(os.path.join(raw_dir, "prices_close.csv"))
    if not trends.empty:
        trends.to_csv(os.path.join(raw_dir, "google_trends.csv"))

    # Merge & add features
    merged = prices.join(trends, how="inner")
    merged = add_technicals(merged, cols=list(ASSETS.values()))
    merged = merged.dropna()

    # Save processed dataset
    out_path = os.path.join(proc_dir, "merged_daily.csv")
    merged.to_csv(out_path)
    print(f"[ok] wrote {out_path} with shape {merged.shape}")

if __name__ == "__main__":
    main()

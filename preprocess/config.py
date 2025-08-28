
# ============================
# Project Configuration
# ============================

# Yahoo Finance tickers  →  friendly names used in columns
ASSETS = {
    "GC=F":    "XAUUSD",   # Gold Futures (proxy for XAU/USD)
    "CNY=X":   "USDCNY",   # USD/CNY exchange rate
    "BTC-USD": "BTCUSD",   # Bitcoin in USD
    "CL=F":    "OILUSD",   # WTI Crude Oil Futures
}

# Seed Google Trends keywords (expand/refine later)
KEYWORDS = [
    "gold price",
    "usd cny",
    "bitcoin price",
    "us inflation",
    "interest rates",
]

# Analysis window
START_DATE = "2015-01-01"
END_DATE   = "2025-01-01"

# ---------- Storage roots ----------
# In Colab this folder will be created inside your Google Drive.
# TIP: Create this folder in Drive first so you can see outputs easily.
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/gt-markets-data"

# When running locally (not in Colab), files will be written here.
LOCAL_PROJECT_DIR = "."

# ---------- Google Trends options ----------
# geo: "" for worldwide, or "US", "AU", etc.
TRENDS_GEO = ""
# cat: 0 = all categories; leave 0 unless you need to filter
TRENDS_CAT = 0

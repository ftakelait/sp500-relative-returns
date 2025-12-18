#!/usr/bin/env python
# coding: utf-8

# ## S&P 500 by date
# 
# Get snapshot of S&P 500 components at a given date; `2022-01-01`.
# 
# 
# **Reference**
# 
# Htun, H. H., Biehl, M., & Petkov, N. (2024). *Forecasting relative returns for S&P 500 stocks using machine learning.*  
# **Financial Innovation, 10**, 118.  
# [PDF link](https://link.springer.com/content/pdf/10.1186/s40854-024-00644-0.pdf)  
# 
# ---

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

# Pandas display options (keep for Jupyter notebook compatibility)
# pd.options.mode.chained_assignment = None  # silence SettingWithCopyWarning
# pd.set_option("display.max_rows", 600)
# pd.set_option("display.expand_frame_repr", False)

# Better terminal display formatting
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# Directories - all relative to script location
DATA_DIR = Path(__file__).parent / "data"
FMP_STOCKS_DIR = DATA_DIR / "fmp_stocks"
OUTPUT_DIR = Path(__file__).parent / "output"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FMP_STOCKS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: tqdm progress bar
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# Filenames
INDEX_FILE = "S&P 500 Historical Components & Changes(11-16-2025).csv"
SNP_DATA_FILE = "sp500.csv"

# --- Constants from the paper ---
# Date to use as snapshot of S&P 500 components (as in the paper)
SNAPSHOT_DATE = "2022-01-01"

# Date range for fetching stock price data
START_DATE = "2017-01-01"
END_DATE   = "2022-01-01"  # paper uses data up to 2022-01-01 

# Lags used as FEATURES (Table 2): 260,180,150,120,100,80,60,40,20,15,10,5,1 
FEATURE_LAGS = [260, 180, 150, 120, 100, 80, 60, 40, 20, 15, 10, 5, 1]

# For labels: horizon and threshold d
HORIZON_DAYS = 10          # "ten trading days (horizon)" 
THRESHOLD_D  = 2.0         # 2% (they work in percent)


def load_historical_components(path: Path) -> pd.DataFrame:
    """
    Load the historical S&P 500 components table.
    Assumes a CSV with at least:
      - 'date' column (string or datetime)
      - 'tickers' column (comma-separated symbols)
    Returns a DataFrame indexed by datetime 'date', sorted.
    """
    df = pd.read_csv(path)

    # Ensure we have the expected columns
    if "date" not in df.columns or "tickers" not in df.columns:
        raise ValueError("CSV must contain 'date' and 'tickers' columns")

    # Convert 'date' to datetime and set as index
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Convert each tickers cell from "AAPL,MSFT,..." → sorted list
    df["tickers"] = df["tickers"].apply(
        lambda x: sorted(str(x).split(","))
    )

    return df


# Load historical components
index_path = DATA_DIR / INDEX_FILE
hist_df = load_historical_components(index_path)

# Add n_tickers column for display
hist_df["n_tickers"] = hist_df["tickers"].apply(len)

# ====== TABLE 1 REPLICATION ======
# Examples of stocks in different sectors of the S&P 500 index
print("\n=== S&P 500 Historical Components (Table 1 Replication) ===")
print("Examples of stocks in different sectors of the S&P 500 index\n")

# Show recent rows with 505 tickers (descending by date, show first 7 tickers only)
recent_505 = hist_df[hist_df["n_tickers"] == 505].sort_index(ascending=False).head(5).copy()
recent_505["tickers_preview"] = recent_505["tickers"].apply(lambda x: ", ".join(x[:7]) + ", ...")
recent_505_display = recent_505.reset_index()[["date", "tickers_preview", "n_tickers"]]
recent_505_display.columns = ["date", "tickers", "n_tickers"]
print(recent_505_display.to_string(index=False))
print()

# Filter rows up to the SNAPSHOT_DATE and take the last row
# This gives us the S&P 500 composition as of 2022-01-01 (should be 505 tickers)
df_snapshot = hist_df.loc[hist_df.index <= SNAPSHOT_DATE]

if df_snapshot.empty:
    raise ValueError(f"No rows found on or before {SNAPSHOT_DATE}")

last_row = df_snapshot.tail(1)              # still indexed by date
last_row.index.name = None                  # remove index name 'date' if you ever display it
tickers_snapshot = last_row["tickers"].iloc[0]  # list of tickers on snapshot date

print(f"Number of tickers on {SNAPSHOT_DATE}: {len(tickers_snapshot)}")
print(tickers_snapshot[:10], "...")

# Verify we have exactly 505 tickers as expected
assert len(tickers_snapshot) == 505, f"Expected 505 tickers on {SNAPSHOT_DATE}, got {len(tickers_snapshot)}"


# --- Ticker normalization / aliasing  (for FMP or other APIs) --------------

# Known renames where the same company continues under a new ticker.
# This list can be extended as you discover more cases.
TICKER_ALIASES = {
    "FB":   "META",  # Facebook → Meta Platforms
    "ANTM": "ELV",   # Anthem → Elevance Health
    "VIAC": "PARA",  # ViacomCBS → Paramount Global
    "NLOK": "GEN",   # NortonLifeLock → Gen Digital
    "FBHS": "FBIN",  # Fortune Brands Home & Security → Fortune Brands Innovations

    # Extra renames / rebrandings you added
    "ABC":  "COR",   # AmerisourceBergen → Cencora Inc.
    "BLL":  "BALL",  # Ball Corp, BLL → BALL
    "PKI":  "RVTY",  # PerkinElmer → Revvity
    "RE":   "EG",    # Everest Re Group → Everest Group
    "WLTW": "WTW",   # Willis Towers Watson → WTW
}

def normalize_for_fmp(symbol: str) -> str:
    """
    Map an S&P-style ticker symbol to the symbol used with your data source:

      - apply alias mapping for renamed tickers
      - convert share-class dots to hyphens (e.g. 'BRK.B' -> 'BRK-B')

    This is useful for APIs that expect hyphenated share classes.
    """
    mapped = TICKER_ALIASES.get(symbol, symbol)
    mapped = mapped.replace(".", "-")  # convert 'BRK.B' -> 'BRK-B'
    return mapped


# Normalizing snapshot tickers
# (run this after you have `tickers_snapshot` defined)
fmp_tickers_snapshot = [normalize_for_fmp(s) for s in tickers_snapshot]
fmp_tickers_snapshot_unique = sorted(set(fmp_tickers_snapshot))

print(f"Original snapshot tickers: {len(tickers_snapshot)}")
print(f"Unique normalized tickers after aliasing: {len(fmp_tickers_snapshot_unique)}")
print(fmp_tickers_snapshot_unique[:10], "...")


# Put your real API key here (or load it from an environment variable)
FMP_KEY = "mTaUiQEsDpR2ol3XLglZu1o9vJ4lt7eD"


def fetch_fmp_sector(symbol: str,
                     api_key: str = FMP_KEY,
                     session: requests.Session | None = None) -> pd.DataFrame:
    """
    Fetch the sector for a single ticker from FinancialModelingPrep.

    Parameters
    ----------
    symbol : str
        Ticker symbol as used by FMP (e.g., 'AAPL', 'MSFT', 'BRK.B').
    api_key : str, optional
        FMP API key. Defaults to global FMP_KEY.
    session : requests.Session or None, optional
        If provided, this session is used for the HTTP request (more efficient
        when calling this function many times in a loop). If None, a one-off
        requests.get is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            ['symbol', 'sector']
        If anything goes wrong or sector is missing, returns an empty DataFrame.
    """
    symbol = str(symbol).strip()
    if not symbol:
        return pd.DataFrame()

    # FMP profile endpoint (stable)
    base_url = "https://financialmodelingprep.com/stable/profile"

    params = {
        "symbol": symbol,
        "apikey": api_key,
    }

    # Allow caller to reuse a Session for efficiency
    client = session if session is not None else requests

    try:
        r = client.get(base_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return pd.DataFrame()

    # FMP returns a LIST with one element
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame()

    profile = data[0]
    sector = profile.get("sector", None)

    if sector is None:
        return pd.DataFrame()

    df = pd.DataFrame(
        [{
            "symbol": symbol,
            "sector": sector
        }],
        columns=["symbol", "sector"],
    )

    return df


# ====== SECTOR TABLE (Table 1b) ======
# Fetch sector data for all tickers using FMP API
print("\n" + "="*70)
print("=== SECTOR TABLE (Companies by Sector) ===")
print("="*70)
print("Fetching sector data from FMP API...")

sector_dfs = []
sector_errors = []

# Use a session for efficiency
with requests.Session() as session:
    for sym in tqdm(fmp_tickers_snapshot_unique, desc="Fetching sector data"):
        df_sector = fetch_fmp_sector(sym, session=session)
        if not df_sector.empty:
            sector_dfs.append(df_sector)
        else:
            sector_errors.append(sym)

# Combine all sector DataFrames
if sector_dfs:
    sector_df = pd.concat(sector_dfs, ignore_index=True)
else:
    sector_df = pd.DataFrame(columns=["symbol", "sector"])

# Print errors after fetching (if any)
if sector_errors:
    print(f"\n[FMP] No sector data for {len(sector_errors)} tickers: {sector_errors[:10]}{'...' if len(sector_errors) > 10 else ''}")

# Group by sector and create summary table
sector_summary = (
    sector_df.groupby("sector")["symbol"]
    .apply(lambda x: ", ".join(sorted(x.tolist())[:5]) + ", ..." if len(x) > 5 else ", ".join(sorted(x.tolist())))
    .reset_index()
)
sector_summary.columns = ["Sector", "Companies (symbol)"]

# Add count column
sector_counts = sector_df.groupby("sector")["symbol"].count().reset_index()
sector_counts.columns = ["Sector", "n_companies"]
sector_summary = sector_summary.merge(sector_counts, on="Sector")

# Sort by n_companies descending
sector_summary = sector_summary.sort_values("n_companies", ascending=False).reset_index(drop=True)

print("\n" + sector_summary.to_string(index=True))
print(f"\nUnique sectors: {len(sector_summary)}  |  Total companies: {sector_summary['n_companies'].sum()}")
print()

# Save Table 1 to txt file
table1_path = OUTPUT_DIR / "table1_sectors.txt"
with open(table1_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("TABLE 1: Examples of stocks in different sectors of the S&P 500 index\n")
    f.write("=" * 70 + "\n\n")
    f.write(sector_summary.to_string(index=True))
    f.write(f"\n\nUnique sectors: {len(sector_summary)}  |  Total companies: {sector_summary['n_companies'].sum()}\n")
print(f"[OK] Saved Table 1 to: {table1_path}")


def fetch_fmp_(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily end-of-day prices from FMP and return a DataFrame with:
        ['Date', 'Close']
    filtered to the window [start, end] (inclusive).
    """
    url = (
        "https://financialmodelingprep.com/stable/"
        f"historical-price-eod/full?symbol={symbol}"
        f"&from={start}&to={end}&apikey={FMP_KEY}"
    )

    r = requests.get(url)
    data = r.json()

    # FMP returns a LIST of dicts:
    # [{'symbol': 'CDAY', 'date': '2017-01-03', 'open': ..., 'close': ...}, ...]
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(), f"[FMP] No usable data for {symbol}: {data}"

    df = pd.DataFrame(data)

    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame(), f"[FMP] Unexpected schema for {symbol}: columns={df.columns.tolist()}"

    df = df.rename(columns={"date": "Date", "close": "Close"})
    df["Date"] = pd.to_datetime(df["Date"])

    start_ts = pd.to_datetime(start)
    end_ts   = pd.to_datetime(end)

    mask = (df["Date"] >= start_ts) & (df["Date"] <= end_ts)
    df = df.loc[mask].sort_values("Date").reset_index(drop=True)

    # Keep only the columns we care about
    return df[["Date", "Close"]], None


# Test fetch with single ticker
print("\n" + "="*70)
print("=== TESTING FMP API CONNECTION ===")
print("="*70)

ticker = "GOOGL"  # or any symbol you want to inspect
test0, test_err = fetch_fmp_(ticker, START_DATE, END_DATE)

if test0.empty:
    print(f"No data returned for {ticker} from FMP")
else:
    min_date = test0["Date"].min()
    max_date = test0["Date"].max()

    summary_test = pd.DataFrame(
        [{
            "symbol": ticker,
            "min_date": min_date,
            "max_date": max_date,
            "n_obs": len(test0),
        }]
    )

    # display() replaced with print() for terminal
    print(summary_test.to_string(index=False))


# Fetch price data for all tickers to identify which ones have issues
print("\n" + "="*70)
print("=== FETCHING PRICE DATA FOR ALL TICKERS ===")
print("="*70)

summaries, missing_fmp, fetch_errors = [], [], []

for ticker in tqdm(fmp_tickers_snapshot_unique, desc="Fetching ticker summaries"):
    df, err = fetch_fmp_(ticker, START_DATE, END_DATE)

    if df.empty:
        # Record the ticker and error message (print later)
        missing_fmp.append(ticker)
        if err:
            fetch_errors.append(err)
        continue

    summaries.append(
        {
            "symbol": ticker,
            "min_date": df["Date"].min(),
            "max_date": df["Date"].max(),
            "n_obs": len(df),
        }
    )

summary_df = pd.DataFrame(summaries).sort_values("symbol").reset_index(drop=True)

# Print any FMP errors that occurred during fetching
if fetch_errors:
    print(f"\n=== FMP FETCH ERRORS ({len(fetch_errors)}) ===")
    for err in fetch_errors:
        print(err)

print("\n" + "="*70)
print("=== PRICE DATA SUMMARY ===")
print("="*70)
print(f"Summary shape: {summary_df.shape}")


# display() replaced with print() for terminal
print(summary_df.head().to_string(index=False))


print(f"n_obs range: min={summary_df.n_obs.min()}, max={summary_df.n_obs.max()}")


# How many tickers have fewer than 1259 observations?
n_less_than_full = (summary_df["n_obs"] < 1259).sum()
print("Tickers with n_obs < 1259:", n_less_than_full)


# Filter tickers with fewer than 1259 observations
print("\n" + "="*70)
print("=== TICKERS WITH INSUFFICIENT DATA ===")
print("="*70)

subset = summary_df[summary_df["n_obs"] < 1259].copy()

# Overall combined time range across these tickers
overall_min_date = subset["min_date"].min()
overall_max_date = subset["max_date"].max()

print("Number of tickers with n_obs < 1259:", len(subset))
print("Combined time range for these tickers:")
print("  earliest min_date:", overall_min_date)
print("  latest   max_date:", overall_max_date)


print(subset.sort_values("min_date").reset_index(drop=True).to_string())


# 10 tickers with short history (from your summary_df)
short_history_df = summary_df[summary_df["n_obs"] < 1259].copy()
short_history_tickers = short_history_df["symbol"].tolist()

print("Short-history tickers:", len(short_history_tickers))
print(short_history_tickers)


# 'missing_fmp' is the list of tickers where fetch_fmp_ returned empty
print("No-data tickers:", len(missing_fmp), missing_fmp)

# INFO should be the 1 ticker with no data
bad_tickers = set(short_history_tickers) | set(missing_fmp)
print("Total bad tickers:", len(bad_tickers))
print(sorted(bad_tickers))


final_494_tickers = sorted(
    set(fmp_tickers_snapshot_unique) - bad_tickers
)

print("Final ticker count:", len(final_494_tickers))
print(final_494_tickers[:10], "...")

# ====== IMPORTANT VERIFICATION ======
# Expected: 505 initial -> 10 short history -> 1 missing (INFO) -> 494 final
print("\n" + "="*70)
print("=== TICKER COUNT VERIFICATION ===")
print("="*70)
print(f"Initial unique tickers: {len(fmp_tickers_snapshot_unique)}")
print(f"Short history tickers (n_obs < 1259): {len(short_history_tickers)}")
print(f"Missing/no-data tickers: {len(missing_fmp)}")
print(f"Bad tickers to remove: {len(bad_tickers)}")
print(f"Final tickers: {len(final_494_tickers)}")
expected_final = len(fmp_tickers_snapshot_unique) - len(bad_tickers)
print(f"Expected final: {expected_final}")

# Verify the expected counts: 505 -> (10 short + 1 INFO) = 494
assert len(fmp_tickers_snapshot_unique) == 505, f"Expected 505 unique tickers, got {len(fmp_tickers_snapshot_unique)}"
assert len(short_history_tickers) == 10, f"Expected 10 short-history tickers, got {len(short_history_tickers)}"
assert len(missing_fmp) == 1, f"Expected 1 missing ticker (INFO), got {len(missing_fmp)}: {missing_fmp}"
assert len(final_494_tickers) == 494, f"Expected 494 final tickers, got {len(final_494_tickers)}"


# freeze the universe for reproducibility
final_path = DATA_DIR / "sp500_final_494_tickers_fmp.csv"

pd.DataFrame({"symbol": final_494_tickers}).to_csv(final_path, index=False)

print(f"\nSaved final tickers to: {final_path}")


# Now download price data for the final 494 tickers
print("\n" + "="*70)
print(f"=== DOWNLOADING PRICE DATA FOR {len(final_494_tickers)} TICKERS ===")
print("="*70)

fmp_missing = []  # tickers where FMP returned nothing
downloaded_count = 0
skipped_count = 0

for symbol in final_494_tickers:
    csv_path = FMP_STOCKS_DIR / f"{symbol}.csv"
    
    # If already cached, skip
    if csv_path.exists():
        skipped_count += 1
        continue

    df, err = fetch_fmp_(symbol, START_DATE, END_DATE)

    if df.empty:
        fmp_missing.append(symbol)
        print(f"[FAILED] {symbol} -> FMP returned empty data")
        continue

    # Save per-ticker CSV for all symbols in our snapshot universe
    df.to_csv(csv_path, index=False)
    downloaded_count += 1
    print(f"[OK] Saved {symbol} -> {csv_path}")

print(f"\nDownloaded: {downloaded_count}, Skipped (cached): {skipped_count}")

# FMP per-ticker price summary
cached_count = len(list(FMP_STOCKS_DIR.glob('*.csv')))
print(f"\nSuccessfully cached price data for: {cached_count} tickers")

if fmp_missing:
    print("\nTickers with NO FMP data (should be 0 if all went well):")
    print(len(fmp_missing), sorted(fmp_missing))


# ====== FINAL VALIDATION ======
# Verify ALL 494 final tickers have exactly 1259 observations with correct date range

print("\n" + "="*70)
print(f"=== VALIDATING ALL {len(final_494_tickers)} FINAL TICKERS ===")
print("="*70)

validation_rows = []

# Expected values
EXPECTED_N_OBS = 1259
EXPECTED_MIN_DATE = pd.to_datetime("2017-01-03")  # First trading day of 2017
EXPECTED_MAX_DATE = pd.to_datetime("2021-12-31")  # Last trading day before 2022-01-01

for sym in final_494_tickers:
    csv_path = FMP_STOCKS_DIR / f"{sym}.csv"

    if not csv_path.exists():
        validation_rows.append({
            "symbol": sym,
            "valid": False,
            "issue": "File missing",
            "min_date": pd.NaT,
            "max_date": pd.NaT,
            "n_obs": 0,
        })
        continue

    df_sym = pd.read_csv(csv_path, parse_dates=["Date"])

    if df_sym.empty:
        validation_rows.append({
            "symbol": sym,
            "valid": False,
            "issue": "Empty file",
            "min_date": pd.NaT,
            "max_date": pd.NaT,
            "n_obs": 0,
        })
        continue

    min_date = df_sym["Date"].min()
    max_date = df_sym["Date"].max()
    n_obs = len(df_sym)

    # Check for issues
    issues = []
    if n_obs != EXPECTED_N_OBS:
        issues.append(f"n_obs={n_obs} (expected {EXPECTED_N_OBS})")
    if min_date != EXPECTED_MIN_DATE:
        issues.append(f"min_date={min_date.date()} (expected {EXPECTED_MIN_DATE.date()})")
    if max_date != EXPECTED_MAX_DATE:
        issues.append(f"max_date={max_date.date()} (expected {EXPECTED_MAX_DATE.date()})")

    validation_rows.append({
        "symbol": sym,
        "valid": len(issues) == 0,
        "issue": "; ".join(issues) if issues else "OK",
        "min_date": min_date,
        "max_date": max_date,
        "n_obs": n_obs,
    })

validation_df = pd.DataFrame(validation_rows)

# Summary
n_valid = validation_df["valid"].sum()
n_invalid = len(validation_df) - n_valid

print(f"\n=== VALIDATION SUMMARY ===")
print(f"Total tickers validated: {len(validation_df)}")
print(f"Valid (1259 obs, correct dates): {n_valid}")
print(f"Invalid: {n_invalid}")

if n_invalid > 0:
    print("\n=== INVALID TICKERS ===")
    invalid_df = validation_df[~validation_df["valid"]].reset_index(drop=True)
    print(invalid_df.to_string())
else:
    print(f"\n[OK] All 494 tickers have exactly {EXPECTED_N_OBS} observations ({EXPECTED_MIN_DATE.date()} to {EXPECTED_MAX_DATE.date()})!")

# Display random sample as verification
import random
sample_tickers = random.sample(final_494_tickers, min(5, len(final_494_tickers)))
print(f"\n=== RANDOM SAMPLE VERIFICATION ({len(sample_tickers)} tickers) ===")
sample_df = validation_df[validation_df["symbol"].isin(sample_tickers)][["symbol", "n_obs", "min_date", "max_date"]]
sample_df = sample_df.copy()
sample_df["min_date"] = sample_df["min_date"].dt.date
sample_df["max_date"] = sample_df["max_date"].dt.date
print(sample_df.to_string(index=False))

# Final assertion
assert n_valid == 494, f"Expected all 494 tickers to be valid, but only {n_valid} are valid"

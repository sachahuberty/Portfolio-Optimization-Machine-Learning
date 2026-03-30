"""
Phase 1 – Bias-Immune Data Layer
=================================
Point-in-time ingestion for prices, SEC fundamentals, macro series,
and historical universe reconstitution.  Every function enforces the
rule that no observation may use data published after the decision
timestamp.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import ssl
import numpy as np
import pandas as pd
import yaml
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# Fix SSL certificate verification on macOS Python installs
# where "Install Certificates.command" hasn't been run.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "config" / "default.yaml"


def load_config(path: Path = CFG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# 1. Historical Universe Reconstitution
# ──────────────────────────────────────────────────────────────

# Fallback: current S&P 500 tickers scraped from Wikipedia.
# Production systems should replace this with a monthly-reconstituted
# membership table that includes delisted/acquired names.
_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _read_wiki_tables():
    """Fetch Wikipedia S&P 500 tables, handling SSL + User-Agent issues."""
    import requests as _req
    from io import StringIO
    resp = _req.get(_WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def _fetch_current_sp500() -> pd.DataFrame:
    """Scrape current constituents + historical changes from Wikipedia."""
    tables = _read_wiki_tables()
    current = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry",
                          "Date added", "CIK"]].copy()
    current.rename(columns={"Symbol": "ticker", "GICS Sector": "sector",
                             "Security": "company"}, inplace=True)
    current["ticker"] = current["ticker"].str.replace(".", "-", regex=False)
    return current


def _fetch_sp500_changes() -> pd.DataFrame:
    """Historical additions/removals from the second Wikipedia table.
    Returns DataFrame with columns: date, added, removed."""
    tables = _read_wiki_tables()
    if len(tables) < 2:
        return pd.DataFrame(columns=["date", "added", "removed"])

    raw = tables[1].copy()

    # Wikipedia uses MultiIndex columns: ('Added','Ticker'), ('Removed','Ticker'), etc.
    if isinstance(raw.columns, pd.MultiIndex):
        date_col = [c for c in raw.columns if "date" in str(c).lower()][0]
        added_col = [c for c in raw.columns if "added" in str(c[0]).lower() and "ticker" in str(c[1]).lower()]
        removed_col = [c for c in raw.columns if "removed" in str(c[0]).lower() and "ticker" in str(c[1]).lower()]

        out = pd.DataFrame({
            "date": raw[date_col],
            "added": raw[added_col[0]] if added_col else np.nan,
            "removed": raw[removed_col[0]] if removed_col else np.nan,
        })
    else:
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
        out = raw.rename(columns={
            c: "date" for c in raw.columns if "date" in c
        }).rename(columns={
            c: "added" for c in raw.columns if "added" in c
        }).rename(columns={
            c: "removed" for c in raw.columns if "removed" in c
        })

    return out[["date", "added", "removed"]].dropna(subset=["date"])


def build_historical_universe(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Reconstruct a monthly (ticker, date) membership table.

    Uses current constituents as the anchor and walks backwards through
    the historical changes table.  Tickers that were *removed* from the
    index are re-added for the months they were members, which mitigates
    survivorship bias relative to naively using today's list.

    Returns
    -------
    DataFrame with columns ['date', 'ticker', 'sector'] indexed monthly.
    """
    current = _fetch_current_sp500()
    changes = _fetch_sp500_changes()

    months = pd.date_range(start_date, end_date, freq="ME")
    tickers_now = set(current["ticker"].tolist())
    sector_map = dict(zip(current["ticker"], current["sector"]))

    membership: Dict[pd.Timestamp, set] = {}

    for m in reversed(months):
        membership[m] = set(tickers_now)

        if "date" in changes.columns:
            month_changes = changes[
                pd.to_datetime(changes["date"], errors="coerce").dt.to_period("M")
                == m.to_period("M")
            ]
            for _, row in month_changes.iterrows():
                added = str(row.get("added", "")).replace(".", "-")
                removed = str(row.get("removed", "")).replace(".", "-")
                if added in tickers_now:
                    tickers_now.discard(added)
                if removed and removed != "nan":
                    tickers_now.add(removed)

    rows = []
    for dt, members in sorted(membership.items()):
        for t in members:
            rows.append({"date": dt, "ticker": t,
                         "sector": sector_map.get(t, "Unknown")})

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 2. Market Data (yfinance)
# ──────────────────────────────────────────────────────────────

def fetch_pricing(
    tickers: List[str],
    start: str,
    end: str,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Download adjusted daily OHLCV for a list of tickers.

    Returns long-form DataFrame:
        date | ticker | open | high | low | close | adj_close | volume
    """
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        raw = yf.download(
            batch, start=start, end=end,
            group_by="ticker", auto_adjust=False, threads=True,
        )
        if raw.empty:
            continue

        # Normalise columns: yfinance may return flat Index (1 ticker)
        # or MultiIndex (multiple tickers).  Flatten to (ticker, field).
        if isinstance(raw.columns, pd.MultiIndex):
            # Newer yfinance: level-0 may be "Ticker" or "Price",
            # level-1 the other.  Detect which level holds ticker symbols.
            lvl0_vals = raw.columns.get_level_values(0).unique().tolist()
            lvl1_vals = raw.columns.get_level_values(1).unique().tolist()
            tickers_in_0 = any(t in lvl0_vals for t in batch)
            tickers_in_1 = any(t in lvl1_vals for t in batch)

            if tickers_in_1 and not tickers_in_0:
                raw = raw.swaplevel(axis=1)
        else:
            # Single ticker, flat columns – wrap into MultiIndex
            raw.columns = pd.MultiIndex.from_product(
                [batch, [str(c) for c in raw.columns]]
            )

        for t in batch:
            try:
                if t not in raw.columns.get_level_values(0):
                    continue
                sub = raw[t].dropna(how="all").copy()
            except (KeyError, TypeError):
                continue
            sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
            sub["ticker"] = t
            sub.index.name = "date"
            sub = sub.reset_index()
            frames.append(sub)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    # ── Filter corrupted / recycled tickers ──
    # These are delisted S&P 500 ticker symbols that the data provider
    # maps to unrelated penny stocks or OTC shells, producing nonsensical
    # returns (e.g. 34,000x daily moves) that corrupt the entire pipeline.
    _BAD_TICKERS = {
        "BMC", "CBE", "COL", "CPWR", "GR", "MI",
        "PTV", "RSH", "RX", "SLE", "STI", "SW", "TIE",
    }
    before = df["ticker"].nunique()
    df = df[~df["ticker"].isin(_BAD_TICKERS)]
    after = df["ticker"].nunique()
    if before != after:
        print(f"  Filtered {before - after} corrupted tickers: {_BAD_TICKERS & set(df['ticker'])}")

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 3. Point-in-Time Fundamentals (SEC EDGAR companyfacts API)
# ──────────────────────────────────────────────────────────────

def _fetch_cik_map() -> Dict[str, str]:
    """Download the SEC ticker → CIK mapping (one-time, cached in memory)."""
    import requests

    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "QuantFramework research@university.edu"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Map uppercase ticker → zero-padded CIK string
    return {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in data.values()
    }


# XBRL US-GAAP tags to extract, with fallbacks for tag name variations.
# Each key becomes a column; values are tried in order until one is found.
_XBRL_TAG_MAP: Dict[str, list] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}


def _parse_companyfacts(
    ticker: str,
    cik: str,
    filing_types: tuple,
    since_dt: pd.Timestamp,
) -> list[dict]:
    """Fetch and parse the companyfacts JSON for one ticker."""
    import requests

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": "QuantFramework research@university.edu"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    us_gaap = data.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return []

    # Build a dict: (form, end_date) → {metric: value, filed: date}
    # so we can merge multiple tags into a single row per filing period.
    filing_map: dict[tuple, dict] = {}
    forms_set = set(filing_types)

    for col_name, tag_candidates in _XBRL_TAG_MAP.items():
        for tag in tag_candidates:
            concept = us_gaap.get(tag)
            if concept is None:
                continue
            facts = concept.get("units", {}).get("USD", [])
            for fact in facts:
                form = fact.get("form", "")
                if form not in forms_set:
                    continue
                end = fact.get("end")
                filed = fact.get("filed")
                val = fact.get("val")
                if end is None or filed is None or val is None:
                    continue
                if pd.Timestamp(filed) < since_dt:
                    continue

                key = (form, end)
                if key not in filing_map:
                    filing_map[key] = {
                        "ticker": ticker,
                        "fiscal_period_end": end,
                        "sec_acceptance_date": filed,
                        "form_type": form,
                    }
                # First tag candidate to populate a (key, metric) wins;
                # later candidates fill gaps (e.g. Revenues covers 10-K
                # pre-2018, RevenueFromContract... covers 10-Q and post-2018).
                if col_name not in filing_map[key]:
                    filing_map[key][col_name] = val

    return list(filing_map.values())


def fetch_pit_fundamentals(
    tickers: List[str],
    filing_types: tuple = ("10-K", "10-Q"),
    since: str = "2010-01-01",
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Pull key fundamental metrics from SEC EDGAR via the bulk
    companyfacts API (one request per ticker).

    For each ticker, fetches ALL historical XBRL data in a single
    JSON response and extracts revenue, net income, total assets,
    total liabilities, stockholders equity, and operating cash flow
    for every 10-K and 10-Q filing back to *since*.

    Parameters
    ----------
    tickers : list of ticker strings
    filing_types : tuple of SEC form types to keep (default: 10-K and 10-Q)
    since : earliest filing date to include (default: "2010-01-01")
    max_workers : concurrent requests (default: 8, within SEC 10 req/s limit)

    Returns DataFrame:
        ticker | fiscal_period_end | sec_acceptance_date |
        form_type | revenue | net_income | total_assets |
        total_liabilities | stockholders_equity | operating_cash_flow
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    since_dt = pd.Timestamp(since)

    # Step 1: Get ticker → CIK mapping
    print("  [SEC] Fetching ticker → CIK mapping …")
    try:
        cik_map = _fetch_cik_map()
    except Exception as e:
        warnings.warn(f"Failed to fetch CIK mapping: {e}", stacklevel=2)
        return pd.DataFrame()

    # Resolve CIKs for requested tickers
    ticker_cik = {}
    unmapped = []
    for t in tickers:
        cik = cik_map.get(t.upper())
        if cik:
            ticker_cik[t] = cik
        else:
            unmapped.append(t)

    if unmapped:
        print(f"  [SEC] No CIK found for {len(unmapped)} tickers: "
              f"{unmapped[:10]}{'…' if len(unmapped) > 10 else ''}")

    # Step 2: Fetch companyfacts in parallel with rate limiting
    print(f"  [SEC] Fetching companyfacts for {len(ticker_cik)} tickers "
          f"(~{len(ticker_cik) // max_workers + 1}s) …")

    all_rows: list[dict] = []
    failed: list[str] = []
    completed = 0

    def _fetch_one(ticker_cik_pair):
        ticker, cik = ticker_cik_pair
        return ticker, _parse_companyfacts(ticker, cik, filing_types, since_dt)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, item in enumerate(ticker_cik.items()):
            futures[pool.submit(_fetch_one, item)] = item[0]
            # Throttle submission to stay under 10 req/sec
            if (i + 1) % max_workers == 0:
                time.sleep(1.0)

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, rows = future.result()
                all_rows.extend(rows)
            except Exception as e:
                failed.append(f"{ticker}({type(e).__name__})")
            completed += 1
            if completed % 100 == 0:
                print(f"    … {completed}/{len(ticker_cik)} tickers done")

    if failed:
        print(f"  [SEC] Failed for {len(failed)} tickers: "
              f"{failed[:10]}{'…' if len(failed) > 10 else ''}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    for col in ["fiscal_period_end", "sec_acceptance_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Deduplicate: keep one row per (ticker, fiscal_period_end, form_type).
    # Prefer the row with the most non-null metrics.
    metric_cols = list(_XBRL_TAG_MAP.keys())
    df["_n_metrics"] = df[metric_cols].notna().sum(axis=1)
    df = df.sort_values("_n_metrics", ascending=False).drop_duplicates(
        subset=["ticker", "fiscal_period_end", "form_type"],
        keep="first",
    ).drop(columns=["_n_metrics"])

    df = df.sort_values(["ticker", "sec_acceptance_date"]).reset_index(drop=True)
    print(f"  [SEC] Done: {len(df):,} filing rows for {df['ticker'].nunique()} tickers")
    return df


# Duration metrics (income statement / cash flow) are cumulative YTD in
# 10-Qs, so we source them from 10-K only.  Instant metrics (balance
# sheet) are point-in-time snapshots safe to use from any form type.
_DURATION_COLS = ["revenue", "net_income", "operating_cash_flow"]
_INSTANT_COLS = ["total_assets", "total_liabilities", "stockholders_equity"]


def _asof_merge_per_ticker(
    left: pd.DataFrame,
    right: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    """merge_asof per ticker, keeping only *value_cols* from the right side."""
    right = right[["ticker", "sec_acceptance_date"] + value_cols].copy()
    right = right.dropna(subset=["sec_acceptance_date"])

    # Align datetime resolution to avoid merge_asof dtype mismatch
    left["date"] = left["date"].astype("datetime64[us]")
    right["sec_acceptance_date"] = right["sec_acceptance_date"].astype("datetime64[us]")

    parts = []
    for ticker in left["ticker"].unique():
        l = left[left["ticker"] == ticker].sort_values("date")
        r = right[right["ticker"] == ticker].sort_values("sec_acceptance_date")
        if r.empty:
            parts.append(l)
            continue
        m = pd.merge_asof(
            l, r,
            left_on="date",
            right_on="sec_acceptance_date",
            direction="backward",
            suffixes=("", "_fund"),
        )
        parts.append(m)

    if not parts:
        return left
    merged = pd.concat(parts, ignore_index=True)
    for col in ["ticker_fund", "sec_acceptance_date_fund"]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)
    return merged


def pit_asof_join(
    feature_dates: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every (ticker, decision_date) in *feature_dates*, attach the
    most recent fundamental row whose sec_acceptance_date <= decision_date.

    Duration metrics (revenue, net_income, operating_cash_flow) are sourced
    from 10-K filings only to avoid cumulative YTD distortion from 10-Qs.
    Instant metrics (total_assets, total_liabilities, stockholders_equity)
    use both 10-K and 10-Q for quarterly freshness.

    Parameters
    ----------
    feature_dates : DataFrame with columns ['ticker', 'date']
    fundamentals  : DataFrame from fetch_pit_fundamentals()

    Returns
    -------
    Merged DataFrame.  Any fundamental column that violates PIT is NaN.
    """
    if fundamentals.empty:
        return feature_dates

    left = feature_dates.copy()
    fund = fundamentals.copy()

    left["date"] = pd.to_datetime(left["date"], utc=True).dt.tz_localize(None)
    fund["sec_acceptance_date"] = pd.to_datetime(
        fund["sec_acceptance_date"], utc=True
    ).dt.tz_localize(None)

    left = left.dropna(subset=["date"])

    # Duration metrics: 10-K only (annual figures, no YTD mixing)
    duration_present = [c for c in _DURATION_COLS if c in fund.columns]
    annual = fund[fund["form_type"] == "10-K"].copy()

    # Instant metrics: all form types (balance sheet snapshots)
    instant_present = [c for c in _INSTANT_COLS if c in fund.columns]

    # Merge duration metrics from 10-K
    if duration_present and not annual.empty:
        merged = _asof_merge_per_ticker(left, annual, duration_present)
    else:
        merged = left

    # Merge instant metrics from all filings
    if instant_present and not fund.empty:
        merged = _asof_merge_per_ticker(merged, fund, instant_present)

    # Keep sec_acceptance_date and form_type from the annual merge for PIT audit
    if "sec_acceptance_date" not in merged.columns and not annual.empty:
        merged = _asof_merge_per_ticker(
            merged, annual[["ticker", "sec_acceptance_date", "form_type"]],
            ["form_type"],
        )

    # Forward-fill per-metric gaps within each ticker.
    # merge_asof picks the single best-matching filing row, but that row
    # may be missing a metric that an earlier filing reported (e.g., a
    # 10-Q that lacks total_liabilities when the prior 10-K had it).
    # Propagating the last known value is PIT-safe: we only carry forward
    # values that were already public at or before the decision date.
    all_fund_cols = _DURATION_COLS + _INSTANT_COLS
    fund_present = [c for c in all_fund_cols if c in merged.columns]
    if fund_present:
        merged = merged.sort_values(["ticker", "date"])
        merged[fund_present] = merged.groupby("ticker")[fund_present].ffill()

    return merged


# ──────────────────────────────────────────────────────────────
# 4. Macro Series (FRED)
# ──────────────────────────────────────────────────────────────

def fetch_macro_series(
    series_ids: List[str],
    start: str,
    end: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull macro indicators from FRED.

    Tries pandas_datareader first (no API key needed), then fredapi.
    Returns a date-indexed DataFrame with one column per series.
    """
    # pandas_datareader works without an API key — try it first
    try:
        import pandas_datareader.data as web
        df = web.DataReader(series_ids, "fred", start, end).ffill()
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback: fredapi (requires FRED_API_KEY)
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            frames = {}
            for sid in series_ids:
                try:
                    frames[sid] = fred.get_series(sid, start, end)
                except Exception:
                    continue
            if frames:
                return pd.DataFrame(frames).ffill()
        except Exception:
            pass

    warnings.warn(
        "Could not fetch FRED data – install pandas_datareader "
        "or set FRED_API_KEY for fredapi.",
        stacklevel=2,
    )
    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# 5. Leakage Diagnostics
# ──────────────────────────────────────────────────────────────

def check_pit_compliance(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Assert that no fundamental data point has
    sec_acceptance_date > date (the decision timestamp).

    Returns a diagnostics DataFrame with violation counts and
    publication-lag statistics per ticker.
    """
    if "sec_acceptance_date" not in merged.columns:
        return pd.DataFrame({"status": ["no_fundamentals_attached"]})

    merged = merged.dropna(subset=["sec_acceptance_date"])
    merged["_pit_lag_days"] = (
        merged["date"] - merged["sec_acceptance_date"]
    ).dt.days

    violations = merged[merged["_pit_lag_days"] < 0]
    stats = merged.groupby("ticker")["_pit_lag_days"].agg(
        ["count", "mean", "median", "min"]
    )
    stats["violations"] = merged.groupby("ticker").apply(
        lambda g: (g["_pit_lag_days"] < 0).sum()
    )

    assert len(violations) == 0, (
        f"PIT VIOLATION: {len(violations)} rows have "
        f"sec_acceptance_date after decision date!"
    )

    return stats.rename(columns={
        "count": "n_obs", "mean": "avg_lag_days",
        "median": "median_lag_days", "min": "min_lag_days",
    })


def survivorship_report(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise how many unique tickers appear per month and how many
    are no longer in the current index (proxy for dead/delisted names).
    """
    current_tickers = set(_fetch_current_sp500()["ticker"].tolist())
    monthly = universe.groupby("date")["ticker"].agg(
        total=lambda x: x.nunique(),
        dead_or_removed=lambda x: sum(t not in current_tickers for t in x),
    )
    return monthly


# ──────────────────────────────────────────────────────────────
# 6. Convenience Orchestrator
# ──────────────────────────────────────────────────────────────

def run_data_pipeline(cfg: Optional[dict] = None) -> dict:
    """
    Execute the full data acquisition phase and return a dict of
    DataFrames ready for feature engineering.
    """
    cfg = cfg or load_config()
    start = cfg["universe"]["start_date"]
    end = cfg["universe"]["end_date"]

    print("[Phase 1] Building historical universe …")
    universe = build_historical_universe(start, end)
    all_tickers = sorted(universe["ticker"].unique().tolist())
    print(f"  Universe: {len(all_tickers)} unique tickers, "
          f"{len(universe)} ticker-months")

    print("[Phase 1] Downloading pricing data …")
    pricing = fetch_pricing(all_tickers, start, end)
    print(f"  Pricing: {len(pricing)} rows, "
          f"{pricing['ticker'].nunique()} tickers")

    print("[Phase 1] Fetching PIT fundamentals from SEC EDGAR …")
    fundamentals = fetch_pit_fundamentals(all_tickers[:50])
    print(f"  Fundamentals: {len(fundamentals)} filing rows")

    print("[Phase 1] Fetching macro series from FRED …")
    macro = fetch_macro_series(
        cfg["data"]["fred_series"], start, end,
        api_key=cfg["data"].get("fred_api_key"),
    )
    print(f"  Macro: {macro.shape}")

    return {
        "universe": universe,
        "pricing": pricing,
        "fundamentals": fundamentals,
        "macro": macro,
        "config": cfg,
    }

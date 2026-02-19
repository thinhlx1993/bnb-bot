"""
Data loading and signals cache for RL training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from rl_agent.config import (
    INITIAL_BALANCE,
    SIGNALS_CACHE_DIR,
    USE_BACKTEST_SIGNALS,
    USE_TECHNICAL_SIGNALS,
)
from rl_agent.signals import generate_entry_signals, generate_exit_signals

logger = logging.getLogger(__name__)


def filter_data_by_date(
    data: pd.Series,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """Filter time-series by date range (YYYY-MM-DD or None)."""
    filtered = data.copy()
    if start_date is not None:
        filtered = filtered[filtered.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        filtered = filtered[filtered.index <= pd.to_datetime(end_date)]
    return filtered


def _signals_cache_path(
    ticker: str,
    start_date: Optional[str],
    end_date: Optional[str],
    strategy: str,
    use_backtest: bool,
) -> Path:
    """Path to cache file for one ticker's signals."""
    safe_strategy = (strategy or "Combined").replace("/", "_").replace("\\", "_")
    name = f"{ticker}_{start_date or 'none'}_{end_date or 'none'}_{safe_strategy}_{use_backtest}.parquet"
    return SIGNALS_CACHE_DIR / name


def _load_signals_cache(
    cache_path: Path,
    price_index: pd.Index,
) -> Optional[Tuple[pd.Series, pd.Series]]:
    """Load entry/exit signals from cache if file exists and index matches."""
    if not cache_path.exists():
        return None
    try:
        cached = pd.read_parquet(cache_path)
        if len(cached.index) != len(price_index) or not cached.index.equals(price_index):
            return None
        return (
            cached["entry_signals"].astype(bool),
            cached["exit_signals"].astype(bool),
        )
    except Exception as e:
        logger.debug("Cache read failed %s: %s", cache_path, e)
        return None


def _save_signals_cache(
    cache_path: Path,
    entry_signals: pd.Series,
    exit_signals: pd.Series,
) -> None:
    """Save entry/exit signals to cache."""
    try:
        if entry_signals is None or exit_signals is None or not entry_signals.index.equals(exit_signals.index):
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"entry_signals": entry_signals, "exit_signals": exit_signals}).to_parquet(
            cache_path, index=True
        )
        logger.info("Cache saved: %s", cache_path.name)
    except Exception as e:
        logger.warning("Cache write failed %s: %s", cache_path, e)


def get_tickers_with_cache(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    strategy: str = "Combined",
    use_backtest: bool = True,
) -> List[str]:
    """
    Return list of tickers that have a signals cache file for the given date range and options.
    Use this to preload the available cache list before loading (faster when cache_only=True).
    """
    if not SIGNALS_CACHE_DIR.is_dir():
        return []
    safe_strategy = (strategy or "Combined").replace("/", "_").replace("\\", "_")
    suffix = f"_{start_date or 'none'}_{end_date or 'none'}_{safe_strategy}_{use_backtest}.parquet"
    tickers = []
    for path in SIGNALS_CACHE_DIR.glob("*.parquet"):
        stem = path.name
        if stem.endswith(suffix):
            ticker = stem[: -len(suffix)]
            if ticker:
                tickers.append(ticker)
    return sorted(tickers)


def load_all_tickers_data(
    tickers_list: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    results_dir: Path = Path("results"),
    strategy: str = "Combined",
    cache_only: bool = False,
) -> Dict[str, Dict]:
    """
    Load price, balance, entry and exit signals for all tickers.
    Uses signals cache under data/signals_cache when available.
    Returns dict of {ticker: {'price', 'balance', 'entry_signals', 'exit_signals', 'ohlcv'}}.

    When cache_only=True, tickers without an existing cache are skipped (no signal computation),
    making the loader faster when cache was pre-built (e.g. via build_signals_cache_parallel.py).
    """
    use_backtest = USE_BACKTEST_SIGNALS
    if cache_only:
        cached_tickers = get_tickers_with_cache(
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            use_backtest=use_backtest,
        )
        tickers_list = [t for t in tickers_list if t in cached_tickers]
        logger.info(
            "Preloaded cache list: %d tickers have cache for this range (loading only these)",
            len(tickers_list),
        )
        if not tickers_list:
            logger.warning("No tickers with cache for this date range; run build_signals_cache_parallel.py")
            return {}

    date_range_str = f" ({start_date or 'start'} to {end_date or 'end'})" if (start_date or end_date) else ""
    logger.info("Loading data for %d tickers%s...", len(tickers_list), date_range_str)

    all_tickers_data = {}
    data_dir = Path("data")
    dataset_file = data_dir / "dataset.csv"

    if not dataset_file.exists():
        logger.error("Dataset file not found: %s. Run: python backtest.py --download-only", dataset_file)
        return all_tickers_data

    try:
        df = pd.read_csv(dataset_file)
        if "tic" not in df.columns or "time" not in df.columns:
            logger.error("Dataset missing 'tic' or 'time' columns")
            return all_tickers_data

        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()

        for ticker in tickers_list:
            ticker_df = df[df["tic"] == ticker].copy()
            if len(ticker_df) == 0:
                logger.warning("No data found for %s", ticker)
                continue

            if start_date is not None:
                ticker_df = ticker_df[ticker_df.index >= pd.to_datetime(start_date)]
            if end_date is not None:
                ticker_df = ticker_df[ticker_df.index <= pd.to_datetime(end_date)]
            if len(ticker_df) == 0:
                logger.warning("No data for %s in date range %s to %s", ticker, start_date, end_date)
                continue

            price_series = ticker_df["close"]
            balance_series = pd.Series(INITIAL_BALANCE, index=price_series.index)
            entry_signals = None
            exit_signals = None

            if USE_TECHNICAL_SIGNALS:
                cache_path = _signals_cache_path(
                    ticker, start_date, end_date, strategy, USE_BACKTEST_SIGNALS
                )
                cached = _load_signals_cache(cache_path, price_series.index)
                if cached is not None:
                    entry_signals, exit_signals = cached
                    logger.info(
                        "  %s: %d points, %d entry, %d exit signals (from cache)",
                        ticker, len(price_series), entry_signals.sum(), exit_signals.sum(),
                    )
                else:
                    if cache_only:
                        logger.debug("Skipping %s (no cache, cache_only=True)", ticker)
                        continue
                    if USE_BACKTEST_SIGNALS:
                        try:
                            from entry_signal_generator import get_strategy_signals
                            entry_signals, exit_signals = get_strategy_signals(
                                ticker_df, price_series, strategy=strategy
                            )
                        except Exception as e:
                            logger.warning("entry_signal_generator failed for %s: %s, using crossover", ticker, e)
                            entry_signals = generate_entry_signals(ticker_df, price_series)
                            exit_signals = generate_exit_signals(ticker_df, price_series)
                    else:
                        entry_signals = generate_entry_signals(ticker_df, price_series)
                        exit_signals = generate_exit_signals(ticker_df, price_series)
                    logger.info(
                        "  %s: %d points, %d entry, %d exit signals",
                        ticker, len(price_series), entry_signals.sum(), exit_signals.sum(),
                    )
                    _save_signals_cache(cache_path, entry_signals, exit_signals)
            else:
                logger.info("  %s: %d points (random entry)", ticker, len(price_series))

            all_tickers_data[ticker] = {
                "price": price_series,
                "balance": balance_series,
                "entry_signals": entry_signals,
                "exit_signals": exit_signals,
                "ohlcv": ticker_df,
            }
    except Exception as e:
        logger.error("Error loading dataset: %s", e, exc_info=True)

    if len(all_tickers_data) == 0:
        logger.error("No ticker data loaded!")
    else:
        logger.info("Successfully loaded data for %d tickers", len(all_tickers_data))
    return all_tickers_data

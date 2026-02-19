"""
Build signals cache for all tickers in parallel using multiple processes.
Reads each ticker from data/{ticker}.csv (no need to open dataset.csv).
Signal computation is CPU-bound (divergence detection), so we use processes
instead of threads to get real parallelism (threads are limited by the GIL).

Usage:
  python build_signals_cache_parallel.py
  python build_signals_cache_parallel.py --start-date 2025-01-01 --end-date 2026-01-01
  python build_signals_cache_parallel.py --all-ranges   # train, val, and eval ranges
  python build_signals_cache_parallel.py --workers 60
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from rl_agent.config import (
    EVAL_END_DATE,
    EVAL_START_DATE,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
    USE_BACKTEST_SIGNALS,
    USE_TECHNICAL_SIGNALS,
    VAL_END_DATE,
    VAL_START_DATE,
)
from rl_agent.data import (
    _load_signals_cache,
    _save_signals_cache,
    _signals_cache_path,
)
from rl_agent.signals import generate_entry_signals, generate_exit_signals

Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/build_signals_cache_parallel.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

STRATEGY = "Combined"
# CPU-bound work: use processes. Default = min(60, CPU count) to avoid thrashing on small machines.
DEFAULT_WORKERS = min(60, os.cpu_count() or 32)


def _compute_and_save_signals(
    ticker: str,
    ticker_df: pd.DataFrame,
    price_series: pd.Series,
    start_date: Optional[str],
    end_date: Optional[str],
    strategy: str,
    use_backtest: bool,
) -> Tuple[str, bool, Optional[str]]:
    """
    For one ticker: load from cache or compute entry/exit signals and save.
    Returns (ticker, from_cache, error_message).
    """
    cache_path = _signals_cache_path(ticker, start_date, end_date, strategy, use_backtest)
    cached = _load_signals_cache(cache_path, price_series.index)
    if cached is not None:
        entry_signals, exit_signals = cached
        return (
            ticker,
            True,
            None,
        )

    try:
        if use_backtest:
            try:
                from entry_signal_generator import get_strategy_signals
                entry_signals, exit_signals = get_strategy_signals(
                    ticker_df, price_series, strategy=strategy
                )
            except Exception as e:
                logger.warning(
                    "entry_signal_generator failed for %s: %s, using crossover",
                    ticker,
                    e,
                )
                entry_signals = generate_entry_signals(ticker_df, price_series)
                exit_signals = generate_exit_signals(ticker_df, price_series)
        else:
            entry_signals = generate_entry_signals(ticker_df, price_series)
            exit_signals = generate_exit_signals(ticker_df, price_series)

        _save_signals_cache(cache_path, entry_signals, exit_signals)
        n_entry = int(entry_signals.sum())
        n_exit = int(exit_signals.sum())
        logger.info(
            "  %s: %d points, %d entry, %d exit signals",
            ticker,
            len(price_series),
            n_entry,
            n_exit,
        )
        return (ticker, False, None)
    except Exception as e:
        logger.exception("Failed %s: %s", ticker, e)
        return (ticker, False, str(e))


def get_tickers_from_data_folder(data_dir: Path) -> List[str]:
    """Discover tickers from data/{ticker}.csv files (excludes dataset.csv)."""
    if not data_dir.is_dir():
        return []
    tickers = []
    for p in data_dir.glob("*.csv"):
        if p.name.lower() == "dataset.csv":
            continue
        tickers.append(p.stem)
    return sorted(tickers)


def _worker(
    ticker: str,
    data_dir: str,
    start_date: Optional[str],
    end_date: Optional[str],
    strategy: str,
    use_backtest: bool,
) -> Tuple[str, bool, Optional[str]]:
    """Process worker: read ticker CSV, filter by date, build cache (no dataset.csv)."""
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{ticker}.csv"
    if not csv_path.exists():
        return (ticker, False, "no csv file")

    try:
        ticker_df = pd.read_csv(csv_path)
    except Exception as e:
        return (ticker, False, f"read csv: {e}")

    if "time" not in ticker_df.columns or "close" not in ticker_df.columns:
        return (ticker, False, "missing time/close columns")
    ticker_df["time"] = pd.to_datetime(ticker_df["time"])
    ticker_df = ticker_df.set_index("time").sort_index()

    if len(ticker_df) == 0:
        return (ticker, False, "no data")

    if start_date is not None:
        ticker_df = ticker_df[ticker_df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        ticker_df = ticker_df[ticker_df.index <= pd.to_datetime(end_date)]
    if len(ticker_df) == 0:
        return (ticker, False, "no data in range")

    price_series = ticker_df["close"]
    return _compute_and_save_signals(
        ticker,
        ticker_df,
        price_series,
        start_date,
        end_date,
        strategy,
        use_backtest,
    )


def build_cache_for_range(
    data_dir: Path,
    tickers_list: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
    strategy: str = STRATEGY,
    use_backtest: bool = True,
    n_workers: int = DEFAULT_WORKERS,
) -> Dict[str, Any]:
    """Build signals cache for one date range in parallel (processes; reads each data/{ticker}.csv). Returns stats."""
    date_str = f"{start_date or 'start'} to {end_date or 'end'}"
    logger.info(
        "Building cache for %d tickers (%s) with %d processes (per-ticker CSV)...",
        len(tickers_list),
        date_str,
        n_workers,
    )
    built = 0
    from_cache = 0
    skipped_no_data = 0  # no data in range (e.g. newer tickers) - normal, not a failure
    failed = 0
    errors: List[Tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _worker,
                ticker,
                str(data_dir),
                start_date,
                end_date,
                strategy,
                use_backtest,
            ): ticker
            for ticker in tickers_list
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                t, from_c, err = future.result()
                if err:
                    if err == "no data in range":
                        skipped_no_data += 1
                    else:
                        failed += 1
                        errors.append((t, err))
                elif from_c:
                    from_cache += 1
                else:
                    built += 1
            except Exception as e:
                failed += 1
                errors.append((ticker, str(e)))
                logger.exception("Worker failed for %s", ticker)

    for t, err in errors:
        logger.warning("  %s: %s", t, err)
    if skipped_no_data:
        logger.info(
            "Range %s done: %d built, %d from cache, %d skipped (no data in range), %d failed",
            date_str,
            built,
            from_cache,
            skipped_no_data,
            failed,
        )
    else:
        logger.info(
            "Range %s done: %d built, %d from cache, %d failed",
            date_str,
            built,
            from_cache,
            failed,
        )
    return {
        "built": built,
        "from_cache": from_cache,
        "skipped_no_data": skipped_no_data,
        "failed": failed,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build RL signals cache in parallel using multiple processes (CPU-bound signal computation)."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: VAL_START_DATE for single range)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: VAL_END_DATE for single range)",
    )
    parser.add_argument(
        "--all-ranges",
        action="store_true",
        help="Build cache for train, val, and eval date ranges",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel processes (default: {DEFAULT_WORKERS}, use CPU count for CPU-bound work)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=STRATEGY,
        help=f"Strategy name for cache key (default: {STRATEGY})",
    )
    args = parser.parse_args()

    if not USE_TECHNICAL_SIGNALS:
        logger.warning("USE_TECHNICAL_SIGNALS is False; cache is not used by training.")
    use_backtest = USE_BACKTEST_SIGNALS

    data_dir = Path("data")
    if not data_dir.is_dir():
        logger.error("Data directory not found: %s", data_dir)
        return

    tickers_list = get_tickers_from_data_folder(data_dir)
    if not tickers_list:
        logger.error("No ticker CSVs found in %s (expected data/{TICKER}.csv)", data_dir)
        return
    logger.info("Found %d ticker CSVs in data/ (dataset.csv not loaded)", len(tickers_list))

    if args.all_ranges:
        ranges = [
            ("train", TRAIN_START_DATE, TRAIN_END_DATE),
            ("val", VAL_START_DATE, VAL_END_DATE),
            ("eval", EVAL_START_DATE, EVAL_END_DATE),
        ]
        for name, start, end in ranges:
            logger.info("--- %s range: %s to %s ---", name, start, end)
            build_cache_for_range(
                data_dir,
                tickers_list,
                start,
                end,
                strategy=args.strategy,
                use_backtest=use_backtest,
                n_workers=args.workers,
            )
    else:
        start_date = args.start_date or VAL_START_DATE
        end_date = args.end_date or VAL_END_DATE
        build_cache_for_range(
            data_dir,
            tickers_list,
            start_date,
            end_date,
            strategy=args.strategy,
            use_backtest=use_backtest,
            n_workers=args.workers,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()

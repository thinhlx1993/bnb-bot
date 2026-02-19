"""Live data fetching and signal generation."""

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Tuple

import pandas as pd
import requests

from live_trading.config import LOCAL_TIMEZONE
from entry_signal_generator import get_strategy_signals

if TYPE_CHECKING:
    from live_trading.binance_trader import BinanceTrader

logger = logging.getLogger(__name__)


def fetch_live_data(
    trader: "BinanceTrader",
    ticker_list: list,
    time_interval: str,
    lookback_periods: int = 500,
) -> pd.DataFrame:
    """
    Fetch live data directly from Binance API (real-time).
    Returns DataFrame with OHLCV data in FinRL format.
    """
    all_data = []
    try:
        for ticker in ticker_list:
            if trader.client:
                klines = trader.client.get_klines(symbol=ticker, interval=time_interval, limit=lookback_periods)
            else:
                url = f"{trader.base_url}/api/v3/klines"
                params = {
                    "symbol": ticker,
                    "interval": time_interval,
                    "limit": min(lookback_periods, 1000),
                }
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                klines = response.json()
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base",
                    "taker_buy_quote", "ignore",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["time"] = df["timestamp"]
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df["tic"] = ticker
            df = df[["time", "tic", "open", "high", "low", "close", "volume"]]
            all_data.append(df)
        if not all_data:
            logger.error("No data fetched from Binance")
            return pd.DataFrame()
        combined_df = pd.concat(all_data, ignore_index=True)
        latest_time = combined_df["time"].max()
        if isinstance(latest_time, pd.Timestamp):
            latest_time_utc7 = latest_time.tz_localize("UTC").tz_convert("Asia/Bangkok") if latest_time.tz is None else latest_time.tz_convert("Asia/Bangkok")
        else:
            latest_time_utc7 = pd.to_datetime(latest_time).tz_localize("UTC").tz_convert("Asia/Bangkok")
        min_time = combined_df["time"].min()
        if isinstance(min_time, pd.Timestamp):
            min_time_utc7 = min_time.tz_localize("UTC").tz_convert("Asia/Bangkok") if min_time.tz is None else min_time.tz_convert("Asia/Bangkok")
        else:
            min_time_utc7 = pd.to_datetime(min_time).tz_localize("UTC").tz_convert("Asia/Bangkok")
        max_time = combined_df["time"].max()
        if isinstance(max_time, pd.Timestamp):
            max_time_utc7 = max_time.tz_localize("UTC").tz_convert("Asia/Bangkok") if max_time.tz is None else max_time.tz_convert("Asia/Bangkok")
        else:
            max_time_utc7 = pd.to_datetime(max_time).tz_localize("UTC").tz_convert("Asia/Bangkok")
        logger.info(f"📊 Fetched data: Latest candle at {latest_time_utc7.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
        logger.info(f"📊 Current time: {datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')} (Local UTC+7)")
        logger.info(f"📊 Data range: {min_time_utc7.strftime('%Y-%m-%d %H:%M:%S')} to {max_time_utc7.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
        return combined_df
    except Exception as e:
        logger.error(f"Error fetching live data from Binance: {e}")
        logger.error("Falling back to FinRL-Meta data source...")
        try:
            from meta.data_processor import DataProcessor
            from meta.data_processors._base import DataSource
            end_date = datetime.now()
            if "m" in time_interval:
                minutes = int(time_interval.replace("m", ""))
                start_date = end_date - timedelta(minutes=minutes * lookback_periods)
            elif "h" in time_interval:
                hours = int(time_interval.replace("h", ""))
                start_date = end_date - timedelta(hours=hours * lookback_periods)
            elif "d" in time_interval:
                days = int(time_interval.replace("d", ""))
                start_date = end_date - timedelta(days=days * lookback_periods)
            else:
                start_date = end_date - timedelta(days=30)
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            dp = DataProcessor(
                data_source=DataSource.binance,
                start_date=start_date_str,
                end_date=end_date_str,
                time_interval=time_interval,
            )
            dp.download_data(ticker_list=ticker_list)
            dp.clean_data()
            return dp.dataframe
        except Exception as e2:
            logger.error(f"Error with FinRL-Meta fallback: {e2}")
            return pd.DataFrame()


def generate_signals(df: pd.DataFrame, ticker: str, strategy: str = "Combined") -> Tuple[pd.Series, pd.Series]:
    """
    Generate buy/sell signals for a ticker using shared entry_signal_generator.
    Returns (entries, exits) boolean series.
    """
    ticker_df = df[df["tic"] == ticker].copy()
    if ticker_df.empty:
        return pd.Series(), pd.Series()
    ticker_df["time"] = pd.to_datetime(ticker_df["time"])
    ticker_df = ticker_df.set_index("time").sort_index()
    price = ticker_df["close"]
    return get_strategy_signals(ticker_df, price, strategy=strategy)

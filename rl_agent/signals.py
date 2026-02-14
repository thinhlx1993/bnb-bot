"""
Technical indicators and entry/exit signal generation for RL training.
Used when not using the shared entry_signal_generator (USE_BACKTEST_SIGNALS=False).
"""

import logging

import numpy as np
import pandas as pd

from rl_agent.config import (
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
)

logger = logging.getLogger(__name__)


def calculate_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
):
    """Calculate MACD, Signal, and Histogram."""
    close = df["close"] if "close" in df.columns else df
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_rsi(price: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate RSI (Relative Strength Index)."""
    delta = price.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))


def identify_macd_signals(
    price: pd.Series,
    macd: pd.Series,
    signal: pd.Series,
    histogram: pd.Series,
) -> pd.Series:
    """Identify MACD-based entry signals (bullish crossover). Returns boolean series."""
    macd_cross_up = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    hist_positive = (histogram > 0) & (histogram.shift(1) <= 0)
    entries = macd_cross_up | hist_positive
    return entries.fillna(False)


def identify_rsi_signals(price: pd.Series, rsi: pd.Series) -> pd.Series:
    """Identify RSI-based entry signals (oversold bounce). Returns boolean series."""
    rsi_cross_up = (rsi > RSI_OVERSOLD) & (rsi.shift(1) <= RSI_OVERSOLD)
    rsi_rising = (
        (rsi < 50)
        & (rsi > rsi.shift(1))
        & (rsi.shift(1) < RSI_OVERSOLD + 10)
    )
    entries = rsi_cross_up | rsi_rising
    return entries.fillna(False)


def identify_macd_exit_signals(
    price: pd.Series,
    macd: pd.Series,
    signal: pd.Series,
    histogram: pd.Series,
) -> pd.Series:
    """Identify MACD-based exit signals (bearish crossover). Returns boolean series."""
    macd_cross_down = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    hist_negative = (histogram < 0) & (histogram.shift(1) >= 0)
    exits = macd_cross_down | hist_negative
    return exits.fillna(False)


def identify_rsi_exit_signals(price: pd.Series, rsi: pd.Series) -> pd.Series:
    """Identify RSI-based exit signals (overbought reversal). Returns boolean series."""
    rsi_cross_down = (rsi < RSI_OVERBOUGHT) & (rsi.shift(1) >= RSI_OVERBOUGHT)
    rsi_falling = (
        (rsi > 50)
        & (rsi < rsi.shift(1))
        & (rsi.shift(1) > RSI_OVERBOUGHT - 10)
    )
    exits = rsi_cross_down | rsi_falling
    return exits.fillna(False)


def generate_entry_signals(ticker_df: pd.DataFrame, price: pd.Series) -> pd.Series:
    """Generate combined entry signals from MACD and RSI."""
    entries = pd.Series(False, index=price.index)
    try:
        macd, signal, histogram = calculate_macd(ticker_df)
        entries = entries | identify_macd_signals(price, macd, signal, histogram)
        rsi = calculate_rsi(price)
        entries = entries | identify_rsi_signals(price, rsi)
    except Exception as e:
        logger.warning("Error generating entry signals: %s", e)
        return pd.Series(False, index=price.index)
    return entries.fillna(False)


def generate_exit_signals(ticker_df: pd.DataFrame, price: pd.Series) -> pd.Series:
    """Generate combined exit signals from MACD and RSI."""
    exits = pd.Series(False, index=price.index)
    try:
        macd, signal, histogram = calculate_macd(ticker_df)
        exits = exits | identify_macd_exit_signals(price, macd, signal, histogram)
        rsi = calculate_rsi(price)
        exits = exits | identify_rsi_exit_signals(price, rsi)
    except Exception as e:
        logger.warning("Error generating exit signals: %s", e)
        return pd.Series(False, index=price.index)
    return exits.fillna(False)

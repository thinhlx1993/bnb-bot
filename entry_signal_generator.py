"""
Shared entry/exit signal generator for training, evaluation, and live trading.
Uses backtest strategy logic (divergence-based MACD/RSI and optional bullish confirmation).
"""

import pandas as pd
from typing import Tuple


def get_strategy_signals(
    ticker_df: pd.DataFrame,
    price: pd.Series,
    strategy: str = "Combined",
) -> Tuple[pd.Series, pd.Series]:
    """
    Get entry and exit signals for a strategy.
    Same logic as used in evaluate_rl_agent and backtest (divergence-based).

    Args:
        ticker_df: DataFrame with OHLCV data (must have 'close' or use price index).
        price: Price series (typically ticker_df['close']).
        strategy: Strategy name: "Combined", "MACD_Trend_Reversal", "RSI_Trend_Reversal", "Bullish_Trend_Confirmation", "EMA_25_99_Crossover".

    Returns:
        entries, exits: Boolean series aligned with price index.
    """
    from backtest import (
        calculate_macd,
        identify_trend_reversals,
        calculate_rsi,
        identify_rsi_trend_reversals,
        identify_bullish_trend_confirmation,
        identify_ema_25_99_crossover,
        ENABLE_MACD_TREND_REVERSAL,
        ENABLE_RSI_TREND_REVERSAL,
        ENABLE_BULLISH_CONFIRMATION,
        ENABLE_EMA_25_99_CROSSOVER,
    )

    entries = pd.Series(False, index=price.index)
    exits = pd.Series(False, index=price.index)

    if strategy == "Combined" or strategy == "MACD_Trend_Reversal":
        if ENABLE_MACD_TREND_REVERSAL:
            macd, signal, histogram = calculate_macd(ticker_df)
            macd_signals = identify_trend_reversals(price, macd, signal, histogram)
            entries = entries | macd_signals.get(
                "strong_bullish", pd.Series(False, index=price.index)
            )
            exits = exits | macd_signals.get(
                "strong_bearish", pd.Series(False, index=price.index)
            )

    if strategy == "Combined" or strategy == "RSI_Trend_Reversal":
        if ENABLE_RSI_TREND_REVERSAL:
            rsi = calculate_rsi(price)
            rsi_signals = identify_rsi_trend_reversals(price, rsi)
            entries = entries | rsi_signals.get(
                "bullish_reversal", pd.Series(False, index=price.index)
            )
            exits = exits | rsi_signals.get(
                "bearish_reversal", pd.Series(False, index=price.index)
            )

    if strategy == "Combined" or strategy == "Bullish_Trend_Confirmation":
        if ENABLE_BULLISH_CONFIRMATION:
            bullish_signals = identify_bullish_trend_confirmation(price)
            entries = entries | bullish_signals.get(
                "bullish_reversal", pd.Series(False, index=price.index)
            )
            exits = exits | bullish_signals.get(
                "bearish_reversal", pd.Series(False, index=price.index)
            )

    if strategy == "Combined" or strategy == "EMA_25_99_Crossover":
        if ENABLE_EMA_25_99_CROSSOVER:
            ema_25_99_signals = identify_ema_25_99_crossover(price)
            entries = entries | ema_25_99_signals.get(
                "entry", pd.Series(False, index=price.index)
            )
            exits = exits | ema_25_99_signals.get(
                "exit", pd.Series(False, index=price.index)
            )

    return entries.fillna(False), exits.fillna(False)

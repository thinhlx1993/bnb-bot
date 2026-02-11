"""
Shared entry/exit signal generator for training, evaluation, and live trading.
Uses backtest strategy logic (divergence-based: MACD, RSI, Stochastic, OBV, MFI, CCI, Williams, TSI, ROC, A/D, PVT).
"""

import pandas as pd
from typing import Tuple
from backtest import (
    calculate_macd,
    identify_trend_reversals,
    calculate_rsi,
    identify_rsi_trend_reversals,
    identify_stochastic_divergence,
    identify_obv_divergence,
    identify_mfi_divergence,
    identify_cci_divergence,
    identify_williams_divergence,
    identify_tsi_divergence,
    identify_roc_divergence,
    identify_ad_divergence,
    identify_pvt_divergence,
    ENABLE_MACD_TREND_REVERSAL,
    ENABLE_RSI_TREND_REVERSAL,
    ENABLE_STOCHASTIC_DIVERGENCE,
    ENABLE_OBV_DIVERGENCE,
    ENABLE_MFI_DIVERGENCE,
    ENABLE_CCI_DIVERGENCE,
    ENABLE_WILLIAMS_DIVERGENCE,
    ENABLE_TSI_DIVERGENCE,
    ENABLE_ROC_DIVERGENCE,
    ENABLE_AD_DIVERGENCE,
    ENABLE_PVT_DIVERGENCE,
    STOCH_K_PERIOD,
    STOCH_D_PERIOD,
    MFI_PERIOD,
    CCI_PERIOD,
    WILLIAMS_PERIOD,
    TSI_FAST,
    TSI_SLOW,
    ROC_PERIOD,
)

def get_strategy_signals(
    ticker_df: pd.DataFrame,
    price: pd.Series,
    strategy: str = "Combined",
) -> Tuple[pd.Series, pd.Series]:
    """
    Get entry and exit signals for a strategy.
    Same logic as used in evaluate_rl_agent and backtest (divergence-based).

    Args:
        ticker_df: DataFrame with OHLCV data (must have 'close'; volume required for OBV, MFI, A/D, PVT).
        price: Price series (typically ticker_df['close']).
        strategy: "Combined" or any single strategy: "MACD_Trend_Reversal", "RSI_Trend_Reversal",
                  "Stochastic_Divergence", "OBV_Divergence", "MFI_Divergence", "CCI_Divergence",
                  "Williams_Divergence", "TSI_Divergence", "ROC_Divergence", "AD_Divergence", "PVT_Divergence".

    Returns:
        entries, exits: Boolean series aligned with price index.
    """
    entries = pd.Series(False, index=price.index)
    exits = pd.Series(False, index=price.index)
    empty = pd.Series(False, index=price.index)

    def add(sig):
        nonlocal entries, exits
        if sig is None:
            return
        entries = entries | sig.get("bullish_reversal", empty).fillna(False)
        exits = exits | sig.get("bearish_reversal", empty).fillna(False)

    if strategy == "Combined" or strategy == "MACD_Trend_Reversal":
        if ENABLE_MACD_TREND_REVERSAL:
            macd, signal, histogram = calculate_macd(ticker_df)
            s = identify_trend_reversals(price, macd, signal, histogram)
            entries |= s.get("strong_bullish", empty).fillna(False)
            exits |= s.get("strong_bearish", empty).fillna(False)

    if strategy == "Combined" or strategy == "RSI_Trend_Reversal":
        if ENABLE_RSI_TREND_REVERSAL:
            rsi = calculate_rsi(price)
            add(identify_rsi_trend_reversals(price, rsi))

    if strategy == "Combined" or strategy == "Stochastic_Divergence":
        if ENABLE_STOCHASTIC_DIVERGENCE:
            add(identify_stochastic_divergence(price, ticker_df, STOCH_K_PERIOD, STOCH_D_PERIOD))

    if strategy == "Combined" or strategy == "OBV_Divergence":
        if ENABLE_OBV_DIVERGENCE and "volume" in ticker_df.columns:
            add(identify_obv_divergence(price, ticker_df))

    if strategy == "Combined" or strategy == "MFI_Divergence":
        if ENABLE_MFI_DIVERGENCE and "volume" in ticker_df.columns:
            add(identify_mfi_divergence(price, ticker_df, MFI_PERIOD))

    if strategy == "Combined" or strategy == "CCI_Divergence":
        if ENABLE_CCI_DIVERGENCE:
            add(identify_cci_divergence(price, ticker_df, CCI_PERIOD))

    if strategy == "Combined" or strategy == "Williams_Divergence":
        if ENABLE_WILLIAMS_DIVERGENCE:
            add(identify_williams_divergence(price, ticker_df, WILLIAMS_PERIOD))

    if strategy == "Combined" or strategy == "TSI_Divergence":
        if ENABLE_TSI_DIVERGENCE:
            add(identify_tsi_divergence(price, TSI_FAST, TSI_SLOW))

    if strategy == "Combined" or strategy == "ROC_Divergence":
        if ENABLE_ROC_DIVERGENCE:
            add(identify_roc_divergence(price, ROC_PERIOD))

    if strategy == "Combined" or strategy == "AD_Divergence":
        if ENABLE_AD_DIVERGENCE and "volume" in ticker_df.columns:
            add(identify_ad_divergence(price, ticker_df))

    if strategy == "Combined" or strategy == "PVT_Divergence":
        if ENABLE_PVT_DIVERGENCE and "volume" in ticker_df.columns:
            add(identify_pvt_divergence(price, ticker_df))

    return entries.fillna(False), exits.fillna(False)

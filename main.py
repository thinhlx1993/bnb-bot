"""
MACD Trend Reversal Strategy using VectorBT
This strategy spots trend reversals using MACD Divergence detection:

1. Bullish Divergence: Price makes lower low, but MACD histogram makes higher low
   - Indicates bearish momentum fading, upward reversal coming (BUY signal)

2. Bearish Divergence: Price makes higher high, but MACD histogram makes lower high
   - Indicates bullish momentum weakening, downward reversal coming (SELL signal)
"""

import sys
import os
from pathlib import Path

# Add FinRL-Meta to sys.path
finrl_meta_path = Path("/mnt/data/FinRL-Tutorials/3-Practical/FinRL-Meta")
if str(finrl_meta_path) not in sys.path:
    sys.path.insert(0, str(finrl_meta_path))

import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
from meta.data_processor import DataProcessor
from meta.data_processors._base import DataSource

# Configuration
START_DATE = "2026-01-01" # YYYY-MM-DD
END_DATE = "2026-01-15" # YYYY-MM-DD
TIME_INTERVAL = "15m" # 1m, 5m, 15m, 30m, 1h, 1d, etc.
TICKER_LIST = ["BTCUSDT", "ETHUSDT"]  # You can add more cryptocurrencies

# MACD Parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Trend Filter Parameters
USE_MA99_FILTER = False        # Enable EMA trend filter
MA99_PERIOD = 50              # EMA period (default: 50, originally 99)

# Strategy Configuration - Enable/Disable Strategies
ENABLE_MACD_TREND_REVERSAL = True   # Strategy 1: MACD Trend Reversals (Divergence)
ENABLE_RSI_TREND_REVERSAL = True    # Strategy 2: RSI Trend Reversals (Divergence)
ENABLE_BULLISH_CONFIRMATION = False  # Strategy 3: Bullish Trend Confirmation (EMA Crossover)

# RSI Parameters
RSI_PERIOD = 14                     # RSI calculation period
RSI_OVERSOLD = 30                   # RSI oversold level
RSI_OVERBOUGHT = 70                 # RSI overbought level

# Bullish Trend Confirmation Parameters
BULLISH_EMA_FAST = 9                # Fast EMA for bullish confirmation
BULLISH_EMA_SLOW = 21               # Slow EMA for bullish confirmation

# Portfolio Parameters
INITIAL_BALANCE = 1000.0  # Starting balance in USD ($)

# Risk Management Parameters
ENABLE_RISK_MANAGEMENT = False  # Master switch: Enable/disable all risk management features
USE_STOP_LOSS = True           # Enable stop-loss orders (requires ENABLE_RISK_MANAGEMENT = True)
STOP_LOSS_PCT = -0.1          # Stop loss at -10% (exit if loss reaches 10%)
USE_TAKE_PROFIT = True         # Enable take-profit orders (requires ENABLE_RISK_MANAGEMENT = True)
TAKE_PROFIT_PCT = 0.3         # Take profit at +30% (exit if gain reaches 30%)
USE_MAX_HOLDING = True         # Enable maximum holding period (requires ENABLE_RISK_MANAGEMENT = True)
MAX_HOLDING_PERIODS = 720         # Maximum periods to hold (e.g., 720 periods = 5 days for 15m interval)


def fetch_binance_data(ticker_list, start_date, end_date, time_interval):
    """Fetch data from Binance using FinRL-Meta datasource."""
    print(f"Fetching data from Binance for {ticker_list}...")
    
    dp = DataProcessor(
        data_source=DataSource.binance,
        start_date=start_date,
        end_date=end_date,
        time_interval=time_interval
    )
    
    dp.download_data(ticker_list=ticker_list)
    dp.clean_data()
    
    return dp.dataframe


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal, and Histogram."""
    # Ensure we have close prices
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    # Calculate MACD
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram


def calculate_ema(price, period=MA99_PERIOD):
    """Calculate Exponential Moving Average (EMA)."""
    ema = price.ewm(span=period, adjust=False).mean()
    return ema


def calculate_rsi(price, period=RSI_PERIOD):
    """Calculate Relative Strength Index (RSI)."""
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def find_local_extrema(series, window=5):
    """
    Find local minima and maxima in a time series using rolling window.
    Returns boolean series indicating local minima and maxima.
    """
    # Local minima: point is minimum in surrounding window
    local_min = pd.Series(False, index=series.index)
    local_max = pd.Series(False, index=series.index)
    
    for i in range(window, len(series) - window):
        window_data = series.iloc[i-window:i+window+1]
        current_val = series.iloc[i]
        
        # Check if current point is local minimum
        if current_val == window_data.min() and current_val < series.iloc[i-1] and current_val < series.iloc[i+1]:
            local_min.iloc[i] = True
        
        # Check if current point is local maximum
        if current_val == window_data.max() and current_val > series.iloc[i-1] and current_val > series.iloc[i+1]:
            local_max.iloc[i] = True
    
    return local_min, local_max


def identify_trend_reversals(price, macd, signal, histogram, lookback_periods=30):
    """
    Identify trend reversal signals based on MACD divergence:
    
    1. Bullish Divergence: Price makes lower low, but MACD/histogram makes higher low
       - Indicates bearish momentum fading, upward reversal coming
    
    2. Bearish Divergence: Price makes higher high, but MACD/histogram makes lower high
       - Indicates bullish momentum weakening, downward reversal coming
    """
    # Initialize signals
    bullish_divergence = pd.Series(False, index=price.index)
    bearish_divergence = pd.Series(False, index=price.index)
    
    # Use histogram for divergence detection (as specified)
    indicator = histogram
    
    # Find local extrema in price and indicator
    window = 30
    price_min, price_max = find_local_extrema(price, window=window)
    indicator_min, indicator_max = find_local_extrema(indicator, window=window)
    
    # Get indices of local extrema
    price_low_indices = price_min[price_min].index
    price_high_indices = price_max[price_max].index
    indicator_low_indices = indicator_min[indicator_min].index
    indicator_high_indices = indicator_max[indicator_max].index
    
    # Detect bullish divergence (price lower low, indicator higher low)
    for i in range(1, len(price_low_indices)):
        # Get the two most recent price lows
        price_low_idx_1 = price_low_indices[i-1]
        price_low_idx_2 = price_low_indices[i]
        
        # Find closest indicator lows to these price lows
        # Get indicator lows that occurred before or near each price low
        indicator_lows_before_1 = indicator_low_indices[indicator_low_indices <= price_low_idx_1]
        indicator_lows_before_2 = indicator_low_indices[indicator_low_indices <= price_low_idx_2]
        
        if len(indicator_lows_before_1) > 0 and len(indicator_lows_before_2) > 0:
            # Use the most recent indicator low before each price low
            indicator_low_idx_1 = indicator_lows_before_1[-1]
            indicator_low_idx_2 = indicator_lows_before_2[-1]
            
            # Ensure we're comparing different indicator lows
            if indicator_low_idx_1 != indicator_low_idx_2:
                # Check for divergence: price lower low, indicator higher low
                price_low_1 = price.loc[price_low_idx_1]
                price_low_2 = price.loc[price_low_idx_2]
                indicator_low_1 = indicator.loc[indicator_low_idx_1]
                indicator_low_2 = indicator.loc[indicator_low_idx_2]
                
                if price_low_2 < price_low_1 and indicator_low_2 > indicator_low_1:
                    # Bullish divergence detected at the second low
                    bullish_divergence.loc[price_low_idx_2] = True
    
    # Detect bearish divergence (price higher high, indicator lower high)
    for i in range(1, len(price_high_indices)):
        # Get the two most recent price highs
        price_high_idx_1 = price_high_indices[i-1]
        price_high_idx_2 = price_high_indices[i]
        
        # Find closest indicator highs to these price highs
        # Get indicator highs that occurred before or near each price high
        indicator_highs_before_1 = indicator_high_indices[indicator_high_indices <= price_high_idx_1]
        indicator_highs_before_2 = indicator_high_indices[indicator_high_indices <= price_high_idx_2]
        
        if len(indicator_highs_before_1) > 0 and len(indicator_highs_before_2) > 0:
            # Use the most recent indicator high before each price high
            indicator_high_idx_1 = indicator_highs_before_1[-1]
            indicator_high_idx_2 = indicator_highs_before_2[-1]
            
            # Ensure we're comparing different indicator highs
            if indicator_high_idx_1 != indicator_high_idx_2:
                # Check for divergence: price higher high, indicator lower high
                price_high_1 = price.loc[price_high_idx_1]
                price_high_2 = price.loc[price_high_idx_2]
                indicator_high_1 = indicator.loc[indicator_high_idx_1]
                indicator_high_2 = indicator.loc[indicator_high_idx_2]
                
                if price_high_2 > price_high_1 and indicator_high_2 < indicator_high_1:
                    # Bearish divergence detected at the second high
                    bearish_divergence.loc[price_high_idx_2] = True
    
    return {
        'bullish_reversal': bullish_divergence,
        'bearish_reversal': bearish_divergence,
        'strong_bullish': bullish_divergence,
        'strong_bearish': bearish_divergence,
        'macd': macd,
        'signal': signal,
        'histogram': histogram
    }


def identify_rsi_trend_reversals(price, rsi, lookback_periods=30):
    """
    Identify trend reversal signals based on RSI divergence:
    
    1. Bullish Divergence: Price makes lower low, but RSI makes higher low
    2. Bearish Divergence: Price makes higher high, but RSI makes lower high
    """
    # Initialize signals
    bullish_divergence = pd.Series(False, index=price.index)
    bearish_divergence = pd.Series(False, index=price.index)
    
    # Find local extrema in price and RSI
    window = 30
    price_min, price_max = find_local_extrema(price, window=window)
    rsi_min, rsi_max = find_local_extrema(rsi, window=window)
    
    # Get indices of local extrema
    price_low_indices = price_min[price_min].index
    price_high_indices = price_max[price_max].index
    rsi_low_indices = rsi_min[rsi_min].index
    rsi_high_indices = rsi_max[rsi_max].index
    
    # Detect bullish divergence (price lower low, RSI higher low)
    for i in range(1, len(price_low_indices)):
        price_low_idx_1 = price_low_indices[i-1]
        price_low_idx_2 = price_low_indices[i]
        
        rsi_lows_before_1 = rsi_low_indices[rsi_low_indices <= price_low_idx_1]
        rsi_lows_before_2 = rsi_low_indices[rsi_low_indices <= price_low_idx_2]
        
        if len(rsi_lows_before_1) > 0 and len(rsi_lows_before_2) > 0:
            rsi_low_idx_1 = rsi_lows_before_1[-1]
            rsi_low_idx_2 = rsi_lows_before_2[-1]
            
            if rsi_low_idx_1 != rsi_low_idx_2:
                price_low_1 = price.loc[price_low_idx_1]
                price_low_2 = price.loc[price_low_idx_2]
                rsi_low_1 = rsi.loc[rsi_low_idx_1]
                rsi_low_2 = rsi.loc[rsi_low_idx_2]
                
                if price_low_2 < price_low_1 and rsi_low_2 > rsi_low_1:
                    bullish_divergence.loc[price_low_idx_2] = True
    
    # Detect bearish divergence (price higher high, RSI lower high)
    for i in range(1, len(price_high_indices)):
        price_high_idx_1 = price_high_indices[i-1]
        price_high_idx_2 = price_high_indices[i]
        
        rsi_highs_before_1 = rsi_high_indices[rsi_high_indices <= price_high_idx_1]
        rsi_highs_before_2 = rsi_high_indices[rsi_high_indices <= price_high_idx_2]
        
        if len(rsi_highs_before_1) > 0 and len(rsi_highs_before_2) > 0:
            rsi_high_idx_1 = rsi_highs_before_1[-1]
            rsi_high_idx_2 = rsi_highs_before_2[-1]
            
            if rsi_high_idx_1 != rsi_high_idx_2:
                price_high_1 = price.loc[price_high_idx_1]
                price_high_2 = price.loc[price_high_idx_2]
                rsi_high_1 = rsi.loc[rsi_high_idx_1]
                rsi_high_2 = rsi.loc[rsi_high_idx_2]
                
                if price_high_2 > price_high_1 and rsi_high_2 < rsi_high_1:
                    bearish_divergence.loc[price_high_idx_2] = True
    
    return {
        'bullish_reversal': bullish_divergence,
        'bearish_reversal': bearish_divergence,
        'rsi': rsi
    }


def identify_bullish_trend_confirmation(price, ema_fast_period=BULLISH_EMA_FAST, ema_slow_period=BULLISH_EMA_SLOW):
    """
    Identify bullish trend confirmation using EMA crossover:
    - Buy when fast EMA crosses above slow EMA (golden cross)
    - Sell when fast EMA crosses below slow EMA (death cross)
    """
    ema_fast = calculate_ema(price, ema_fast_period)
    ema_slow = calculate_ema(price, ema_slow_period)
    
    # Golden cross: fast EMA crosses above slow EMA (bullish)
    bullish_signal = pd.Series(False, index=price.index)
    bearish_signal = pd.Series(False, index=price.index)
    
    # Detect crossovers
    for i in range(1, len(price)):
        # Golden cross: fast was below slow, now above
        if ema_fast.iloc[i-1] <= ema_slow.iloc[i-1] and ema_fast.iloc[i] > ema_slow.iloc[i]:
            bullish_signal.iloc[i] = True
        # Death cross: fast was above slow, now below
        elif ema_fast.iloc[i-1] >= ema_slow.iloc[i-1] and ema_fast.iloc[i] < ema_slow.iloc[i]:
            bearish_signal.iloc[i] = True
    
    return {
        'bullish_reversal': bullish_signal,
        'bearish_reversal': bearish_signal,
        'ema_fast': ema_fast,
        'ema_slow': ema_slow
    }


def apply_risk_management(entries, exits, price):
    """Apply stop-loss, take-profit, and maximum holding period to exits.
    
    Only applies if ENABLE_RISK_MANAGEMENT is True.
    """
    # If risk management is disabled, return original exits
    if not ENABLE_RISK_MANAGEMENT:
        return exits.copy()
    
    exits_with_rm = exits.copy()
    positions = []  # Track open positions: [(entry_idx, entry_price)]
    
    for i in range(len(price)):
        # Check for new entries
        if entries.iloc[i]:
            positions.append((i, price.iloc[i]))
        
        # Check open positions for exit conditions
        positions_to_remove = []
        for pos_idx, (entry_idx, entry_price) in enumerate(positions):
            if entry_idx >= i:  # Position not yet opened
                continue
            
            current_price = price.iloc[i]
            returns = (current_price - entry_price) / entry_price
            periods_held = i - entry_idx
            
            exit_triggered = False
            
            # Stop-loss check
            if USE_STOP_LOSS and returns <= STOP_LOSS_PCT:
                exits_with_rm.iloc[i] = True
                exit_triggered = True
            
            # Take-profit check
            if USE_TAKE_PROFIT and returns >= TAKE_PROFIT_PCT and not exit_triggered:
                exits_with_rm.iloc[i] = True
                exit_triggered = True
            
            # Maximum holding period check
            if USE_MAX_HOLDING and periods_held >= MAX_HOLDING_PERIODS and not exit_triggered:
                exits_with_rm.iloc[i] = True
                exit_triggered = True
            
            # Mark position for removal if exited
            if exit_triggered:
                positions_to_remove.append(pos_idx)
        
        # Remove exited positions (in reverse order to maintain indices)
        for pos_idx in reversed(positions_to_remove):
            positions.pop(pos_idx)
    
    # Print risk management summary
    original_exits = exits.sum()
    new_exits = exits_with_rm.sum()
    total_rm_exits = new_exits - original_exits
    if total_rm_exits > 0:
        print(f"  Risk management exits added: {total_rm_exits}")
        if USE_STOP_LOSS:
            print(f"    - Stop-loss: {STOP_LOSS_PCT:.1%}")
        if USE_TAKE_PROFIT:
            print(f"    - Take-profit: {TAKE_PROFIT_PCT:.1%}")
        if USE_MAX_HOLDING:
            print(f"    - Max holding: {MAX_HOLDING_PERIODS} periods")
    
    return exits_with_rm


def create_vectorbt_signals(df, all_strategies, price):
    """
    Create entry and exit signals by combining multiple strategies.
    
    Args:
        all_strategies: Dictionary with strategy names as keys and their signal dictionaries as values
        price: Price series
    
    Returns:
        entries, exits: Combined boolean series for entry and exit signals
    """
    # Initialize combined signals
    entries = pd.Series(False, index=price.index)
    exits = pd.Series(False, index=price.index)
    
    strategy_counts = {}
    
    # Combine signals from all enabled strategies (OR logic: any strategy can trigger)
    if ENABLE_MACD_TREND_REVERSAL and 'macd' in all_strategies:
        macd_signals = all_strategies['macd']
        macd_entries = macd_signals.get('strong_bullish', pd.Series(False, index=price.index)).fillna(False)
        macd_exits = macd_signals.get('strong_bearish', pd.Series(False, index=price.index)).fillna(False)
        entries = entries | macd_entries
        exits = exits | macd_exits
        strategy_counts['MACD Trend Reversal'] = {
            'entries': macd_entries.sum(),
            'exits': macd_exits.sum()
        }
    
    if ENABLE_RSI_TREND_REVERSAL and 'rsi' in all_strategies:
        rsi_signals = all_strategies['rsi']
        rsi_entries = rsi_signals.get('bullish_reversal', pd.Series(False, index=price.index)).fillna(False)
        rsi_exits = rsi_signals.get('bearish_reversal', pd.Series(False, index=price.index)).fillna(False)
        entries = entries | rsi_entries
        exits = exits | rsi_exits
        strategy_counts['RSI Trend Reversal'] = {
            'entries': rsi_entries.sum(),
            'exits': rsi_exits.sum()
        }
    
    if ENABLE_BULLISH_CONFIRMATION and 'bullish_confirmation' in all_strategies:
        bullish_signals = all_strategies['bullish_confirmation']
        bullish_entries = bullish_signals.get('bullish_reversal', pd.Series(False, index=price.index)).fillna(False)
        bullish_exits = bullish_signals.get('bearish_reversal', pd.Series(False, index=price.index)).fillna(False)
        entries = entries | bullish_entries
        exits = exits | bullish_exits
        strategy_counts['Bullish Trend Confirmation'] = {
            'entries': bullish_entries.sum(),
            'exits': bullish_exits.sum()
        }
    
    # Apply EMA trend filter to entries (only buy when price > EMA)
    if USE_MA99_FILTER:
        ema = calculate_ema(price, MA99_PERIOD)
        price_above_ema = price > ema
        original_entries = entries.sum()
        entries = entries & price_above_ema
        filtered_count = original_entries - entries.sum()
        if filtered_count > 0:
            print(f"  EMA{MA99_PERIOD} filter: {filtered_count} buy signals filtered (price below EMA)")
    
    # Print strategy breakdown
    print(f"\n  Strategy Signal Breakdown:")
    for strategy_name, counts in strategy_counts.items():
        print(f"    {strategy_name}: {counts['entries']} buys, {counts['exits']} sells")
    
    # Apply risk management exit conditions (if enabled)
    if ENABLE_RISK_MANAGEMENT:
        print(f"  Risk Management: ENABLED")
        exits = apply_risk_management(entries, exits, price)
    else:
        print(f"  Risk Management: DISABLED")
    
    # Debug: Print combined signal counts
    num_entries = entries.sum()
    num_exits = exits.sum()
    print(f"\n  Combined Signals:")
    print(f"    Total Buy Signals: {num_entries}")
    print(f"    Total Sell Signals: {num_exits}")
    
    if num_entries == 0:
        print(f"  ⚠️  WARNING: No buy signals generated! Check strategy configurations.")
    if num_exits == 0:
        print(f"  ⚠️  WARNING: No sell signals generated! Check strategy configurations.")
    
    # Calculate expected fee impact
    if num_entries > 0:
        estimated_fees = (num_entries + num_exits) * 0.001  # 0.1% per trade
        print(f"  Estimated trading fees: {estimated_fees:.2%} (0.1% per trade)")
    
    return entries, exits


def convert_interval_to_freq(interval):
    """Convert Binance interval format to pandas frequency format."""
    interval_map = {
        '1m': '1min',
        '3m': '3min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1H',
        '2h': '2H',
        '4h': '4H',
        '6h': '6H',
        '8h': '8H',
        '12h': '12H',
        '1d': '1D',
        '3d': '3D',
        '1w': '1W',
        '1M': '1M'
    }
    return interval_map.get(interval, interval)


def backtest_strategy(price, entries, exits, ticker_name):
    """Backtest the strategy using vectorbt."""
    print(f"\nBacktesting {ticker_name}...")
    
    # Convert interval to proper pandas frequency
    freq = convert_interval_to_freq(TIME_INTERVAL)
    
    # Create portfolio with initial balance
    pf = vbt.Portfolio.from_signals(
        price,
        entries=entries,
        exits=exits,
        fees=0.001,  # 0.1% trading fee (typical for Binance)
        freq=freq,
        init_cash=INITIAL_BALANCE  # Starting balance: $1000
    )
    
    return pf


def calculate_annualized_return_manual(total_return, start_date, end_date):
    """Calculate annualized return based on actual time period.
    
    Formula: (1 + total_return)^(365/days) - 1
    
    Note: Annualizing returns from very short periods (< 30 days) can be misleading
    as it assumes the same performance will continue for a full year.
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Calculate actual time period in days
    time_diff = end_date - start_date
    days = time_diff.total_seconds() / (24 * 3600)
    
    if days <= 0:
        return np.nan
    
    # Annualize: (1 + total_return)^(365/days) - 1
    if total_return <= -1:
        return -1.0  # Complete loss
    
    annualized = (1 + total_return) ** (365.0 / days) - 1
    return annualized


def analyze_results(portfolio, ticker_name, start_date=None, end_date=None, entries=None, exits=None):
    """Analyze and display backtest results.
    
    Total Return Explanation:
    - Total Return is the cumulative return from start to end of the backtest period
    - It represents: (Ending Value - Starting Value) / Starting Value
    - Even if individual trades have small returns (0.01%), they compound over multiple trades
    - Example: 8 trades averaging 0.32% each can result in ~2.54% total return
    """
    print(f"\n{'='*60}")
    print(f"Results for {ticker_name}")
    print(f"{'='*60}")
    
    # Signal statistics
    if entries is not None and exits is not None:
        num_entries = entries.sum()
        num_exits = exits.sum()
        print(f"\nSignal Statistics:")
        print(f"  Buy Signals Generated: {num_entries}")
        print(f"  Sell Signals Generated: {num_exits}")
    
    # Basic statistics
    total_return = portfolio.total_return()
    num_trades = portfolio.trades.count()
    
    # Get portfolio value progression
    equity = portfolio.value()
    initial_value = equity.iloc[0] if len(equity) > 0 else 1.0
    final_value = equity.iloc[-1] if len(equity) > 0 else 1.0
    
    print(f"\nPerformance Metrics:")
    print(f"{'='*60}")
    print(f"PORTFOLIO VALUE:")
    print(f"  Starting Value: ${initial_value:,.2f}")
    print(f"  Ending Value: ${final_value:,.2f}")
    print(f"  Total Return: {total_return:.2%} ({(final_value - initial_value) / initial_value * 100:.2f}%)")
    print(f"  Net Profit: ${final_value - initial_value:,.2f}")
    print(f"\nRISK METRICS:")
    print(f"  Sharpe Ratio: {portfolio.sharpe_ratio():.2f}")
    
    # Max Drawdown - vectorbt returns negative value, we'll show as positive for clarity
    max_dd = portfolio.max_drawdown()
    max_dd_pct = abs(max_dd)  # Convert to positive for display
    print(f"  Max Drawdown: {max_dd_pct:.2%} (largest peak-to-trough decline)")
    print(f"  Sortino Ratio: {portfolio.sortino_ratio():.2f}")
    print(f"  Calmar Ratio: {portfolio.calmar_ratio():.2f}")
    print(f"\nTRADE STATISTICS:")
    print(f"  Total Trades: {num_trades}")
    print(f"  Win Rate: {portfolio.trades.win_rate():.2%}")
    
    # Trade analysis
    if num_trades > 0:
        try:
            trades = portfolio.trades.records_readable
            if len(trades) > 0:
                # Check available columns (vectorbt column names may vary)
                available_cols = trades.columns.tolist()
                
                # Try to find return column (could be 'Return [%]', 'Return', 'Pnl', etc.)
                return_col = None
                for col in ['Return [%]', 'Return', 'Pnl [%]', 'Pnl', 'Return %']:
                    if col in available_cols:
                        return_col = col
                        break
                
                # Try to find duration column
                duration_col = None
                for col in ['Duration', 'Exit Timestamp', 'Entry Timestamp']:
                    if col in available_cols:
                        duration_col = col
                        break
                
                print(f"\nINDIVIDUAL TRADE ANALYSIS:")
                print(f"  (Note: Individual trade returns are relative to entry price,")
                print(f"   not portfolio value. Small per-trade returns can compound to larger total returns.)")
                
                if return_col:
                    avg_return = trades[return_col].mean()
                    best_trade = trades[return_col].max()
                    worst_trade = trades[return_col].min()
                    print(f"  Average Return per Trade: {avg_return:.4f}%")
                    print(f"  Best Trade: {best_trade:.4f}%")
                    print(f"  Worst Trade: {worst_trade:.4f}%")
                
                # Show PnL information
                if 'PnL' in available_cols:
                    total_pnl = trades['PnL'].sum()
                    avg_pnl = trades['PnL'].mean()
                    print(f"  Total PnL (all trades): ${total_pnl:.2f}")
                    print(f"  Average PnL per Trade: ${avg_pnl:.2f}")
                
                # Show entry/exit prices if available
                if 'Entry Price' in available_cols and 'Exit Price' in available_cols:
                    print(f"\n  Price Movement:")
                    print(f"    Average Entry Price: ${trades['Entry Price'].mean():.2f}")
                    print(f"    Average Exit Price: ${trades['Exit Price'].mean():.2f}")
                    avg_price_change = ((trades['Exit Price'].mean() - trades['Entry Price'].mean()) / trades['Entry Price'].mean()) * 100
                    print(f"    Average Price Change: {avg_price_change:.2f}%")
                
                if duration_col:
                    avg_duration = trades[duration_col].mean()
                    print(f"  Average Trade Duration: {avg_duration}")
                
                # Explain the compounding effect
                if num_trades > 1:
                    print(f"\n  COMPOUNDING EFFECT:")
                    print(f"    With {num_trades} trades, even small per-trade returns compound.")
                    print(f"    Example: 8 trades of ~0.32% each ≈ 2.54% total (with compounding)")
        except Exception as e:
            print(f"\nTrade Analysis: Could not retrieve detailed trade info ({e})")
    
    # Calculate annualized return manually for accuracy
    if start_date and end_date:
        annualized_return = calculate_annualized_return_manual(total_return, start_date, end_date)
        # Calculate actual period for context
        time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
        days = time_diff.total_seconds() / (24 * 3600)
        
        print(f"\nANNUALIZED METRICS:")
        if days < 30:
            print(f"  Annualized Return: {annualized_return:.2%} (extrapolated from {days:.1f} days - may be misleading)")
            print(f"  ⚠️  Note: Annualizing short periods assumes same performance for full year")
        else:
            print(f"  Annualized Return: {annualized_return:.2%}")
    else:
        # Fallback to portfolio's calculation
        annualized_return = portfolio.annualized_return()
        print(f"\nANNUALIZED METRICS:")
        print(f"  Annualized Return: {annualized_return:.2%}")
    
    # Store max_drawdown as positive value for consistency
    max_dd = abs(portfolio.max_drawdown())
    
    return {
        'total_return': total_return,
        'sharpe_ratio': portfolio.sharpe_ratio(),
        'max_drawdown': max_dd,  # Store as positive value
        'win_rate': portfolio.trades.win_rate(),
        'total_trades': portfolio.trades.count(),
        'annualized_return': annualized_return,
        'calmar_ratio': portfolio.calmar_ratio(),
        'sortino_ratio': portfolio.sortino_ratio()
    }


def plot_account_balance(portfolio, ticker_name):
    """Plot account balance (equity curve) over time."""
    equity = portfolio.value()
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(equity.index, equity.values, linewidth=2, label='Account Balance', color='blue')
    ax.axhline(y=equity.iloc[0], color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Starting Balance')
    ax.set_title(f'{ticker_name} - Account Balance Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key points
    initial_value = equity.iloc[0]
    final_value = equity.iloc[-1]
    max_value = equity.max()
    min_value = equity.min()
    
    # Annotate starting value
    ax.annotate(f'Start: ${initial_value:.2f}', 
                xy=(equity.index[0], initial_value),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3),
                fontsize=9)
    
    # Annotate ending value
    ax.annotate(f'End: ${final_value:.2f}', 
                xy=(equity.index[-1], final_value),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                fontsize=9)
    
    # Calculate and show total return
    total_return = (final_value - initial_value) / initial_value * 100
    ax.text(0.02, 0.98, f'Total Return: {total_return:.2f}%', 
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    return fig


def save_trade_history(portfolio, ticker, strategy_dir):
    """Save trade history to CSV file in strategy-specific folder."""
    try:
        trades = portfolio.trades.records_readable
        
        if len(trades) == 0:
            print(f"  No trades to save for {ticker}")
            return None
        
        # Save to CSV in strategy folder
        csv_path = strategy_dir / f"{ticker}_trade_history.csv"
        trades.to_csv(csv_path, index=False)
        print(f"  Trade history saved: {csv_path} ({len(trades)} trades)")
        
        return csv_path
    except Exception as e:
        print(f"  Warning: Could not save trade history: {e}")
        return None


def save_account_balance(portfolio, ticker, strategy_dir):
    """Save account balance (equity curve) to CSV file in strategy-specific folder."""
    try:
        equity = portfolio.value()
        
        # Create DataFrame with account balance
        balance_df = pd.DataFrame({
            'Date': equity.index,
            'Account_Balance': equity.values,
            'Return_Pct': ((equity.values - equity.iloc[0]) / equity.iloc[0] * 100)
        })
        
        # Save to CSV in strategy folder
        csv_path = strategy_dir / f"{ticker}_account_balance.csv"
        balance_df.to_csv(csv_path, index=False)
        print(f"  Account balance saved: {csv_path} ({len(balance_df)} records)")
        
        return csv_path
    except Exception as e:
        print(f"  Warning: Could not save account balance: {e}")
        return None


def plot_strategy_comparison(strategy_equity_curves, ticker, output_dir):
    """Create comparison plot showing all strategies' equity curves together."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    color_idx = 0
    
    for strategy_name, equity in strategy_equity_curves.items():
        # Calculate total return
        initial = equity.iloc[0]
        final = equity.iloc[-1]
        total_return = (final - initial) / initial * 100
        
        # Plot equity curve
        ax.plot(equity.index, equity.values, 
               linewidth=2, label=f'{strategy_name} ({total_return:.2f}%)',
               color=colors[color_idx % len(colors)], alpha=0.8)
        color_idx += 1
    
    ax.axhline(y=equity.iloc[0], color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Starting Balance')
    ax.set_title(f'{ticker} - Strategy Comparison (Equity Curves)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot in comparison folder
    comparison_dir = output_dir / "strategy_comparison"
    comparison_dir.mkdir(exist_ok=True)
    comparison_path = comparison_dir / f"{ticker}_strategy_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Strategy comparison plot saved: {comparison_path}")


def plot_results(portfolio, df, signals_dict, ticker_name, output_dir):
    """Plot backtest results and indicators.
    
    Args:
        portfolio: VectorBT portfolio object
        df: DataFrame with price data
        signals_dict: Dictionary with signals and indicators
        ticker_name: Ticker name (e.g., 'BTCUSDT')
        output_dir: Directory to save plots (strategy-specific folder)
    """
    print(f"\nGenerating plots for {ticker_name}...")
    
    # Create portfolio plots - use default plots which work across vectorbt versions
    try:
        # Try to plot with default settings
        fig = portfolio.plot(figsize=(16, 12))
    except Exception as e:
        print(f"Warning: Could not create portfolio plot: {e}")
        fig = None
    
    # Plot account balance
    fig_balance = plot_account_balance(portfolio, ticker_name)
    
    # Add MACD plot
    fig2, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # Price with buy/sell signals
    axes[0].plot(df.index, df['close'], label='Close Price', linewidth=1.5)
    
    # Plot EMA if available
    if 'ma99' in signals_dict or 'ema' in signals_dict:
        ema_data = signals_dict.get('ema', signals_dict.get('ma99'))
        axes[0].plot(df.index, ema_data, label=f'EMA{MA99_PERIOD}', 
                    linewidth=1.5, linestyle='--', color='orange', alpha=0.7)
    
    # Get buy/sell signals - check for different possible keys
    buy_signal_key = None
    sell_signal_key = None
    for key in ['strong_bullish', 'bullish_reversal']:
        if key in signals_dict:
            buy_signal_key = key
            break
    for key in ['strong_bearish', 'bearish_reversal']:
        if key in signals_dict:
            sell_signal_key = key
            break
    
    if buy_signal_key:
        buy_signals = df[signals_dict[buy_signal_key]]
        axes[0].scatter(buy_signals.index, buy_signals['close'], 
                        color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    if sell_signal_key:
        sell_signals = df[signals_dict[sell_signal_key]]
        axes[0].scatter(sell_signals.index, sell_signals['close'], 
                        color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    filter_status = "ON" if USE_MA99_FILTER else "OFF"
    axes[0].set_title(f'{ticker_name} - Price with Signals (EMA{MA99_PERIOD} Filter: {filter_status})')
    axes[0].set_ylabel('Price (USDT)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot indicators based on what's available
    if 'macd' in signals_dict and 'signal' in signals_dict:
        # MACD and Signal lines
        axes[1].plot(df.index, signals_dict['macd'], label='MACD', linewidth=1.5)
        axes[1].plot(df.index, signals_dict['signal'], label='Signal', linewidth=1.5)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axes[1].set_title('MACD and Signal Lines')
        axes[1].set_ylabel('MACD')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Histogram - use filled area plot for better visualization with high-frequency data
        if 'histogram' in signals_dict:
            histogram = signals_dict['histogram']
            axes[2].fill_between(df.index, 0, histogram, 
                                 where=(histogram >= 0), 
                                 color='green', alpha=0.3, label='Positive')
            axes[2].fill_between(df.index, 0, histogram, 
                                 where=(histogram < 0), 
                                 color='red', alpha=0.3, label='Negative')
            axes[2].plot(df.index, histogram, color='black', linewidth=1.0, alpha=0.7)
            axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            axes[2].set_title('MACD Histogram')
            axes[2].set_ylabel('Histogram')
            axes[2].set_xlabel('Date')
            axes[2].legend(loc='upper left')
            axes[2].grid(True, alpha=0.3)
    elif 'rsi' in signals_dict:
        # RSI indicator
        rsi = signals_dict['rsi']
        axes[1].plot(df.index, rsi, label='RSI', linewidth=1.5, color='purple')
        axes[1].axhline(y=RSI_OVERBOUGHT, color='red', linestyle='--', linewidth=0.8, label=f'Overbought ({RSI_OVERBOUGHT})')
        axes[1].axhline(y=RSI_OVERSOLD, color='green', linestyle='--', linewidth=0.8, label=f'Oversold ({RSI_OVERSOLD})')
        axes[1].axhline(y=50, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        axes[1].set_title('RSI Indicator')
        axes[1].set_ylabel('RSI')
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel('Date')
        # Hide third subplot for RSI
        axes[2].axis('off')
    elif 'ema_fast' in signals_dict and 'ema_slow' in signals_dict:
        # EMA Crossover indicators
        axes[1].plot(df.index, signals_dict['ema_fast'], label=f'EMA{BULLISH_EMA_FAST}', linewidth=1.5, color='blue')
        axes[1].plot(df.index, signals_dict['ema_slow'], label=f'EMA{BULLISH_EMA_SLOW}', linewidth=1.5, color='orange')
        axes[1].set_title('EMA Crossover')
        axes[1].set_ylabel('Price (USDT)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel('Date')
        # Hide third subplot for EMA crossover
        axes[2].axis('off')
    else:
        # No specific indicators, just show price
        axes[1].axis('off')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save plots to strategy-specific folder (passed as output_dir)
    if fig is not None:
        fig.savefig(output_dir / f"{ticker_name}_portfolio.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Save account balance plot
    fig_balance.savefig(output_dir / f"{ticker_name}_account_balance.png", dpi=150, bbox_inches='tight')
    plt.close(fig_balance)
    
    fig2.savefig(output_dir / f"{ticker_name}_signals.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Plots saved to {output_dir}/")
    
    return fig, fig2, fig_balance


def main():
    """Main execution function."""
    print("="*60)
    print("Multi-Strategy Trading System using VectorBT")
    print("="*60)
    
    # Print enabled strategies
    enabled_strategies = []
    if ENABLE_MACD_TREND_REVERSAL:
        enabled_strategies.append("1. MACD Trend Reversals")
    if ENABLE_RSI_TREND_REVERSAL:
        enabled_strategies.append("2. RSI Trend Reversals")
    if ENABLE_BULLISH_CONFIRMATION:
        enabled_strategies.append("3. Bullish Trend Confirmation")
    
    print(f"\nEnabled Strategies:")
    for strategy in enabled_strategies:
        print(f"  ✓ {strategy}")
    if not enabled_strategies:
        print("  ⚠️  No strategies enabled! Please enable at least one strategy.")
        return
    
    # Fetch data
    df = fetch_binance_data(TICKER_LIST, START_DATE, END_DATE, TIME_INTERVAL)
    
    if df.empty:
        print("Error: No data fetched. Please check your ticker list and date range.")
        return
    
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Process each ticker
    all_strategy_results = {}  # Store results for each strategy separately
    
    for ticker in TICKER_LIST:
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")
        print(f"{'='*60}")
        
        ticker_df = df[df['tic'] == ticker].copy()
        
        if ticker_df.empty:
            print(f"\nWarning: No data for {ticker}, skipping...")
            continue
        
        # Set time as index
        ticker_df['time'] = pd.to_datetime(ticker_df['time'])
        ticker_df = ticker_df.set_index('time').sort_index()
        
        price = ticker_df['close']
        base_output_dir = Path("results")
        base_output_dir.mkdir(exist_ok=True)
        
        # Store portfolios and equity curves for comparison
        strategy_portfolios = {}
        strategy_equity_curves = {}
        
        # Strategy 1: MACD Trend Reversals
        if ENABLE_MACD_TREND_REVERSAL:
            strategy_name = "MACD_Trend_Reversal"
            print(f"\n[Strategy 1] {strategy_name}")
            
            # Create strategy-specific folder
            strategy_dir = base_output_dir / strategy_name
            strategy_dir.mkdir(exist_ok=True)
            
            macd, signal, histogram = calculate_macd(
                ticker_df, 
                fast=MACD_FAST, 
                slow=MACD_SLOW, 
                signal=MACD_SIGNAL
            )
            macd_signals = identify_trend_reversals(price, macd, signal, histogram)
            
            # Create signals for this strategy only
            entries = macd_signals['strong_bullish'].fillna(False)
            exits = macd_signals['strong_bearish'].fillna(False).copy()
            
            # Apply EMA filter if enabled
            if USE_MA99_FILTER:
                ema = calculate_ema(price, MA99_PERIOD)
                entries = entries & (price > ema)
            
            # Apply risk management
            if ENABLE_RISK_MANAGEMENT:
                exits = apply_risk_management(entries, exits, price)
            
            # Backtest this strategy
            portfolio = backtest_strategy(price, entries, exits, ticker)
            strategy_portfolios[strategy_name] = portfolio
            strategy_equity_curves[strategy_name] = portfolio.value()
            
            # Analyze and save results
            actual_start = ticker_df.index.min()
            actual_end = ticker_df.index.max()
            results = analyze_results(portfolio, f"{ticker}_{strategy_name}", actual_start, actual_end, entries, exits)
            all_strategy_results[f"{ticker}_{strategy_name}"] = results
            
            # Save CSV files for this strategy
            save_trade_history(portfolio, ticker, strategy_dir)
            save_account_balance(portfolio, ticker, strategy_dir)
            
            # Plot individual strategy
            signals_dict = macd_signals.copy()
            signals_dict['macd'] = macd
            signals_dict['signal'] = signal
            signals_dict['histogram'] = histogram
            plot_results(portfolio, ticker_df, signals_dict, ticker, strategy_dir)
        
        # Strategy 2: RSI Trend Reversals
        if ENABLE_RSI_TREND_REVERSAL:
            strategy_name = "RSI_Trend_Reversal"
            print(f"\n[Strategy 2] {strategy_name}")
            
            # Create strategy-specific folder
            strategy_dir = base_output_dir / strategy_name
            strategy_dir.mkdir(exist_ok=True)
            
            rsi = calculate_rsi(price, RSI_PERIOD)
            rsi_signals = identify_rsi_trend_reversals(price, rsi)
            
            # Create signals for this strategy only
            entries = rsi_signals['bullish_reversal'].fillna(False)
            exits = rsi_signals['bearish_reversal'].fillna(False).copy()
            
            # Apply EMA filter if enabled
            if USE_MA99_FILTER:
                ema = calculate_ema(price, MA99_PERIOD)
                entries = entries & (price > ema)
            
            # Apply risk management
            if ENABLE_RISK_MANAGEMENT:
                exits = apply_risk_management(entries, exits, price)
            
            # Backtest this strategy
            portfolio = backtest_strategy(price, entries, exits, ticker)
            strategy_portfolios[strategy_name] = portfolio
            strategy_equity_curves[strategy_name] = portfolio.value()
            
            # Analyze and save results
            actual_start = ticker_df.index.min()
            actual_end = ticker_df.index.max()
            results = analyze_results(portfolio, f"{ticker}_{strategy_name}", actual_start, actual_end, entries, exits)
            all_strategy_results[f"{ticker}_{strategy_name}"] = results
            
            # Save CSV files for this strategy
            save_trade_history(portfolio, ticker, strategy_dir)
            save_account_balance(portfolio, ticker, strategy_dir)
            
            # Plot individual strategy
            signals_dict = {
                'rsi': rsi,
                'bullish_reversal': rsi_signals['bullish_reversal'],
                'bearish_reversal': rsi_signals['bearish_reversal']
            }
            plot_results(portfolio, ticker_df, signals_dict, ticker, strategy_dir)
        
        # Strategy 3: Bullish Trend Confirmation
        if ENABLE_BULLISH_CONFIRMATION:
            strategy_name = "Bullish_Trend_Confirmation"
            print(f"\n[Strategy 3] {strategy_name}")
            
            # Create strategy-specific folder
            strategy_dir = base_output_dir / strategy_name
            strategy_dir.mkdir(exist_ok=True)
            
            bullish_signals = identify_bullish_trend_confirmation(price)
            
            # Create signals for this strategy only
            entries = bullish_signals['bullish_reversal'].fillna(False)
            exits = bullish_signals['bearish_reversal'].fillna(False).copy()
            
            # Apply EMA filter if enabled
            if USE_MA99_FILTER:
                ema = calculate_ema(price, MA99_PERIOD)
                entries = entries & (price > ema)
            
            # Apply risk management
            if ENABLE_RISK_MANAGEMENT:
                exits = apply_risk_management(entries, exits, price)
            
            # Backtest this strategy
            portfolio = backtest_strategy(price, entries, exits, ticker)
            strategy_portfolios[strategy_name] = portfolio
            strategy_equity_curves[strategy_name] = portfolio.value()
            
            # Analyze and save results
            actual_start = ticker_df.index.min()
            actual_end = ticker_df.index.max()
            results = analyze_results(portfolio, f"{ticker}_{strategy_name}", actual_start, actual_end, entries, exits)
            all_strategy_results[f"{ticker}_{strategy_name}"] = results
            
            # Save CSV files for this strategy
            save_trade_history(portfolio, ticker, strategy_dir)
            save_account_balance(portfolio, ticker, strategy_dir)
            
            # Plot individual strategy
            signals_dict = {
                'ema_fast': bullish_signals.get('ema_fast'),
                'ema_slow': bullish_signals.get('ema_slow'),
                'bullish_reversal': bullish_signals['bullish_reversal'],
                'bearish_reversal': bullish_signals['bearish_reversal']
            }
            plot_results(portfolio, ticker_df, signals_dict, ticker, strategy_dir)
        
        # Create comparison plot for all strategies
        if len(strategy_equity_curves) > 1:
            plot_strategy_comparison(strategy_equity_curves, ticker, base_output_dir)
        
        # Also create combined strategy (if multiple enabled)
        if len([s for s in [ENABLE_MACD_TREND_REVERSAL, ENABLE_RSI_TREND_REVERSAL, ENABLE_BULLISH_CONFIRMATION] if s]) > 1:
            print(f"\n[Combined Strategy] Running all strategies together...")
            
            # Create combined strategy folder
            combined_strategy_dir = base_output_dir / "Combined"
            combined_strategy_dir.mkdir(exist_ok=True)
            
            all_strategies = {}
            signals_dict = {}
            
            if ENABLE_MACD_TREND_REVERSAL:
                macd, signal, histogram = calculate_macd(ticker_df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
                macd_signals = identify_trend_reversals(price, macd, signal, histogram)
                all_strategies['macd'] = macd_signals
                signals_dict.update(macd_signals)
            
            if ENABLE_RSI_TREND_REVERSAL:
                rsi = calculate_rsi(price, RSI_PERIOD)
                rsi_signals = identify_rsi_trend_reversals(price, rsi)
                all_strategies['rsi'] = rsi_signals
                signals_dict['rsi'] = rsi
            
            if ENABLE_BULLISH_CONFIRMATION:
                bullish_signals = identify_bullish_trend_confirmation(price)
                all_strategies['bullish_confirmation'] = bullish_signals
                signals_dict['ema_fast'] = bullish_signals.get('ema_fast')
                signals_dict['ema_slow'] = bullish_signals.get('ema_slow')
            
            entries, exits = create_vectorbt_signals(ticker_df, all_strategies, price)
            portfolio = backtest_strategy(price, entries, exits, ticker)
            
            actual_start = ticker_df.index.min()
            actual_end = ticker_df.index.max()
            results = analyze_results(portfolio, f"{ticker}_Combined", actual_start, actual_end, entries, exits)
            all_strategy_results[f"{ticker}_Combined"] = results
            
            save_trade_history(portfolio, ticker, combined_strategy_dir)
            save_account_balance(portfolio, ticker, combined_strategy_dir)
            
            strategy_equity_curves['Combined'] = portfolio.value()
            plot_results(portfolio, ticker_df, signals_dict, ticker, combined_strategy_dir)
            plot_strategy_comparison(strategy_equity_curves, ticker, base_output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - All Strategies")
    print(f"{'='*60}")
    
    if all_strategy_results:
        summary_df = pd.DataFrame(all_strategy_results).T
        print("\n", summary_df.to_string())
        
        # Save summary
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        summary_df.to_csv(output_dir / "strategy_summary.csv")
        print(f"\nSummary saved to {output_dir}/strategy_summary.csv")
        print(f"\nResults organized by strategy in:")
        print(f"  {output_dir}/MACD_Trend_Reversal/")
        print(f"  {output_dir}/RSI_Trend_Reversal/")
        print(f"  {output_dir}/Bullish_Trend_Confirmation/")
        print(f"  {output_dir}/Combined/")
        print(f"  {output_dir}/strategy_comparison/")
    else:
        print("No strategy results to summarize.")


if __name__ == "__main__":
    main()

"""
XAUUSDT RL Agent Backtesting on Binance Futures
This script fetches XAUUSDT data from Binance Futures and runs RL agent backtesting.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import warnings
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings('ignore')

# Add FinRL-Meta to sys.path
finrl_meta_path = Path("/mnt/data/FinRL-Tutorials/3-Practical/FinRL-Meta")
if str(finrl_meta_path) not in sys.path:
    sys.path.insert(0, str(finrl_meta_path))

# Import backtest functions
from backtest import (
    backtest_strategy,
    analyze_results,
    save_account_balance,
    save_trade_history,
    calculate_macd,
    identify_trend_reversals,
    calculate_rsi,
    identify_rsi_trend_reversals,
    INITIAL_BALANCE,
    TIME_INTERVAL,
    ENABLE_MACD_TREND_REVERSAL,
    ENABLE_RSI_TREND_REVERSAL,
)

# Import RL risk management
from rl_risk_management import apply_rl_risk_management, RLRiskManager

# Configure logging
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest_xauusdt_rl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
TICKER = "XRPUSDT"
RESULTS_DIR = Path("results")
XAUUSDT_RESULTS_DIR = RESULTS_DIR / "XAUUSDT_RL"
XAUUSDT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_LOAD_DIR = Path("models/rl_agent")
DEFAULT_MODEL_NAME = "best_model"

# Date range for backtesting
START_DATE = "2025-12-06"
END_DATE = datetime.now().strftime("%Y-%m-%d")
TIME_INTERVAL = "15m"  # 15 minutes

# Binance Futures API endpoint
BINANCE_FUTURES_API = "https://fapi.binance.com"


def fetch_binance_futures_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    limit: int = 1500
) -> pd.DataFrame:
    """
    Fetch historical klines data from Binance Futures.
    Paginates by startTime only (no endTime in request) so full date ranges work;
    Binance can return only the most recent candles when both startTime and endTime are sent.
    
    Args:
        symbol: Trading pair (e.g., 'XAUUSDT')
        interval: Time interval (e.g., '15m', '1h', '1d')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        limit: Maximum number of klines per request (max 1500)
    
    Returns:
        DataFrame with OHLCV data in FinRL format
    """
    logger.info(f"Fetching {symbol} data from Binance Futures...")
    logger.info(f"  Interval: {interval}")
    logger.info(f"  Date range: {start_date} to {end_date}")
    
    # Convert dates to UTC timestamps (Binance uses UTC)
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)
    # End of day for end_date so we include that day's candles
    end_dt = end_dt + timedelta(days=1)
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    all_klines = []
    current_start = start_ts
    
    # Paginate using startTime only (no endTime); stop when we reach or pass end_ts
    while current_start < end_ts:
        try:
            url = f"{BINANCE_FUTURES_API}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
            
            last_timestamp = klines[-1][0]
            # Only include candles before end_ts
            for k in klines:
                if k[0] < end_ts:
                    all_klines.append(k)
                else:
                    break
            
            current_start = last_timestamp + 1
            if last_timestamp >= end_ts or len(klines) < limit:
                break
                
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break
    
    if not all_klines:
        raise ValueError(f"No data fetched for {symbol}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert to proper types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['time'] = df['timestamp']  # FinRL format uses 'time'
    df['tic'] = symbol  # Add ticker column
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    df = df.set_index('timestamp')
    
    # Select required columns for FinRL format
    df = df[['time', 'tic', 'open', 'high', 'low', 'close', 'volume']]
    
    logger.info(f"Fetched {len(df)} records")
    logger.info(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    
    return df


def get_strategy_signals(ticker_df: pd.DataFrame, price: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Get entry and exit signals for the combined strategy.
    
    Args:
        ticker_df: DataFrame with OHLCV data
        price: Price series
    
    Returns:
        entries, exits (boolean series)
    """
    logger.info("Generating strategy signals...")
    
    # Initialize signals
    entries = pd.Series(False, index=price.index)
    exits = pd.Series(False, index=price.index)
    
    # MACD Trend Reversal signals
    if ENABLE_MACD_TREND_REVERSAL:
        logger.info("  Calculating MACD signals...")
        macd, signal, histogram = calculate_macd(ticker_df)
        macd_signals = identify_trend_reversals(price, macd, signal, histogram)
        entries = entries | macd_signals.get('strong_bullish', pd.Series(False, index=price.index))
        exits = exits | macd_signals.get('strong_bearish', pd.Series(False, index=price.index))
        logger.info(f"    MACD entries: {entries.sum()}, exits: {exits.sum()}")
    
    # RSI Trend Reversal signals
    if ENABLE_RSI_TREND_REVERSAL:
        logger.info("  Calculating RSI signals...")
        rsi = calculate_rsi(price)
        rsi_signals = identify_rsi_trend_reversals(price, rsi)
        entries = entries | rsi_signals.get('bullish_reversal', pd.Series(False, index=price.index))
        exits = exits | rsi_signals.get('bearish_reversal', pd.Series(False, index=price.index))
        logger.info(f"    RSI entries: {entries.sum()}, exits: {exits.sum()}")
    
    entries = entries.fillna(False)
    exits = exits.fillna(False)
    
    logger.info(f"  Total entries: {entries.sum()}, Total exits: {exits.sum()}")
    
    return entries, exits


def evaluate_rl_agent(
    ticker_df: pd.DataFrame,
    price: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    model_path: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME
) -> Dict:
    """
    Evaluate RL agent on XAUUSDT data.
    
    Args:
        ticker_df: DataFrame with OHLCV data
        price: Price series
        entries: Entry signals
        exits: Original exit signals
        model_path: Path to model directory
        model_name: Model filename
    
    Returns:
        Results dictionary
    """
    logger.info("="*60)
    logger.info("Evaluating RL Agent on XAUUSDT")
    logger.info("="*60)
    
    # Load RL risk manager
    try:
        logger.info(f"Loading RL model: {model_name}")
        manager = RLRiskManager(
            model_path=model_path,
            model_name=model_name,
            initial_balance=INITIAL_BALANCE
        )
        logger.info("RL model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading RL model: {e}")
        logger.error("Cannot proceed without RL model")
        raise
    
    # Apply RL risk management (pass OHLCV so env can compute volume-based indicators: OBV, MFI, AD, PVT)
    logger.info("Applying RL risk management...")
    balance = pd.Series(INITIAL_BALANCE, index=price.index)
    ohlcv_df = ticker_df[['open', 'high', 'low', 'close', 'volume']].copy() if all(c in ticker_df.columns for c in ['open', 'high', 'low', 'close', 'volume']) else None
    exits_rl = manager.apply_rl_risk_management(entries, exits.copy(), price, balance, ohlcv_df=ohlcv_df)
    
    logger.info(f"  Original exits: {exits.sum()}")
    logger.info(f"  RL-managed exits: {exits_rl.sum()}")
    logger.info(f"  Additional exits from RL: {exits_rl.sum() - exits.sum()}")
    
    # Backtest
    logger.info("Running backtest...")
    portfolio_rl = backtest_strategy(price, entries, exits_rl, TICKER)
    
    # Analyze results
    logger.info("Analyzing results...")
    results = analyze_results(
        portfolio_rl,
        f"{TICKER}_RL_Agent",
        None,
        None,
        entries,
        exits_rl
    )
    
    # Add portfolio for later use
    results['portfolio'] = portfolio_rl
    results['entries'] = entries
    results['exits'] = exits_rl
    results['method'] = 'rl_agent'
    
    return results


def save_results(results: Dict, ticker_df: pd.DataFrame, price: pd.Series):
    """
    Save backtest results to files.
    
    Args:
        results: Results dictionary from analyze_results
        ticker_df: DataFrame with OHLCV data
        price: Price series
    """
    logger.info("Saving results...")
    
    portfolio = results.get('portfolio')
    entries = results.get('entries')
    exits = results.get('exits')
    
    if portfolio is None:
        logger.warning("No portfolio to save")
        return
    
    # Save account balance
    save_account_balance(portfolio, TICKER, XAUUSDT_RESULTS_DIR)
    
    # Save trade history
    save_trade_history(portfolio, TICKER, XAUUSDT_RESULTS_DIR)
    
    # Create signals dictionary for plotting
    signals_dict = {
        'entries': entries,
        'exits': exits
    }
    
    # Plot results
    try:
        from backtest import plot_results
        fig, fig2, fig_balance = plot_results(
            portfolio, ticker_df, signals_dict, TICKER, XAUUSDT_RESULTS_DIR
        )
        if fig:
            plt.close(fig)
        if fig2:
            plt.close(fig2)
        if fig_balance:
            plt.close(fig_balance)
    except Exception as e:
        logger.warning(f"Could not plot results: {e}")
    
    logger.info(f"Results saved to: {XAUUSDT_RESULTS_DIR}")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("XAUUSDT RL Agent Backtesting on Binance Futures")
    logger.info("="*60)
    
    try:
        # Fetch data from Binance Futures
        logger.info(f"\nFetching {TICKER} data from Binance Futures...")
        df = fetch_binance_futures_data(
            symbol=TICKER,
            interval=TIME_INTERVAL,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        if df.empty:
            logger.error("No data fetched. Exiting.")
            return
        
        # Extract ticker data
        ticker_df = df[df['tic'] == TICKER].copy()
        if ticker_df.empty:
            logger.error(f"No data for {TICKER}")
            return
        
        # Set time as index if not already
        if 'time' in ticker_df.columns:
            ticker_df['time'] = pd.to_datetime(ticker_df['time'])
            ticker_df = ticker_df.set_index('time').sort_index()
        
        # Get price series
        price = ticker_df['close']
        
        logger.info(f"\nData loaded:")
        logger.info(f"  Records: {len(ticker_df)}")
        logger.info(f"  Date range: {ticker_df.index.min()} to {ticker_df.index.max()}")
        logger.info(f"  Price range: ${price.min():.2f} - ${price.max():.2f}")
        
        # Generate strategy signals
        logger.info(f"\nGenerating strategy signals...")
        entries, exits = get_strategy_signals(ticker_df, price)
        
        if entries.sum() == 0:
            logger.warning("No entry signals generated. Exiting.")
            return
        
        # Evaluate RL agent
        logger.info(f"\nEvaluating RL agent...")
        results = evaluate_rl_agent(
            ticker_df=ticker_df,
            price=price,
            entries=entries,
            exits=exits,
            model_path=MODEL_LOAD_DIR,
            model_name=DEFAULT_MODEL_NAME
        )
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("Backtest Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total Return: {results.get('total_return', 0):.2%}")
        logger.info(f"Annualized Return: {results.get('annualized_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        logger.info(f"Win Rate: {results.get('win_rate', 0):.2%}")
        logger.info(f"Total Trades: {results.get('total_trades', 0)}")
        logger.info(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        
        # Save results
        logger.info(f"\nSaving results...")
        save_results(results, ticker_df, price)
        
        logger.info(f"\n{'='*60}")
        logger.info("Backtesting complete!")
        logger.info(f"Results saved to: {XAUUSDT_RESULTS_DIR}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

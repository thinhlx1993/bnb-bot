"""
Prepare RL Training Data from Backtest Results
Extract training episodes from existing trade history and account balance data.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prepare_rl_training_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = Path("results")
TRAINING_DATA_DIR = Path("rl_training_episodes")
TRAINING_DATA_DIR.mkdir(exist_ok=True)

# Filter settings
STRATEGY_FILTER = None  # None = all strategies, or specific strategy name
TICKER_FILTER = None    # None = all tickers, or specific ticker name
MIN_EPISODE_LENGTH = 2  # Minimum steps per episode
MAX_EPISODE_LENGTH = 1000  # Maximum steps per episode


def load_trade_history(results_dir: Path, strategy_filter: Optional[str] = None, 
                       ticker_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Load all trade history CSV files from results directory.
    
    Args:
        results_dir: Directory containing strategy results
        strategy_filter: Optional filter for specific strategy
        ticker_filter: Optional filter for specific ticker
    
    Returns:
        DataFrame with all trades
    """
    logger.info(f"Loading trade history from {results_dir}...")
    
    all_trades = []
    
    # Find all strategy directories
    strategy_dirs = [d for d in results_dir.iterdir() if d.is_dir() 
                     and d.name not in ['training_data', 'strategy_comparison']]
    
    for strategy_dir in strategy_dirs:
        strategy_name = strategy_dir.name
        
        # Apply strategy filter
        if strategy_filter and strategy_filter not in strategy_name:
            continue
        
        # Find trade history CSV files
        trade_files = list(strategy_dir.glob("*_trade_history.csv"))
        
        for trade_file in trade_files:
            # Extract ticker from filename
            filename = trade_file.stem
            ticker = filename.replace("_trade_history", "")
            
            # Apply ticker filter
            if ticker_filter and ticker_filter not in ticker:
                continue
            
            try:
                trades_df = pd.read_csv(trade_file)
                
                if len(trades_df) == 0:
                    logger.warning(f"  Empty file: {trade_file}, skipping...")
                    continue
                
                # Add metadata columns
                trades_df['strategy'] = strategy_name
                trades_df['ticker'] = ticker
                trades_df['source_file'] = str(trade_file)
                
                all_trades.append(trades_df)
                logger.info(f"  Loaded {trade_file.name}: {len(trades_df)} trades")
                
            except Exception as e:
                logger.error(f"  Error loading {trade_file}: {e}")
                continue
    
    if len(all_trades) == 0:
        logger.error("No trade history files found!")
        return pd.DataFrame()
    
    # Combine all trades
    combined_trades = pd.concat(all_trades, ignore_index=True)
    logger.info(f"\nTotal trades loaded: {len(combined_trades)}")
    
    return combined_trades


def load_account_balance(results_dir: Path, strategy: str, ticker: str) -> Optional[pd.Series]:
    """
    Load account balance data for a specific strategy and ticker.
    
    Args:
        results_dir: Results directory
        strategy: Strategy name
        ticker: Ticker symbol
    
    Returns:
        Account balance series with datetime index, or None if not found
    """
    balance_file = results_dir / strategy / f"{ticker}_account_balance.csv"
    
    if not balance_file.exists():
        logger.warning(f"  Account balance file not found: {balance_file}")
        return None
    
    try:
        balance_df = pd.read_csv(balance_file)
        balance_df['Date'] = pd.to_datetime(balance_df['Date'])
        balance_series = pd.Series(
            balance_df['Account_Balance'].values,
            index=balance_df['Date']
        )
        return balance_series
    except Exception as e:
        logger.error(f"  Error loading balance file {balance_file}: {e}")
        return None


def load_price_data_from_backtest(ticker: str, time_interval: str = "15m") -> Optional[pd.DataFrame]:
    """
    Load price data by re-running backtest data fetching or using cached data.
    
    Args:
        ticker: Ticker symbol
        time_interval: Time interval
    
    Returns:
        DataFrame with OHLCV data, or None if not available
    """
    # Try to load from cached data directory if available
    data_dir = Path("data")
    if data_dir.exists():
        cache_file = data_dir / f"{ticker}_data.csv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time')
                    return df
            except Exception as e:
                logger.debug(f"Could not load cached data: {e}")
    
    # Try to load from the main dataset.csv file
    dataset_file = data_dir / "dataset.csv"
    if dataset_file.exists():
        try:
            df = pd.read_csv(dataset_file)
            if 'tic' in df.columns and 'time' in df.columns:
                ticker_df = df[df['tic'] == ticker].copy()
                if len(ticker_df) > 0:
                    ticker_df['time'] = pd.to_datetime(ticker_df['time'])
                    ticker_df = ticker_df.set_index('time').sort_index()
                    return ticker_df
        except Exception as e:
            logger.debug(f"Could not load from dataset: {e}")
    
    return None


def extract_episodes_from_trades(
    trades_df: pd.DataFrame,
    results_dir: Path,
    min_length: int = MIN_EPISODE_LENGTH,
    max_length: int = MAX_EPISODE_LENGTH
) -> List[Dict]:
    """
    Extract training episodes from trade history.
    
    Args:
        trades_df: DataFrame with trade history
        results_dir: Results directory
        min_length: Minimum episode length
        max_length: Maximum episode length
    
    Returns:
        List of episode dictionaries
    """
    logger.info("Extracting episodes from trades...")
    
    episodes = []
    
    # Group by strategy and ticker
    grouped = trades_df.groupby(['strategy', 'ticker'])
    
    for (strategy, ticker), group in grouped:
        logger.info(f"Processing {ticker} - {strategy} ({len(group)} trades)...")
        
        # Load account balance data
        balance_series = load_account_balance(results_dir, strategy, ticker)
        
        if balance_series is None:
            logger.warning(f"  Skipping {ticker} - {strategy}: No balance data")
            continue
        
        # Process each trade
        for idx, trade in group.iterrows():
            try:
                # Parse entry and exit timestamps
                entry_time = pd.to_datetime(trade['Entry Timestamp'])
                exit_time = pd.to_datetime(trade['Exit Timestamp'])
                
                # Get entry and exit prices
                entry_price = float(trade['Avg Entry Price'])
                exit_price = float(trade['Avg Exit Price'])
                
                # Get return (ground truth)
                trade_return = float(trade['Return'])
                trade_pnl = float(trade['PnL'])
                
                # Find indices in balance series
                try:
                    entry_idx = balance_series.index.get_loc(entry_time)
                    exit_idx = balance_series.index.get_loc(exit_time)
                except (KeyError, TypeError):
                    # Try nearest timestamp
                    entry_idx = balance_series.index.get_indexer([entry_time], method='nearest')[0]
                    exit_idx = balance_series.index.get_indexer([exit_time], method='nearest')[0]
                
                # Ensure we have valid indices
                if entry_idx < 0 or exit_idx < 0 or entry_idx >= exit_idx:
                    logger.warning(f"  Invalid indices for trade {idx}: entry={entry_idx}, exit={exit_idx}")
                    continue
                
                episode_length = exit_idx - entry_idx + 1
                
                # Filter by episode length
                if episode_length < min_length or episode_length > max_length:
                    logger.debug(f"  Skipping trade {idx}: length {episode_length} out of range")
                    continue
                
                # Load actual price data
                ticker_price_data = load_price_data_from_backtest(ticker)
                
                if ticker_price_data is not None and 'close' in ticker_price_data.columns:
                    # Use actual price data
                    price_index = balance_series.index[entry_idx:exit_idx + 1]
                    try:
                        # Align price data with balance timestamps
                        price_series = ticker_price_data['close'].reindex(price_index, method='nearest')
                    except:
                        # Fallback: extract by index range
                        price_start_idx = ticker_price_data.index.get_indexer([entry_time], method='nearest')[0]
                        price_end_idx = ticker_price_data.index.get_indexer([exit_time], method='nearest')[0]
                        if price_start_idx >= 0 and price_end_idx >= price_start_idx:
                            price_window = ticker_price_data.iloc[price_start_idx:price_end_idx + 1]['close']
                            # Reindex to match balance timestamps
                            price_series = price_window.reindex(price_index, method='nearest')
                        else:
                            price_series = None
                else:
                    # Fallback: create synthetic price series
                    logger.debug(f"  Using synthetic price data for {ticker} trade {idx}")
                    num_periods = exit_idx - entry_idx + 1
                    price_returns = np.linspace(0, trade_return, num_periods)
                    price_values = entry_price * (1 + price_returns)
                    price_index = balance_series.index[entry_idx:exit_idx + 1]
                    price_series = pd.Series(price_values, index=price_index)
                
                if price_series is None or len(price_series) == 0:
                    logger.warning(f"  Could not get price data for trade {idx}, skipping...")
                    continue
                
                # Extract balance window
                balance_window = balance_series.iloc[entry_idx:exit_idx + 1]
                
                # Get initial balance (at entry)
                initial_balance = float(balance_series.iloc[entry_idx])
                
                # Create episode dictionary
                episode = {
                    'strategy': strategy,
                    'ticker': ticker,
                    'trade_id': int(trade.get('Position Id', idx)),
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_idx': int(entry_idx),
                    'exit_idx': int(exit_idx),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'initial_balance': initial_balance,
                    'episode_length': episode_length,
                    'trade_return': trade_return,
                    'trade_pnl': trade_pnl,
                    'price_series': price_series,
                    'balance_series': balance_window,
                    'price_data_full': balance_series.index,  # Store index for reference
                    'balance_data_full': balance_series  # Store full balance for context
                }
                
                episodes.append(episode)
                
            except Exception as e:
                logger.warning(f"  Error processing trade {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    logger.info(f"\nExtracted {len(episodes)} episodes")
    return episodes


def save_episodes(episodes: List[Dict], output_dir: Path):
    """
    Save episodes to disk.
    
    Args:
        episodes: List of episode dictionaries
        output_dir: Output directory
    """
    logger.info(f"Saving episodes to {output_dir}...")
    
    # Save all episodes as pickle
    episodes_file = output_dir / "episodes.pkl"
    with open(episodes_file, 'wb') as f:
        pickle.dump(episodes, f)
    logger.info(f"  Saved {len(episodes)} episodes to {episodes_file}")
    
    # Save metadata as CSV
    metadata = []
    for ep in episodes:
        metadata.append({
            'strategy': ep['strategy'],
            'ticker': ep['ticker'],
            'trade_id': ep['trade_id'],
            'entry_time': ep['entry_time'],
            'exit_time': ep['exit_time'],
            'entry_price': ep['entry_price'],
            'exit_price': ep['exit_price'],
            'initial_balance': ep['initial_balance'],
            'episode_length': ep['episode_length'],
            'trade_return': ep['trade_return'],
            'trade_pnl': ep['trade_pnl']
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_file = output_dir / "episodes_metadata.csv"
    metadata_df.to_csv(metadata_file, index=False)
    logger.info(f"  Saved metadata to {metadata_file}")
    
    # Print statistics
    logger.info(f"\nEpisode Statistics:")
    logger.info(f"  Total episodes: {len(episodes)}")
    logger.info(f"  Average length: {metadata_df['episode_length'].mean():.1f} periods")
    logger.info(f"  Min length: {metadata_df['episode_length'].min()} periods")
    logger.info(f"  Max length: {metadata_df['episode_length'].max()} periods")
    logger.info(f"  Average return: {metadata_df['trade_return'].mean():.4f} ({metadata_df['trade_return'].mean()*100:.2f}%)")
    logger.info(f"  Profitable trades: {(metadata_df['trade_return'] > 0).sum()} ({(metadata_df['trade_return'] > 0).mean()*100:.1f}%)")
    
    # Group by strategy
    if 'strategy' in metadata_df.columns:
        logger.info(f"\nEpisodes by Strategy:")
        for strategy, group in metadata_df.groupby('strategy'):
            logger.info(f"  {strategy}: {len(group)} episodes")
    
    # Group by ticker
    if 'ticker' in metadata_df.columns:
        logger.info(f"\nEpisodes by Ticker:")
        for ticker, group in metadata_df.groupby('ticker'):
            logger.info(f"  {ticker}: {len(group)} episodes")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("RL Training Data Preparation")
    logger.info("="*60)
    
    # Load trade history
    trades_df = load_trade_history(RESULTS_DIR, STRATEGY_FILTER, TICKER_FILTER)
    
    if len(trades_df) == 0:
        logger.error("No trades found! Please run backtest first.")
        return
    
    # Extract episodes
    episodes = extract_episodes_from_trades(
        trades_df,
        RESULTS_DIR,
        min_length=MIN_EPISODE_LENGTH,
        max_length=MAX_EPISODE_LENGTH
    )
    
    if len(episodes) == 0:
        logger.error("No valid episodes extracted!")
        return
    
    # Save episodes
    save_episodes(episodes, TRAINING_DATA_DIR)
    
    logger.info("="*60)
    logger.info("Data preparation complete!")
    logger.info(f"Episodes saved to: {TRAINING_DATA_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

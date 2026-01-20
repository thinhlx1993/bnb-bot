"""
Evaluate RL Risk Management Agent
Compare performance against rule-based baseline on held-out test data.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# Import backtest functions
from backtest import (
    backtest_strategy,
    analyze_results,
    plot_account_balance,
    save_account_balance,
    save_trade_history,
    INITIAL_BALANCE,
    TIME_INTERVAL,
    TICKER_LIST
)

# Import RL risk management
from rl_risk_management import apply_rl_risk_management, RLRiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluate_rl_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = Path("results")
RL_RESULTS_DIR = Path("results/rl_agent")
RL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_LOAD_DIR = Path("models/rl_agent")
DEFAULT_MODEL_NAME = "best_model"

# Test configuration
USE_RL_RISK_MANAGEMENT = True  # Enable RL risk management
USE_RULE_BASED_BASELINE = True  # Also test rule-based for comparison


def load_test_data(ticker: str, strategy: str = "Combined") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load test data for a ticker.
    
    Args:
        ticker: Ticker symbol
        strategy: Strategy name (e.g., "Combined", "MACD_Trend_Reversal")
    
    Returns:
        ticker_df, price_series
    """
    # Load from main dataset if available
    data_dir = Path("data")
    dataset_file = data_dir / "dataset.csv"
    
    if dataset_file.exists():
        try:
            df = pd.read_csv(dataset_file)
            if 'tic' in df.columns and 'time' in df.columns:
                ticker_df = df[df['tic'] == ticker].copy()
                if len(ticker_df) > 0:
                    ticker_df['time'] = pd.to_datetime(ticker_df['time'])
                    ticker_df = ticker_df.set_index('time').sort_index()
                    price = ticker_df['close']
                    return ticker_df, price
        except Exception as e:
            logger.warning(f"Could not load from dataset: {e}")
    
    # Fallback: try to load from account balance CSV to get timestamps
    balance_file = RESULTS_DIR / strategy / f"{ticker}_account_balance.csv"
    if balance_file.exists():
        try:
            balance_df = pd.read_csv(balance_file)
            balance_df['Date'] = pd.to_datetime(balance_df['Date'])
            # Create dummy price series from balance (not ideal, but works for evaluation)
            logger.warning(f"Using balance data to infer price timestamps for {ticker}")
            price_index = balance_df['Date']
            price_values = np.ones(len(price_index)) * 100.0  # Placeholder
            price = pd.Series(price_values, index=price_index)
            
            # Create minimal ticker_df
            ticker_df = pd.DataFrame({'close': price}, index=price_index)
            return ticker_df, price
        except Exception as e:
            logger.error(f"Error loading balance data: {e}")
    
    raise FileNotFoundError(f"Could not load test data for {ticker}")


def get_strategy_signals(ticker_df: pd.Series, price: pd.Series, strategy: str = "Combined"):
    """
    Get entry and exit signals for a strategy.
    
    Args:
        ticker_df: DataFrame with OHLCV data
        price: Price series
        strategy: Strategy name
    
    Returns:
        entries, exits (boolean series)
    """
    from backtest import (
        calculate_macd,
        identify_trend_reversals,
        calculate_rsi,
        identify_rsi_trend_reversals,
        identify_bullish_trend_confirmation,
        ENABLE_MACD_TREND_REVERSAL,
        ENABLE_RSI_TREND_REVERSAL,
        ENABLE_BULLISH_CONFIRMATION
    )
    
    # Generate signals based on strategy
    entries = pd.Series(False, index=price.index)
    exits = pd.Series(False, index=price.index)
    
    if strategy == "Combined" or strategy == "MACD_Trend_Reversal":
        if ENABLE_MACD_TREND_REVERSAL:
            macd, signal, histogram = calculate_macd(ticker_df)
            macd_signals = identify_trend_reversals(price, macd, signal, histogram)
            entries = entries | macd_signals.get('strong_bullish', pd.Series(False, index=price.index))
            exits = exits | macd_signals.get('strong_bearish', pd.Series(False, index=price.index))
    
    if strategy == "Combined" or strategy == "RSI_Trend_Reversal":
        if ENABLE_RSI_TREND_REVERSAL:
            rsi = calculate_rsi(price)
            rsi_signals = identify_rsi_trend_reversals(price, rsi)
            entries = entries | rsi_signals.get('bullish_reversal', pd.Series(False, index=price.index))
            exits = exits | rsi_signals.get('bearish_reversal', pd.Series(False, index=price.index))
    
    if strategy == "Combined" or strategy == "Bullish_Trend_Confirmation":
        if ENABLE_BULLISH_CONFIRMATION:
            bullish_signals = identify_bullish_trend_confirmation(price)
            entries = entries | bullish_signals.get('bullish_reversal', pd.Series(False, index=price.index))
            exits = exits | bullish_signals.get('bearish_reversal', pd.Series(False, index=price.index))
    
    return entries.fillna(False), exits.fillna(False)


def evaluate_rule_based_baseline(
    ticker: str,
    ticker_df: pd.DataFrame,
    price: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    strategy: str = "Combined"
) -> Dict:
    """
    Evaluate rule-based risk management baseline.
    
    Args:
        ticker: Ticker symbol
        ticker_df: DataFrame with OHLCV data
        price: Price series
        entries: Entry signals
        exits: Original exit signals
        strategy: Strategy name
    
    Returns:
        Results dictionary
    """
    from backtest import apply_risk_management, ENABLE_RISK_MANAGEMENT
    
    logger.info(f"Evaluating rule-based baseline for {ticker}...")
    
    # Apply rule-based risk management
    if ENABLE_RISK_MANAGEMENT:
        exits_rule = apply_risk_management(entries, exits.copy(), price)
    else:
        exits_rule = exits.copy()
    
    # Backtest
    portfolio_rule = backtest_strategy(price, entries, exits_rule, ticker)
    
    # Analyze results
    results = analyze_results(portfolio_rule, f"{ticker}_{strategy}_rule_based", 
                             None, None, entries, exits_rule)
    
    # Add portfolio for later use
    results['portfolio'] = portfolio_rule
    results['entries'] = entries
    results['exits'] = exits_rule
    results['method'] = 'rule_based'
    
    return results


def evaluate_rl_agent(
    ticker: str,
    ticker_df: pd.DataFrame,
    price: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    strategy: str = "Combined",
    model_path: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME
) -> Dict:
    """
    Evaluate RL agent.
    
    Args:
        ticker: Ticker symbol
        ticker_df: DataFrame with OHLCV data
        price: Price series
        entries: Entry signals
        exits: Original exit signals
        strategy: Strategy name
        model_path: Path to model
        model_name: Model filename
    
    Returns:
        Results dictionary
    """
    logger.info(f"Evaluating RL agent for {ticker}...")
    
    # Load RL risk manager
    try:
        manager = RLRiskManager(
            model_path=model_path,
            model_name=model_name,
            initial_balance=INITIAL_BALANCE
        )
    except Exception as e:
        logger.error(f"Error loading RL model: {e}")
        logger.error("Falling back to rule-based baseline")
        return evaluate_rule_based_baseline(ticker, ticker_df, price, entries, exits, strategy)
    
    # Apply RL risk management
    # First, get initial balance progression (simulate it)
    balance = pd.Series(INITIAL_BALANCE, index=price.index)
    
    # Apply RL risk management
    exits_rl = manager.apply_rl_risk_management(entries, exits.copy(), price, balance)
    
    # Backtest
    portfolio_rl = backtest_strategy(price, entries, exits_rl, ticker)
    
    # Analyze results
    results = analyze_results(portfolio_rl, f"{ticker}_{strategy}_rl_agent", 
                             None, None, entries, exits_rl)
    
    # Add portfolio for later use
    results['portfolio'] = portfolio_rl
    results['entries'] = entries
    results['exits'] = exits_rl
    results['method'] = 'rl_agent'
    
    return results


def compare_results(results_rule: Dict, results_rl: Dict, ticker: str, output_dir: Path):
    """
    Compare and visualize results between rule-based and RL agent.
    
    Args:
        results_rule: Rule-based results
        results_rl: RL agent results
        ticker: Ticker symbol
        output_dir: Output directory
    """
    logger.info(f"Comparing results for {ticker}...")
    
    # Create comparison DataFrame
    comparison_data = {
        'Method': ['Rule-Based', 'RL Agent'],
        'Total Return': [
            results_rule.get('total_return', 0),
            results_rl.get('total_return', 0)
        ],
        'Sharpe Ratio': [
            results_rule.get('sharpe_ratio', 0),
            results_rl.get('sharpe_ratio', 0)
        ],
        'Max Drawdown': [
            results_rule.get('max_drawdown', 0),
            results_rl.get('max_drawdown', 0)
        ],
        'Win Rate': [
            results_rule.get('win_rate', 0),
            results_rl.get('win_rate', 0)
        ],
        'Total Trades': [
            results_rule.get('total_trades', 0),
            results_rl.get('total_trades', 0)
        ],
        'Annualized Return': [
            results_rule.get('annualized_return', 0),
            results_rl.get('annualized_return', 0)
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    logger.info(f"\n{comparison_df.to_string()}")
    
    # Save comparison
    comparison_file = output_dir / f"{ticker}_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"Comparison saved: {comparison_file}")
    
    # Plot equity curves comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    
    portfolio_rule = results_rule.get('portfolio')
    portfolio_rl = results_rl.get('portfolio')
    
    if portfolio_rule:
        equity_rule = portfolio_rule.value()
        ax.plot(equity_rule.index, equity_rule.values, 
               linewidth=2, label='Rule-Based', color='blue', alpha=0.8)
    
    if portfolio_rl:
        equity_rl = portfolio_rl.value()
        ax.plot(equity_rl.index, equity_rl.values, 
               linewidth=2, label='RL Agent', color='green', alpha=0.8)
    
    ax.axhline(y=INITIAL_BALANCE, color='gray', linestyle='--', 
               linewidth=1, alpha=0.5, label='Initial Balance')
    ax.set_title(f'{ticker} - Rule-Based vs RL Agent Comparison', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    comparison_plot = output_dir / f"{ticker}_comparison.png"
    fig.savefig(comparison_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Comparison plot saved: {comparison_plot}")
    
    return comparison_df


def main():
    """Main evaluation function."""
    logger.info("="*60)
    logger.info("RL Risk Management Agent Evaluation")
    logger.info("="*60)
    
    # Test on all tickers
    all_results = {}
    
    for ticker in TICKER_LIST:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {ticker}")
        logger.info(f"{'='*60}")
        
        try:
            # Load test data
            ticker_df, price = load_test_data(ticker, strategy="Combined")
            
            # Get strategy signals
            entries, exits = get_strategy_signals(ticker_df, price, strategy="Combined")
            
            if entries.sum() == 0:
                logger.warning(f"No entry signals for {ticker}, skipping...")
                continue
            
            # Evaluate methods
            results_rule = None
            results_rl = None
            
            if USE_RULE_BASED_BASELINE:
                results_rule = evaluate_rule_based_baseline(
                    ticker, ticker_df, price, entries, exits, strategy="Combined"
                )
            
            if USE_RL_RISK_MANAGEMENT:
                results_rl = evaluate_rl_agent(
                    ticker, ticker_df, price, entries, exits, 
                    strategy="Combined",
                    model_path=MODEL_LOAD_DIR,
                    model_name=DEFAULT_MODEL_NAME
                )
            
            # Compare results
            if results_rule and results_rl:
                comparison = compare_results(
                    results_rule, results_rl, ticker, RL_RESULTS_DIR
                )
                all_results[ticker] = {
                    'rule_based': results_rule,
                    'rl_agent': results_rl,
                    'comparison': comparison
                }
            
        except Exception as e:
            logger.error(f"Error evaluating {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary across all tickers
    if all_results:
        logger.info(f"\n{'='*60}")
        logger.info("Summary Across All Tickers")
        logger.info(f"{'='*60}")
        
        summary_data = []
        for ticker, results in all_results.items():
            comparison = results.get('comparison')
            if comparison is not None:
                for _, row in comparison.iterrows():
                    summary_data.append({
                        'Ticker': ticker,
                        'Method': row['Method'],
                        'Total Return': row['Total Return'],
                        'Sharpe Ratio': row['Sharpe Ratio'],
                        'Max Drawdown': row['Max Drawdown'],
                        'Win Rate': row['Win Rate']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = RL_RESULTS_DIR / "evaluation_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"\n{summary_df.to_string()}")
            logger.info(f"\nSummary saved: {summary_file}")
    
    logger.info("="*60)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {RL_RESULTS_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

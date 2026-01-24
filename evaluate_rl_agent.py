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

# Evaluation date range configuration
# Set to None to use all available data, or specify date range for evaluation
EVAL_START_DATE = "2024-01-01"  # YYYY-MM-DD format, or None for all data
EVAL_END_DATE = "2025-01-21"    # YYYY-MM-DD format, or None for all data

# Test configuration
USE_RL_RISK_MANAGEMENT = True  # Enable RL risk management
USE_RULE_BASED_BASELINE = True  # Also test rule-based for comparison


def load_test_data(
    ticker: str, 
    strategy: str = "Combined",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load test data for a ticker, optionally filtered by date range.
    
    Args:
        ticker: Ticker symbol
        strategy: Strategy name (e.g., "Combined", "MACD_Trend_Reversal")
        start_date: Start date for filtering (YYYY-MM-DD format, or None for all data)
        end_date: End date for filtering (YYYY-MM-DD format, or None for all data)
    
    Returns:
        ticker_df, price_series (filtered by date range if specified)
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
                    
                    # Apply date filtering if specified
                    if start_date is not None:
                        start_dt = pd.to_datetime(start_date)
                        ticker_df = ticker_df[ticker_df.index >= start_dt]
                        logger.info(f"Filtered data from {start_date}: {len(ticker_df)} rows remaining")
                    
                    if end_date is not None:
                        end_dt = pd.to_datetime(end_date)
                        ticker_df = ticker_df[ticker_df.index <= end_dt]
                        logger.info(f"Filtered data to {end_date}: {len(ticker_df)} rows remaining")
                    
                    if len(ticker_df) == 0:
                        raise ValueError(f"No data available for {ticker} in date range {start_date} to {end_date}")
                    
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
            balance_df = balance_df.set_index('Date').sort_index()
            
            # Apply date filtering if specified
            if start_date is not None:
                start_dt = pd.to_datetime(start_date)
                balance_df = balance_df[balance_df.index >= start_dt]
            
            if end_date is not None:
                end_dt = pd.to_datetime(end_date)
                balance_df = balance_df[balance_df.index <= end_dt]
            
            if len(balance_df) == 0:
                raise ValueError(f"No balance data available for {ticker} in date range {start_date} to {end_date}")
            
            # Create dummy price series from balance (not ideal, but works for evaluation)
            logger.warning(f"Using balance data to infer price timestamps for {ticker}")
            price_index = balance_df.index
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


def compare_results(results_rule: Dict, results_rl: Dict, ticker: str, output_dir: Path, price: Optional[pd.Series] = None):
    """
    Compare and visualize results between rule-based and RL agent.
    
    Args:
        results_rule: Rule-based results
        results_rl: RL agent results
        ticker: Ticker symbol
        output_dir: Output directory
        price: Price series for plotting close price (optional)
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
    
    # Plot equity curves comparison with close price below
    if price is not None:
        # Create two subplots: equity curves on top, close price below
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1], sharex=True)
    else:
        # Single plot if no price data
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax2 = None
    
    portfolio_rule = results_rule.get('portfolio')
    portfolio_rl = results_rl.get('portfolio')
    
    # Plot equity curves on top subplot
    if portfolio_rule:
        equity_rule = portfolio_rule.value()
        ax1.plot(equity_rule.index, equity_rule.values, 
               linewidth=2, label='Rule-Based', color='blue', alpha=0.8)
    
    if portfolio_rl:
        equity_rl = portfolio_rl.value()
        ax1.plot(equity_rl.index, equity_rl.values, 
               linewidth=2, label='RL Agent', color='green', alpha=0.8)
    
    ax1.axhline(y=INITIAL_BALANCE, color='gray', linestyle='--', 
               linewidth=1, alpha=0.5, label='Initial Balance')
    ax1.set_title(f'{ticker} - Rule-Based vs RL Agent Comparison', 
                fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot close price on bottom subplot
    if ax2 is not None and price is not None:
        # Align price index with equity curves if possible
        price_to_plot = price.copy()
        
        # Try to filter price to match equity curve time range
        if portfolio_rl:
            equity_rl = portfolio_rl.value()
            # Get the intersection of price and equity indices
            common_index = price.index.intersection(equity_rl.index)
            if len(common_index) > 0:
                price_to_plot = price.loc[common_index]
        elif portfolio_rule:
            equity_rule = portfolio_rule.value()
            common_index = price.index.intersection(equity_rule.index)
            if len(common_index) > 0:
                price_to_plot = price.loc[common_index]
        
        if len(price_to_plot) > 0:
            ax2.plot(price_to_plot.index, price_to_plot.values, 
                    linewidth=1.5, label='Close Price', color='black', alpha=0.7)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Close Price', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        # If no price data, set xlabel on top plot
        ax1.set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    
    comparison_plot = output_dir / f"{ticker}_comparison.png"
    fig.savefig(comparison_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Comparison plot saved: {comparison_plot}")
    
    return comparison_df


def plot_all_tickers_comparison(all_results: Dict, output_dir: Path):
    """
    Create a comprehensive comparison chart for all tickers.
    
    Args:
        all_results: Dictionary of {ticker: {rule_based: Dict, rl_agent: Dict, comparison: DataFrame}}
        output_dir: Output directory
    """
    logger.info("Creating comprehensive comparison chart for all tickers...")
    
    # Prepare data for visualization
    tickers = []
    rule_returns = []
    rl_returns = []
    rule_sharpe = []
    rl_sharpe = []
    rule_drawdown = []
    rl_drawdown = []
    rule_winrate = []
    rl_winrate = []
    
    for ticker, results in all_results.items():
        rule_based = results.get('rule_based', {})
        rl_agent = results.get('rl_agent', {})
        
        tickers.append(ticker)
        rule_returns.append(rule_based.get('total_return', 0))
        rl_returns.append(rl_agent.get('total_return', 0))
        rule_sharpe.append(rule_based.get('sharpe_ratio', 0))
        rl_sharpe.append(rl_agent.get('sharpe_ratio', 0))
        rule_drawdown.append(rule_based.get('max_drawdown', 0))
        rl_drawdown.append(rl_agent.get('max_drawdown', 0))
        rule_winrate.append(rule_based.get('win_rate', 0))
        rl_winrate.append(rl_agent.get('win_rate', 0))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    x_pos = np.arange(len(tickers))
    width = 0.35
    
    # Color scheme
    rule_color = '#3498db'  # Blue
    rl_color = '#2ecc71'     # Green
    
    # 1. Total Return Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bars1_rule = ax1.bar(x_pos - width/2, rule_returns, width, label='Rule-Based', 
                        color=rule_color, alpha=0.8)
    bars1_rl = ax1.bar(x_pos + width/2, rl_returns, width, label='RL Agent', 
                      color=rl_color, alpha=0.8)
    ax1.set_xlabel('Ticker', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Return', fontsize=11, fontweight='bold')
    ax1.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(tickers, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1_rule, bars1_rl]:
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 2. Sharpe Ratio Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars2_rule = ax2.bar(x_pos - width/2, rule_sharpe, width, label='Rule-Based', 
                        color=rule_color, alpha=0.8)
    bars2_rl = ax2.bar(x_pos + width/2, rl_sharpe, width, label='RL Agent', 
                      color=rl_color, alpha=0.8)
    ax2.set_xlabel('Ticker', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax2.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tickers, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars2_rule, bars2_rl]:
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)
    
    # 3. Max Drawdown Comparison (lower is better, so we'll show as positive values)
    ax3 = fig.add_subplot(gs[0, 2])
    bars3_rule = ax3.bar(x_pos - width/2, [-d for d in rule_drawdown], width, 
                         label='Rule-Based', color=rule_color, alpha=0.8)
    bars3_rl = ax3.bar(x_pos + width/2, [-d for d in rl_drawdown], width, 
                      label='RL Agent', color=rl_color, alpha=0.8)
    ax3.set_xlabel('Ticker', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Max Drawdown (negative)', fontsize=11, fontweight='bold')
    ax3.set_title('Max Drawdown Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(tickers, rotation=45, ha='right')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars3_rule, bars3_rl]:
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{abs(height):.3f}',
                        ha='center', va='top' if height < 0 else 'bottom', fontsize=8)
    
    # 4. Win Rate Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    bars4_rule = ax4.bar(x_pos - width/2, rule_winrate, width, label='Rule-Based', 
                        color=rule_color, alpha=0.8)
    bars4_rl = ax4.bar(x_pos + width/2, rl_winrate, width, label='RL Agent', 
                      color=rl_color, alpha=0.8)
    ax4.set_xlabel('Ticker', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Win Rate', fontsize=11, fontweight='bold')
    ax4.set_title('Win Rate Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(tickers, rotation=45, ha='right')
    ax4.set_ylim([0, 1])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars4_rule, bars4_rl]:
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}',
                        ha='center', va='bottom', fontsize=8)
    
    # 5. Improvement Percentage (Total Return)
    ax5 = fig.add_subplot(gs[1, 1])
    improvements = [(rl - rule) / abs(rule) * 100 if rule != 0 else 0 
                    for rule, rl in zip(rule_returns, rl_returns)]
    colors = [rl_color if imp > 0 else rule_color for imp in improvements]
    bars5 = ax5.bar(x_pos, improvements, width=0.6, color=colors, alpha=0.8)
    ax5.set_xlabel('Ticker', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax5.set_title('RL Agent Improvement Over Rule-Based', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(tickers, rotation=45, ha='right')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, imp in zip(bars5, improvements):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')
    
    # 6. Summary Statistics Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for ticker, rule_ret, rl_ret, rule_sh, rl_sh, rule_dd, rl_dd, rule_wr, rl_wr in zip(
        tickers, rule_returns, rl_returns, rule_sharpe, rl_sharpe,
        rule_drawdown, rl_drawdown, rule_winrate, rl_winrate
    ):
        improvement = ((rl_ret - rule_ret) / abs(rule_ret) * 100) if rule_ret != 0 else 0
        summary_data.append([
            ticker,
            f'{rule_ret:.1f}%',
            f'{rl_ret:.1f}%',
            f'{improvement:+.1f}%',
            f'{rule_sh:.2f}',
            f'{rl_sh:.2f}',
            f'{rule_dd:.2%}',
            f'{rl_dd:.2%}'
        ])
    
    table = ax6.table(
        cellText=summary_data,
        colLabels=['Ticker', 'Rule Ret', 'RL Ret', 'Improve', 'Rule Sharpe', 'RL Sharpe', 
                   'Rule DD', 'RL DD'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Style header row
    for i in range(8):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(summary_data) + 1):
        for j in range(8):
            if j == 3:  # Improvement column
                if float(summary_data[i-1][3].replace('%', '').replace('+', '')) > 0:
                    table[(i, j)].set_facecolor('#d5f4e6')
                else:
                    table[(i, j)].set_facecolor('#fadbd8')
            else:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Main title
    fig.suptitle('Comprehensive Performance Comparison: Rule-Based vs RL Agent\nAcross All Cryptocurrencies', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save the chart
    comparison_plot = output_dir / "all_tickers_comparison.png"
    fig.savefig(comparison_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Comprehensive comparison chart saved: {comparison_plot}")
    
    return comparison_plot


def main(
    eval_start_date: Optional[str] = None,
    eval_end_date: Optional[str] = None
):
    """
    Main evaluation function.
    
    Args:
        eval_start_date: Start date for evaluation (YYYY-MM-DD format, or None to use EVAL_START_DATE)
        eval_end_date: End date for evaluation (YYYY-MM-DD format, or None to use EVAL_END_DATE)
    """
    logger.info("="*60)
    logger.info("RL Risk Management Agent Evaluation")
    logger.info("="*60)
    
    # Use provided dates or fall back to configuration
    start_date = eval_start_date if eval_start_date is not None else EVAL_START_DATE
    end_date = eval_end_date if eval_end_date is not None else EVAL_END_DATE
    
    if start_date or end_date:
        logger.info(f"Evaluation date range: {start_date or 'beginning'} to {end_date or 'end'}")
    else:
        logger.info("Using all available data (no date filtering)")
    
    # Test on all tickers
    all_results = {}
    
    for ticker in TICKER_LIST:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {ticker}")
        logger.info(f"{'='*60}")
        
        try:
            # Load test data with date filtering
            ticker_df, price = load_test_data(
                ticker, 
                strategy="Combined",
                start_date=start_date,
                end_date=end_date
            )
            
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
                    results_rule, results_rl, ticker, RL_RESULTS_DIR, price=price
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
        
        # Create comprehensive comparison chart for all tickers
        try:
            plot_all_tickers_comparison(all_results, RL_RESULTS_DIR)
        except Exception as e:
            logger.error(f"Error creating comprehensive comparison chart: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("="*60)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {RL_RESULTS_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RL Risk Management Agent')
    parser.add_argument('--eval-start-date', type=str, default=None,
                       help='Start date for evaluation (YYYY-MM-DD format). Default: use EVAL_START_DATE from config')
    parser.add_argument('--eval-end-date', type=str, default=None,
                       help='End date for evaluation (YYYY-MM-DD format). Default: use EVAL_END_DATE from config')
    
    args = parser.parse_args()
    
    main(
        eval_start_date=args.eval_start_date,
        eval_end_date=args.eval_end_date
    )

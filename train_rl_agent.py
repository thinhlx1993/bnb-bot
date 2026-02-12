"""
Train RL Risk Management Agent using SBX (Stable Baselines Jax) PPO
SBX provides faster training through JAX JIT compilation and efficient computation.
"""

import os
import numpy as np
from pathlib import Path
import multiprocessing
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
from tqdm import tqdm

# RL imports - Using RecurrentPPO from sb3-contrib for recurrent policies
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Custom environment
from rl_risk_env import RiskManagementEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_rl_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_SAVE_DIR = Path("models/rl_agent")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG_DIR = MODEL_SAVE_DIR / "tensorboard_logs"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Ticker list (should match backtest.py)
TICKER_LIST = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "SOLUSDT"]
INITIAL_BALANCE = 1000.0  # Default initial balance

# Training hyperparameters
TOTAL_TIMESTEPS = 1e7  # Total training steps (use early stopping)
LEARNING_RATE = 3e-4  # Initial learning rate
LEARNING_RATE_END = 1e-5  # Final learning rate (for linear decay)
USE_LR_SCHEDULE = False  # Enable learning rate scheduling
BATCH_SIZE = 1024  # Batch size for stable training (increase to 512-1024 for better GPU utilization if memory allows)
N_STEPS = 2048  # Steps per update
N_EPOCHS = 4  # Optimization epochs per update (further reduced to prevent overfitting)
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda
ENT_COEF = 0.01  # Entropy coefficient (exploration) - increased to prevent premature convergence
VF_COEF = 0.25  # Value function coefficient (further reduced to prevent value loss from dominating)
MAX_GRAD_NORM = 0.5  # Maximum gradient norm for clipping
CLIP_RANGE = 0.2  # PPO clip range (standard value, keeps policy updates conservative)

# Device configuration
# RecurrentPPO uses PyTorch and will automatically use GPU if available
USE_GPU = True  # RecurrentPPO will use GPU if available (CUDA), otherwise CPU

# Parallel training configuration (inspired by ElegantRL's multiprocessing approach)
USE_PARALLEL_ENVS = True  # Use SubprocVecEnv for true parallel training (like ElegantRL)
# When True: Uses multiprocessing to run environments in parallel processes
# When False: Uses DummyVecEnv for sequential execution (slower but simpler)
N_ENVS = None  # Number of parallel environments (None = auto-detect CPU count, or set to specific number)
# Note: SubprocVecEnv uses multiprocessing, so each env runs in a separate process
# Recommended: Set to CPU count or slightly less (e.g., cpu_count() - 1 to leave one core free)
# Auto-detection happens in train_ppo_agent() function
# Performance: Parallel training can significantly speed up data collection by utilizing multiple CPU cores

# Multiprocessing start method (like ElegantRL)
# 'spawn' for Windows, 'forkserver' for Linux (more stable than 'fork' with CUDA)
# ElegantRL uses 'spawn' on Windows and 'forkserver' on Linux to avoid CUDA fork issues
MULTIPROCESSING_START_METHOD = 'forkserver' if os.name != 'nt' else 'spawn'

# Model architecture configuration for RecurrentPPO
POLICY_LAYERS = [256, 256]  # Policy network hidden layers (after LSTM)
VALUE_LAYERS = [256, 256]   # Value network hidden layers (after LSTM)
ACTIVATION_FN = 'tanh'  # Activation function: 'tanh', 'relu', or 'elu'
LSTM_HIDDEN_SIZE = 256  # LSTM hidden size
N_LSTM_LAYERS = 1  # Number of LSTM layers
CHECKPOINT_FREQ = 10000  # Save checkpoint every N steps
EVAL_FREQ = 1000  # Evaluate every N steps (single env, all val entry signals, max 1000 steps per episode)
N_EVAL_EPISODES = 500  # Number of episodes to evaluate

# Early stopping configuration
ENABLE_EARLY_STOPPING = False  # Enable early stopping
EARLY_STOPPING_PATIENCE = 50  # Number of evaluations without improvement before stopping
EARLY_STOPPING_MIN_DELTA = 0.0  # Minimum change to qualify as improvement
EARLY_STOPPING_MONITOR = 'mean_ep_length'  # Metric to monitor: 'mean_reward', 'mean_total_reward', 'mean_ep_length', or 'loss'
EARLY_STOPPING_MODE = 'max'  # 'max' for reward/ep_length (higher is better), 'min' for loss (lower is better)

# ============== Date Range Configuration ==============
# Set to None to use all available data, or specify date range (YYYY-MM-DD format)
# Time-series split: TRAIN -> VALIDATION -> EVAL (chronological order)

# Training data date range  # YYYY-MM-DD format, or None for all data
TRAIN_START_DATE = "2010-01-01"  # Start date for training data (None = beginning of data)
TRAIN_END_DATE = "2024-01-01"    # End date for training data (None = end of data)

# Validation data date range (for evaluation during training)
VAL_START_DATE = "2024-01-01"    # Start date for validation data (None = after TRAIN_END_DATE)
VAL_END_DATE = "2025-01-01"      # End date for validation data (None = end of data)

# Evaluation data date range (for final evaluation after training)
EVAL_START_DATE = "2025-01-01"   # Start date for evaluation (None = use VAL dates)
EVAL_END_DATE = "2026-01-24"              # End date for evaluation (None = end of data)

# Technical indicator settings for entry signal generation
USE_TECHNICAL_SIGNALS = True  # Use technical signals for entry points (if False, random entry)
USE_BACKTEST_SIGNALS = True   # Use shared entry_signal_generator (same as evaluate/live); if False, use in-file crossover signals
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


# ============== Learning Rate Schedule Functions ==============

def linear_schedule(initial_lr: float, final_lr: float = 0.0) -> callable:
    """
    Linear learning rate schedule.
    
    Args:
        initial_lr: Initial learning rate
        final_lr: Final learning rate (default 0.0)
    
    Returns:
        Schedule function that takes progress_remaining (1.0 -> 0.0) and returns LR
    """
    def schedule(progress_remaining: float) -> float:
        """
        Progress remaining goes from 1.0 (start) to 0.0 (end of training).
        """
        return final_lr + progress_remaining * (initial_lr - final_lr)
    
    return schedule


# ============== Technical Indicator Functions ==============

def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL):
    """Calculate MACD, Signal, and Histogram."""
    close = df['close'] if 'close' in df.columns else df
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
    rsi = 100 - (100 / (1 + rs))
    return rsi


def identify_macd_signals(price: pd.Series, macd: pd.Series, signal: pd.Series, histogram: pd.Series) -> pd.Series:
    """
    Identify MACD-based entry signals (bullish divergence / crossover).
    
    Returns:
        Boolean series where True indicates a buy signal
    """
    entries = pd.Series(False, index=price.index)
    
    # MACD line crosses above signal line (bullish crossover)
    macd_cross_up = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    
    # Histogram turns positive (momentum shift)
    hist_positive = (histogram > 0) & (histogram.shift(1) <= 0)
    
    # Combine signals
    entries = macd_cross_up | hist_positive
    
    return entries.fillna(False)


def identify_rsi_signals(price: pd.Series, rsi: pd.Series) -> pd.Series:
    """
    Identify RSI-based entry signals (oversold bounce).
    
    Returns:
        Boolean series where True indicates a buy signal
    """
    entries = pd.Series(False, index=price.index)
    
    # RSI crosses above oversold level (30) - potential reversal
    rsi_cross_up = (rsi > RSI_OVERSOLD) & (rsi.shift(1) <= RSI_OVERSOLD)
    
    # RSI was oversold and now rising
    rsi_rising = (rsi < 50) & (rsi > rsi.shift(1)) & (rsi.shift(1) < RSI_OVERSOLD + 10)
    
    # Combine signals
    entries = rsi_cross_up | rsi_rising
    
    return entries.fillna(False)


def identify_macd_exit_signals(price: pd.Series, macd: pd.Series, signal: pd.Series, histogram: pd.Series) -> pd.Series:
    """
    Identify MACD-based exit signals (bearish divergence / crossover).
    
    Returns:
        Boolean series where True indicates a sell signal
    """
    exits = pd.Series(False, index=price.index)
    
    # MACD line crosses below signal line (bearish crossover)
    macd_cross_down = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    
    # Histogram turns negative (momentum shift)
    hist_negative = (histogram < 0) & (histogram.shift(1) >= 0)
    
    # Combine signals
    exits = macd_cross_down | hist_negative
    
    return exits.fillna(False)


def identify_rsi_exit_signals(price: pd.Series, rsi: pd.Series) -> pd.Series:
    """
    Identify RSI-based exit signals (overbought reversal).
    
    Returns:
        Boolean series where True indicates a sell signal
    """
    exits = pd.Series(False, index=price.index)
    
    # RSI crosses below overbought level (70) - potential reversal
    rsi_cross_down = (rsi < RSI_OVERBOUGHT) & (rsi.shift(1) >= RSI_OVERBOUGHT)
    
    # RSI was overbought and now falling
    rsi_falling = (rsi > 50) & (rsi < rsi.shift(1)) & (rsi.shift(1) > RSI_OVERBOUGHT - 10)
    
    # Combine signals
    exits = rsi_cross_down | rsi_falling
    
    return exits.fillna(False)


def generate_entry_signals(ticker_df: pd.DataFrame, price: pd.Series) -> pd.Series:
    """
    Generate combined entry signals from multiple technical indicators.
    
    Args:
        ticker_df: DataFrame with OHLCV data
        price: Close price series
    
    Returns:
        Boolean series where True indicates a buy signal
    """
    entries = pd.Series(False, index=price.index)
    
    try:
        # MACD signals
        macd, signal, histogram = calculate_macd(ticker_df)
        macd_entries = identify_macd_signals(price, macd, signal, histogram)
        entries = entries | macd_entries
        
        # RSI signals
        rsi = calculate_rsi(price)
        rsi_entries = identify_rsi_signals(price, rsi)
        entries = entries | rsi_entries
        
    except Exception as e:
        logger.warning(f"Error generating entry signals: {e}")
        # Return empty signals on error
        return pd.Series(False, index=price.index)
    
    return entries.fillna(False)


def generate_exit_signals(ticker_df: pd.DataFrame, price: pd.Series) -> pd.Series:
    """
    Generate combined exit signals from multiple technical indicators.
    
    Args:
        ticker_df: DataFrame with OHLCV data
        price: Close price series
    
    Returns:
        Boolean series where True indicates a sell signal
    """
    exits = pd.Series(False, index=price.index)
    
    try:
        # MACD exit signals
        macd, signal, histogram = calculate_macd(ticker_df)
        macd_exits = identify_macd_exit_signals(price, macd, signal, histogram)
        exits = exits | macd_exits
        
        # RSI exit signals
        rsi = calculate_rsi(price)
        rsi_exits = identify_rsi_exit_signals(price, rsi)
        exits = exits | rsi_exits
        
    except Exception as e:
        logger.warning(f"Error generating exit signals: {e}")
        # Return empty signals on error
        return pd.Series(False, index=price.index)
    
    return exits.fillna(False)


def _win_rate_from_returns(episode_returns: List[float]) -> Tuple[float, int, int]:
    """
    Win rate = fraction of episodes with positive balance (positive realized return).
    An episode is a win when episode_return > 0.
    Returns (win_rate, wins, n_episodes).
    """
    if not episode_returns:
        return 0.0, 0, 0
    wins = sum(1 for r in episode_returns if r > 0)
    return wins / len(episode_returns), wins, len(episode_returns)


def _compute_eval_metrics(
    episode_rewards: List[float],
    episode_lengths: List[int],
    episode_returns: Optional[List[float]] = None,
    initial_balance: float = INITIAL_BALANCE,
    failure_count: int = 0,
) -> Dict[str, float]:
    """
    Compute mean/std reward, win rate, holding time, total balance, and failure count.
    Win rate: fraction of completed episodes with positive balance (episode_return > 0).
    Only completed episodes (that closed within max_steps) are included in metrics.
    """
    if episode_rewards and episode_lengths:
        rewards_per_step = [r / max(l, 1) for r, l in zip(episode_rewards, episode_lengths)]
        mean_reward = np.mean(rewards_per_step)
        std_reward = np.std(rewards_per_step)
        mean_total_reward = np.mean(episode_rewards)
        std_total_reward = np.std(episode_rewards)
        
        # Win rate = fraction of completed episodes with positive balance
        if episode_returns is not None and len(episode_returns) == len(episode_rewards):
            win_rate, _, _ = _win_rate_from_returns(episode_returns)
        else:
            win_rate = 0.0
        
        # Total balance: INITIAL_BALANCE + sum of all episode returns
        # total_balance = INITIAL_BALANCE + sum(episode_returns) * INITIAL_BALANCE
        if episode_returns is not None and len(episode_returns) > 0:
            total_balance = initial_balance + sum(episode_returns) * initial_balance
        else:
            total_balance = initial_balance
        
        avg_holding_time = np.mean(episode_lengths)
        std_holding_time = np.std(episode_lengths)
        min_holding_time = np.min(episode_lengths)
        max_holding_time = np.max(episode_lengths)
        mean_ep_length = np.mean(episode_lengths)
    else:
        mean_reward = std_reward = mean_total_reward = std_total_reward = 0.0
        win_rate = avg_holding_time = std_holding_time = min_holding_time = max_holding_time = mean_ep_length = 0.0
        total_balance = initial_balance
    return {
        "mean_reward": mean_reward, "std_reward": std_reward,
        "mean_total_reward": mean_total_reward, "std_total_reward": std_total_reward,
        "win_rate": win_rate, "mean_ep_length": mean_ep_length,
        "avg_holding_time": avg_holding_time, "std_holding_time": std_holding_time,
        "min_holding_time": min_holding_time, "max_holding_time": max_holding_time,
        "total_balance": total_balance,
        "failure_count": failure_count,
    }


class EarlyStoppingCallback(BaseCallback):
    """
    Custom early stopping callback that monitors validation loss/metrics.
    Stops training if the monitored metric doesn't improve for a specified number of evaluations.
    """
    
    def __init__(
        self,
        eval_callback,  # EvalCallback instance to monitor
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = EARLY_STOPPING_MIN_DELTA,
        monitor: str = EARLY_STOPPING_MONITOR,
        mode: str = EARLY_STOPPING_MODE,
        verbose: int = 1
    ):
        """
        Initialize early stopping callback.
        
        Args:
            eval_callback: EvalCallback instance to monitor
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor:
                - 'mean_reward': reward per step (normalized by episode length)
                - 'mean_total_reward': total episode reward (not normalized)
                - 'mean_ep_length': average episode length
                - 'win_rate': win rate (positive rewards / total episodes)
                - 'loss': negative mean reward (for minimization)
            mode: 'max' for reward/ep_length/win_rate, 'min' for loss
            verbose: Verbosity level
        """
        super().__init__(verbose=verbose)
        self.eval_callback = eval_callback
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        # Track best value and patience counter
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.patience_counter = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.last_eval_step = -1
        self.last_mean_reward = None
        
        logger.info(f"Early stopping callback initialized:")
        logger.info(f"  Monitor: {monitor}")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Patience: {patience} evaluations")
        logger.info(f"  Min delta: {min_delta}")
    
    def _on_step(self) -> bool:
        """
        Called at each step. Check if early stopping should be triggered.
        
        Returns:
            True if training should continue, False if should stop
        """
        # Only check after evaluation has occurred
        # Check if a new evaluation has happened by comparing current mean_reward with last
        if not hasattr(self.eval_callback, 'last_mean_reward'):
            return True
        
        current_mean_reward = self.eval_callback.last_mean_reward
        
        # Check if this is a new evaluation (mean_reward changed)
        if self.last_mean_reward is not None and current_mean_reward == self.last_mean_reward:
            return True  # No new evaluation yet
        
        # New evaluation detected
        self.last_mean_reward = current_mean_reward
        
        # Get current metric value from evaluation callback
        if self.monitor == 'mean_reward':
            if not hasattr(self.eval_callback, 'last_mean_reward'):
                return True
            current_value = self.eval_callback.last_mean_reward
            if current_value is None:
                return True
            try:
                current_value = float(current_value)
                if np.isnan(current_value) or np.isinf(current_value):
                    return True  # Skip if NaN or Inf
            except (TypeError, ValueError):
                return True  # Skip if not numeric
        
        elif self.monitor == 'mean_ep_length':
            if not hasattr(self.eval_callback, 'last_mean_ep_length'):
                return True
            current_value = self.eval_callback.last_mean_ep_length
            if current_value is None:
                return True
            try:
                current_value = float(current_value)
                if np.isnan(current_value) or np.isinf(current_value):
                    return True  # Skip if NaN or Inf
            except (TypeError, ValueError):
                return True  # Skip if not numeric
        
        elif self.monitor == 'mean_total_reward':
            # Total reward (not normalized by episode length)
            if not hasattr(self.eval_callback, 'last_mean_total_reward'):
                return True
            current_value = self.eval_callback.last_mean_total_reward
            if current_value is None:
                return True
            try:
                current_value = float(current_value)
                if np.isnan(current_value) or np.isinf(current_value):
                    return True  # Skip if NaN or Inf
            except (TypeError, ValueError):
                return True  # Skip if not numeric
        
        elif self.monitor == 'win_rate':
            # Win rate (positive rewards / total episodes)
            if not hasattr(self.eval_callback, 'last_win_rate'):
                return True
            current_value = self.eval_callback.last_win_rate
            if current_value is None:
                return True
            try:
                current_value = float(current_value)
                if np.isnan(current_value) or np.isinf(current_value):
                    return True  # Skip if NaN or Inf
            except (TypeError, ValueError):
                return True  # Skip if not numeric
        
        elif self.monitor == 'loss':
            # For loss, we'll use negative mean reward as proxy
            # (lower reward = higher loss, so we minimize negative reward)
            if not hasattr(self.eval_callback, 'last_mean_reward'):
                return True
            reward = self.eval_callback.last_mean_reward
            if reward is None:
                return True
            try:
                current_value = -float(reward)
                if np.isnan(current_value) or np.isinf(current_value):
                    return True  # Skip if NaN or Inf
            except (TypeError, ValueError):
                return True  # Skip if not numeric
            # Use min mode for loss
            effective_mode = 'min'
        else:
            logger.warning(f"Unknown monitor metric: {self.monitor}, using mean_reward")
            if not hasattr(self.eval_callback, 'last_mean_reward'):
                return True
            current_value = self.eval_callback.last_mean_reward
            if current_value is None:
                return True
            try:
                current_value = float(current_value)
                if np.isnan(current_value) or np.isinf(current_value):
                    return True
            except (TypeError, ValueError):
                return True
            effective_mode = self.mode
        
        # Use effective mode for loss, otherwise use configured mode
        if self.monitor == 'loss':
            check_mode = effective_mode
        else:
            check_mode = self.mode
        
        # Check for improvement
        if check_mode == 'max':
            improved = current_value >= self.best_value + self.min_delta
        else:  # mode == 'min'
            improved = current_value <= self.best_value - self.min_delta
        
        if improved:
            self.best_value = current_value
            self.wait = 0
            if self.verbose >= 1:
                logger.info(f"Early stopping: {self.monitor} improved to {current_value:.4f} (best: {self.best_value:.4f})")
        else:
            self.wait += 1
            if self.verbose >= 1:
                logger.info(f"Early stopping: {self.monitor} did not improve. Wait: {self.wait}/{self.patience} (current: {current_value:.4f}, best: {self.best_value:.4f})")
        
        # Check if patience exceeded
        if self.wait >= self.patience:
            self.stopped_epoch = self.num_timesteps
            logger.info(f"\n{'='*60}")
            logger.info(f"Early stopping triggered at {self.num_timesteps} timesteps!")
            logger.info(f"  Best {self.monitor}: {self.best_value:.4f}")
            logger.info(f"  Stopped after {self.patience} evaluations without improvement")
            logger.info(f"{'='*60}\n")
            return False  # Stop training
        
        return True  # Continue training


def filter_data_by_date(
    data: pd.Series,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.Series:
    """
    Filter time-series data by date range.
    
    Args:
        data: Series with datetime index
        start_date: Start date (YYYY-MM-DD format, or None)
        end_date: End date (YYYY-MM-DD format, or None)
    
    Returns:
        Filtered series
    """
    filtered = data.copy()
    
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        filtered = filtered[filtered.index >= start_dt]
    
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        filtered = filtered[filtered.index <= end_dt]
    
    return filtered


# Cache for entry/exit signals to avoid recomputing on every run
SIGNALS_CACHE_DIR = Path("data/signals_cache")


def _signals_cache_path(ticker: str, start_date: Optional[str], end_date: Optional[str], strategy: str, use_backtest: bool) -> Path:
    """Path to cache file for one ticker's signals."""
    safe_strategy = (strategy or "Combined").replace("/", "_").replace("\\", "_")
    name = f"{ticker}_{start_date or 'none'}_{end_date or 'none'}_{safe_strategy}_{use_backtest}.parquet"
    return SIGNALS_CACHE_DIR / name


def _load_signals_cache(cache_path: Path, price_index: pd.Index) -> Optional[Tuple[pd.Series, pd.Series]]:
    """Load entry/exit signals from cache if file exists and index matches price_series.index."""
    if not cache_path.exists():
        return None
    try:
        cached = pd.read_parquet(cache_path)
        if len(cached.index) != len(price_index) or not cached.index.equals(price_index):
            return None
        entry_signals = cached["entry_signals"].astype(bool)
        exit_signals = cached["exit_signals"].astype(bool)
        return (entry_signals, exit_signals)
    except Exception as e:
        logger.debug(f"Cache read failed {cache_path}: {e}")
        return None


def _save_signals_cache(cache_path: Path, entry_signals: pd.Series, exit_signals: pd.Series) -> None:
    """Save entry/exit signals to cache."""
    try:
        if entry_signals is None or exit_signals is None:
            logger.warning(f"Cannot save cache: signals are None for {cache_path}")
            return
        if len(entry_signals) == 0 or len(exit_signals) == 0:
            logger.warning(f"Cannot save cache: signals are empty for {cache_path}")
            return
        if not entry_signals.index.equals(exit_signals.index):
            logger.warning(f"Cannot save cache: signal indices don't match for {cache_path}")
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"entry_signals": entry_signals, "exit_signals": exit_signals})
        df.to_parquet(cache_path, index=True)
        logger.info(f"Cache saved: {cache_path.name}")
    except ImportError as e:
        logger.warning(f"Cache write failed: parquet library not available. Install pyarrow or fastparquet: {e}")
    except Exception as e:
        logger.warning(f"Cache write failed {cache_path}: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")


def load_all_tickers_data(
    tickers_list: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    results_dir: Path = Path("results"),
    strategy: str = "Combined"
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Load price, balance, and entry signal data for all tickers.
    Entry/exit signals are cached under data/signals_cache to avoid recomputing.
    
    Args:
        tickers_list: List of ticker symbols
        start_date: Start date for filtering (YYYY-MM-DD format, or None)
        end_date: End date for filtering (YYYY-MM-DD format, or None)
        results_dir: Results directory containing balance data
        strategy: Strategy name for balance data
    
    Returns:
        Dict of {ticker: {'price': Series, 'balance': Series, 'entry_signals': Series, 'exit_signals': Series}}
    """
    date_range_str = ""
    if start_date or end_date:
        date_range_str = f" ({start_date or 'start'} to {end_date or 'end'})"
    logger.info(f"Loading data for {len(tickers_list)} tickers{date_range_str}...")
    
    all_tickers_data = {}
    data_dir = Path("data")
    dataset_file = data_dir / "dataset.csv"
    
    # Load price data from dataset
    if dataset_file.exists():
        try:
            df = pd.read_csv(dataset_file)
            if 'tic' in df.columns and 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time').sort_index()
                
                for ticker in tickers_list:
                    ticker_df = df[df['tic'] == ticker].copy()
                    if len(ticker_df) > 0:
                        # Filter by date range
                        if start_date is not None:
                            start_dt = pd.to_datetime(start_date)
                            ticker_df = ticker_df[ticker_df.index >= start_dt]
                        if end_date is not None:
                            end_dt = pd.to_datetime(end_date)
                            ticker_df = ticker_df[ticker_df.index <= end_dt]
                        
                        if len(ticker_df) == 0:
                            logger.warning(f"  No data for {ticker} in date range {start_date} to {end_date}")
                            continue
                        
                        price_series = ticker_df['close']
                        
                        # Create default balance series
                        balance_series = pd.Series(INITIAL_BALANCE, index=price_series.index)
                        
                        # Generate entry and exit signals (use cache when available)
                        entry_signals = None
                        exit_signals = None
                        if USE_TECHNICAL_SIGNALS:
                            cache_path = _signals_cache_path(
                                ticker, start_date, end_date, strategy, USE_BACKTEST_SIGNALS
                            )
                            cached = _load_signals_cache(cache_path, price_series.index)
                            if cached is not None:
                                entry_signals, exit_signals = cached
                                num_entry_signals = entry_signals.sum()
                                num_exit_signals = exit_signals.sum()
                                logger.info(f"  {ticker}: {len(price_series)} price points, {num_entry_signals} entry, {num_exit_signals} exit signals (from cache)")
                            else:
                                if USE_BACKTEST_SIGNALS:
                                    from entry_signal_generator import get_strategy_signals
                                    try:
                                        entry_signals, exit_signals = get_strategy_signals(ticker_df, price_series, strategy=strategy)
                                    except Exception as e:
                                        logger.warning(f"  entry_signal_generator failed for {ticker}: {e}, falling back to crossover signals")
                                        entry_signals = generate_entry_signals(ticker_df, price_series)
                                        exit_signals = generate_exit_signals(ticker_df, price_series)
                                else:
                                    entry_signals = generate_entry_signals(ticker_df, price_series)
                                    exit_signals = generate_exit_signals(ticker_df, price_series)
                                num_entry_signals = entry_signals.sum()
                                num_exit_signals = exit_signals.sum()
                                logger.info(f"  {ticker}: {len(price_series)} price points, {num_entry_signals} entry signals, {num_exit_signals} exit signals")
                                _save_signals_cache(cache_path, entry_signals, exit_signals)
                        else:
                            logger.info(f"  {ticker}: {len(price_series)} price points (random entry)")
                        
                        all_tickers_data[ticker] = {
                            'price': price_series,
                            'balance': balance_series,
                            'entry_signals': entry_signals,
                            'exit_signals': exit_signals,
                            'ohlcv': ticker_df,  # OHLCV for CCI/Williams/TSI/ROC and OBV/MFI/AD/PVT in obs
                        }
                    else:
                        logger.warning(f"  No data found for {ticker}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.error(f"Dataset file not found: {dataset_file}")
        logger.error("Please run: python backtest.py --download-only")
    
    if len(all_tickers_data) == 0:
        logger.error("No ticker data loaded!")
    
    logger.info(f"Successfully loaded data for {len(all_tickers_data)} tickers")
    return all_tickers_data


def create_env_factory(all_tickers_data: Dict[str, Dict[str, pd.Series]], seed: int = 42) -> callable:
    """
    Create environment factory that uses all tickers' data.
    On reset, randomly selects ticker and entry point.
    
    Args:
        all_tickers_data: Dict of {ticker: {'price': Series, 'balance': Series}}
        seed: Random seed
    
    Returns:
        Function that returns an environment
    """
    np.random.seed(seed)
    
    def _make_env(rank: int = 0):
        # Create environment with all tickers' data
        # Entry point will be randomly selected on reset
        # max_steps = number of bars from entry the bot can play (episode ends at entry + max_steps or end of data)
        env = RiskManagementEnv(
            all_tickers_data=all_tickers_data,
            initial_balance=INITIAL_BALANCE,
            history_length=50,
            max_steps=500,  # Steps from entry per episode
            fee_rate=0.001,
        )
        # Wrap with Monitor for statistics
        env = Monitor(env, filename=None, allow_early_resets=True)
        env.reset(seed=seed + rank)
        return env
    
    return _make_env


def train_ppo_agent(
    all_tickers_data: Dict[str, Dict[str, pd.Series]],
    model_save_dir: Path,
    tensorboard_log_dir: Path,
    total_timesteps: int = TOTAL_TIMESTEPS,
    n_envs: Optional[int] = None,
    val_tickers_data: Optional[Dict[str, Dict[str, pd.Series]]] = None
):
    """
    Train PPO agent on all tickers' data with parallel environment collection.
    
    Args:
        all_tickers_data: Dict of {ticker: {'price': Series, 'balance': Series}} for training
        model_save_dir: Directory to save models
        tensorboard_log_dir: Directory for TensorBoard logs
        total_timesteps: Total training steps
        n_envs: Number of parallel environments (None = use N_ENVS from config)
        val_tickers_data: Dict of {ticker: {'price': Series, 'balance': Series}} for validation/evaluation
    """
    # Use validation tickers data if provided, otherwise use training data
    if val_tickers_data is None:
        val_tickers_data = all_tickers_data
    
    # Set number of environments (auto-detect if not set)
    if n_envs is None:
        if N_ENVS is None:
            cpu_count = os.cpu_count() or 4  # Fallback to 4 if cpu_count() returns None
            n_envs = max(1, cpu_count - 1)  # Use all but one CPU core by default
            logger.info(f"Auto-detected CPU count: {cpu_count}, using {n_envs} parallel environments")
        else:
            n_envs = N_ENVS
    
    logger.info("="*60)
    logger.info("Training PPO Agent")
    logger.info("="*60)
    
    # Set multiprocessing start method (like ElegantRL)
    # This must be done before creating SubprocVecEnv
    try:
        multiprocessing.set_start_method(MULTIPROCESSING_START_METHOD, force=True)
        logger.info(f"Multiprocessing start method set to: {MULTIPROCESSING_START_METHOD}")
    except RuntimeError as e:
        # Start method already set, which is fine
        logger.info(f"Multiprocessing start method already set: {multiprocessing.get_start_method()}")
    
    # Choose vectorized environment class based on configuration
    if USE_PARALLEL_ENVS:
        vec_env_cls = SubprocVecEnv
        env_type = "parallel (SubprocVecEnv)"
        logger.info(f"Using parallel training with {n_envs} environments (true multiprocessing)")
    else:
        vec_env_cls = DummyVecEnv
        env_type = "sequential (DummyVecEnv)"
        logger.info(f"Using sequential training with {n_envs} environments")
    
    # Create vectorized environments
    logger.info(f"Creating {n_envs} {env_type} environments...")
    train_env_factory = create_env_factory(all_tickers_data, seed=42)
    train_env = make_vec_env(
        train_env_factory,
        n_envs=n_envs,
        vec_env_cls=vec_env_cls
    )
    
    # Wrap training environment with VecNormalize for observation normalization
    logger.info("Wrapping training environment with VecNormalize...")
    train_env = VecNormalize(
        train_env,
        norm_obs=True,  # Normalize observations
        norm_reward=False,  # Don't normalize rewards (rewards are already scaled appropriately)
        clip_obs=10.0,  # Clip observations to prevent extreme values
        training=True  # Enable training mode (updates running statistics)
    )
    
    # Create evaluation environment for EvalCallback
    # Use same vec_env_cls as training env to avoid type mismatch warning
    # For evaluation, we use n_envs=1 (single environment)
    logger.info("Creating evaluation environment...")
    eval_env_factory = create_env_factory(val_tickers_data, seed=123)
    eval_env = make_vec_env(
        eval_env_factory,
        n_envs=1,  # Single environment for evaluation
        vec_env_cls=vec_env_cls  # Use same type as training env
    )
    
    # Wrap evaluation environment with VecNormalize (will sync stats from training env)
    logger.info("Wrapping evaluation environment with VecNormalize...")
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False  # Evaluation mode (uses fixed statistics)
    )
    
    
    logger.info("Eval callback will use standard EvalCallback with validation environment")
    
    # Create RecurrentPPO model with LSTM architecture
    logger.info("Creating RecurrentPPO model with LSTM architecture...")
    
    # Map string names to PyTorch activation functions
    import torch.nn as nn
    activation_map = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU
    }
    
    activation_fn_name = ACTIVATION_FN.lower()
    if activation_fn_name not in activation_map:
        logger.warning(f"Unknown activation function {activation_fn_name}, defaulting to 'tanh'")
        activation_fn_name = 'tanh'
    
    activation_fn = activation_map[activation_fn_name]
    
    policy_kwargs = dict(
        # Network architecture after LSTM
        net_arch={
            "pi": POLICY_LAYERS,  # Policy network layers (after LSTM)
            "vf": VALUE_LAYERS    # Value network layers (after LSTM)
        },
        activation_fn=activation_fn,
        # LSTM configuration
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        n_lstm_layers=N_LSTM_LAYERS,
        enable_critic_lstm=True,  # Use separate LSTM for critic
        shared_lstm=False  # Separate LSTMs for actor and critic
    )
    
    # Set device for PyTorch
    import torch
    device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Set up learning rate (constant or scheduled)
    if USE_LR_SCHEDULE:
        lr_schedule = linear_schedule(LEARNING_RATE, LEARNING_RATE_END)
        logger.info(f"Using linear LR schedule: {LEARNING_RATE} -> {LEARNING_RATE_END}")
    else:
        lr_schedule = LEARNING_RATE
        logger.info(f"Using constant LR: {LEARNING_RATE}")
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=lr_schedule,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        clip_range=CLIP_RANGE,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_log_dir),
        verbose=0,
        device=device
    )
    
    logger.info(f"Model architecture (RecurrentPPO):")
    logger.info(f"  LSTM hidden size: {LSTM_HIDDEN_SIZE}")
    logger.info(f"  LSTM layers: {N_LSTM_LAYERS}")
    logger.info(f"  Policy network (after LSTM): {POLICY_LAYERS} ({len(POLICY_LAYERS)} hidden layers)")
    logger.info(f"  Value network (after LSTM): {VALUE_LAYERS} ({len(VALUE_LAYERS)} hidden layers)")
    logger.info(f"  Activation function: {ACTIVATION_FN}")
    logger.info(f"  Framework: RecurrentPPO (PyTorch) - LSTM for temporal dependencies")
    # Estimate parameters: sum of layer sizes squared
    pi_params = sum(POLICY_LAYERS[i] * POLICY_LAYERS[i+1] for i in range(len(POLICY_LAYERS)-1))
    vf_params = sum(VALUE_LAYERS[i] * VALUE_LAYERS[i+1] for i in range(len(VALUE_LAYERS)-1))
    total_params = (pi_params + vf_params) + sum(POLICY_LAYERS) + sum(VALUE_LAYERS)  # Approximate
    logger.info(f"  Estimated parameters: ~{total_params // 1000}K")
    if USE_LR_SCHEDULE:
        logger.info(f"  Learning rate: {LEARNING_RATE} -> {LEARNING_RATE_END} (linear decay)")
    else:
        logger.info(f"  Learning rate: {LEARNING_RATE} (constant)")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Steps per update: {N_STEPS}")
    logger.info(f"  Epochs per update: {N_EPOCHS}")
    
    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(model_save_dir / "checkpoints"),
        name_prefix="ppo_risk_agent",
        save_replay_buffer=True,
        save_vecnormalize=True  # Save VecNormalize statistics with checkpoints
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback - standard EvalCallback with VecNormalize saving
    # Create a wrapper callback that syncs VecNormalize stats before evaluation
    class VecNormalizeSyncCallback(BaseCallback):
        """Callback to sync VecNormalize statistics before evaluation."""
        def __init__(self, train_env, eval_env, verbose=0):
            super().__init__(verbose)
            self.train_env = train_env
            self.eval_env = eval_env
        
        def _on_step(self) -> bool:
            # Sync VecNormalize stats from training to evaluation environment
            # This ensures eval uses the same normalization as training
            if hasattr(self.train_env, 'obs_rms') and hasattr(self.eval_env, 'obs_rms'):
                self.eval_env.obs_rms.mean = self.train_env.obs_rms.mean.copy()
                self.eval_env.obs_rms.var = self.train_env.obs_rms.var.copy()
                self.eval_env.obs_rms.count = self.train_env.obs_rms.count
            return True
    
    # Create sync callback
    sync_callback = VecNormalizeSyncCallback(train_env, eval_env, verbose=1)
    callbacks.append(sync_callback)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_save_dir / "best_model"),
        log_path=str(model_save_dir / "eval_logs"),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=N_EVAL_EPISODES,  # Number of episodes to evaluate
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Also create a callback to save VecNormalize when best model is saved
    # This monitors the eval_callback's best_mean_reward
    class SaveVecNormalizeWithBestModel(BaseCallback):
        """Save VecNormalize statistics when best model is saved by EvalCallback."""
        def __init__(self, train_env, eval_callback, best_model_save_path, verbose=0):
            super().__init__(verbose)
            self.train_env = train_env
            self.eval_callback = eval_callback
            self.best_model_save_path = Path(best_model_save_path)
            self.last_best_mean_reward = float('-inf')
        
        def _on_step(self) -> bool:
            # Check if eval_callback has a new best model
            if hasattr(self.eval_callback, 'best_mean_reward'):
                current_best = self.eval_callback.best_mean_reward
                if current_best > self.last_best_mean_reward:
                    self.last_best_mean_reward = current_best
                    # Save VecNormalize statistics to the same directory as best_model.zip
                    # EvalCallback saves best_model.zip in best_model_save_path directory
                    vec_normalize_path = self.best_model_save_path / "vec_normalize.pkl"
                    vec_normalize_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        self.train_env.save(str(vec_normalize_path))
                        if self.verbose >= 1:
                            logger.info(f"VecNormalize statistics saved with best model (mean_reward: {current_best:.2f})")
                            logger.info(f"  Saved to: {vec_normalize_path}")
                    except Exception as e:
                        logger.warning(f"Could not save VecNormalize with best model: {e}")
            return True
    
    save_vecnorm_callback = SaveVecNormalizeWithBestModel(
        train_env, eval_callback, 
        best_model_save_path=str(model_save_dir / "best_model"),
        verbose=1
    )
    callbacks.append(save_vecnorm_callback)
    
    # Early stopping callback
    if ENABLE_EARLY_STOPPING:
        early_stopping_callback = EarlyStoppingCallback(
            eval_callback=eval_callback,
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            monitor=EARLY_STOPPING_MONITOR,
            mode=EARLY_STOPPING_MODE,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
        logger.info(f"Early stopping enabled:")
        logger.info(f"  Monitor: {EARLY_STOPPING_MONITOR}")
        logger.info(f"  Mode: {EARLY_STOPPING_MODE}")
        logger.info(f"  Patience: {EARLY_STOPPING_PATIENCE} evaluations")
        logger.info(f"  Min delta: {EARLY_STOPPING_MIN_DELTA}")
    else:
        logger.info("Early stopping disabled")
    
    callback_list = CallbackList(callbacks)
    
    # Train model
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    logger.info(f"Parallel training: {USE_PARALLEL_ENVS} ({env_type})")
    logger.info(f"Number of parallel environments: {n_envs}")
    logger.info(f"Checkpoints will be saved every {CHECKPOINT_FREQ} steps")
    logger.info(f"Evaluation every {EVAL_FREQ} steps")
    logger.info(f"TensorBoard logs: {tensorboard_log_dir}")
    if USE_PARALLEL_ENVS:
        logger.info(f"Performance tip: Parallel training should utilize {n_envs} CPU cores for faster data collection")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = model_save_dir / "ppo_risk_agent_final.zip"
        model.save(str(final_model_path))
        logger.info(f"\nFinal model saved: {final_model_path}")
        
        # Save VecNormalize statistics
        vec_normalize_path = model_save_dir / "vec_normalize.pkl"
        train_env.save(str(vec_normalize_path))
        logger.info(f"VecNormalize statistics saved: {vec_normalize_path}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        final_model_path = model_save_dir / "ppo_risk_agent_interrupted.zip"
        model.save(str(final_model_path))
        logger.info(f"Model saved: {final_model_path}")
        
        # Save VecNormalize statistics
        vec_normalize_path = model_save_dir / "vec_normalize.pkl"
        train_env.save(str(vec_normalize_path))
        logger.info(f"VecNormalize statistics saved: {vec_normalize_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    logger.info("="*60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {model_save_dir}")
    logger.info(f"TensorBoard logs: {tensorboard_log_dir}")
    logger.info("="*60)
    
    return model


def _evaluate_with_episode_returns(model, vec_env, n_eval_episodes: int, deterministic: bool = True):
    """
    Run evaluation and collect episode_rewards, episode_lengths, episode_returns.
    Win rate is computed from episode_returns (positive balance = win).
    Returns (episode_rewards, episode_lengths, episode_returns).
    Supports RecurrentPPO with LSTM states.
    """
    episode_rewards = []
    episode_lengths = []
    episode_returns = []
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    n_completed = 0
    reset_result = vec_env.reset()
    # Handle different return formats from reset()
    # VecEnv.reset() should return (observations, infos), but handle edge cases
    if isinstance(reset_result, tuple):
        obs = reset_result[0]  # Always take first element as observations
    else:
        obs = reset_result  # If not a tuple, it's just observations
    step_count = np.zeros(vec_env.num_envs, dtype=int)
    total_rewards = np.zeros(vec_env.num_envs)
    lstm_states = None  # LSTM states for RecurrentPPO
    
    while n_completed < n_eval_episodes:
        actions, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=deterministic
        )
        obs, rewards, dones, infos = vec_env.step(actions)
        total_rewards += rewards
        step_count += 1
        episode_starts = dones  # Update episode starts based on dones
        
        for i in range(vec_env.num_envs):
            if dones[i]:
                episode_rewards.append(float(total_rewards[i]))
                episode_lengths.append(int(step_count[i]))
                episode_returns.append(float(infos[i].get("episode_return", 0.0)))
                total_rewards[i] = 0.0
                step_count[i] = 0
                n_completed += 1
        if n_completed >= n_eval_episodes:
            break
    return episode_rewards, episode_lengths, episode_returns


def evaluate_on_test_data(
    model,
    test_tickers_data: Dict[str, Dict[str, pd.Series]],
    vec_normalize_path: Optional[Path] = None
) -> Dict:
    """
    Evaluate the trained model on ALL entry signals in test data.
    
    Args:
        model: Trained PPO model
        test_tickers_data: Test data for each ticker (with entry_signals)
        vec_normalize_path: Path to VecNormalize statistics file (optional)
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUATING ON TEST DATA")
    logger.info("="*60)
    
    if len(test_tickers_data) == 0:
        logger.warning("No test data available for evaluation!")
        return {}
    
    # Count total entry signals in test data
    total_signals = 0
    for ticker, data in test_tickers_data.items():
        if data['entry_signals'] is not None:
            signals = data['entry_signals'].sum()
            total_signals += signals
            logger.info(f"  {ticker}: {signals} entry signals")
    
    if total_signals == 0:
        logger.warning("No entry signals found in test data!")
        return {}
    
    logger.info(f"  Total entry signals to evaluate: {total_signals}")
    
    # Use all entry signals for evaluation
    n_episodes = total_signals
    
    # Create test environment
    test_env_factory = create_env_factory(test_tickers_data, seed=456)
    n_test_envs = min(n_episodes, 32)  # Use up to 32 parallel envs for speed
    test_env = make_vec_env(
        test_env_factory,
        n_envs=n_test_envs,
        vec_env_cls=DummyVecEnv
    )
    
    # Wrap test environment with VecNormalize
    logger.info("Wrapping test environment with VecNormalize...")
    test_env = VecNormalize(
        test_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False  # Evaluation mode (uses fixed statistics)
    )
    
    # Load VecNormalize statistics if provided
    if vec_normalize_path is not None and vec_normalize_path.exists():
        logger.info(f"Loading VecNormalize statistics from: {vec_normalize_path}")
        test_env = VecNormalize.load(str(vec_normalize_path), test_env)
    else:
        logger.warning("VecNormalize statistics not provided - test environment will use default normalization")
    
    # Run evaluation and collect episode returns (for win rate = positive balance)
    logger.info(f"\nRunning evaluation on ALL {n_episodes} entry signals...")
    episode_rewards, episode_lengths, episode_returns = _evaluate_with_episode_returns(
        model, test_env, n_eval_episodes=n_episodes, deterministic=True
    )
    
    # Calculate metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    # Calculate reward per step (normalized)
    rewards_per_step = [r / max(l, 1) for r, l in zip(episode_rewards, episode_lengths)]
    mean_reward_per_step = np.mean(rewards_per_step)
    
    # Win rate = fraction of episodes with positive balance (episode_return > 0)
    win_rate, wins, n_episodes_returns = _win_rate_from_returns(episode_returns)
    
    # Calculate total return proxy (sum of all rewards)
    total_reward = sum(episode_rewards)
    
    # Calculate holding time statistics
    avg_holding_time = np.mean(episode_lengths)
    std_holding_time = np.std(episode_lengths)
    min_holding_time = np.min(episode_lengths)
    max_holding_time = np.max(episode_lengths)
    
    # Log results
    logger.info("\n" + "-"*40)
    logger.info("TEST DATA EVALUATION RESULTS")
    logger.info("-"*40)
    logger.info(f"  Episodes evaluated: {len(episode_rewards)}")
    logger.info(f"  Mean total reward:  {mean_reward:.2f} +/- {std_reward:.2f}")
    logger.info(f"  Mean reward/step:   {mean_reward_per_step:.4f}")
    logger.info(f"  Win rate (positive balance): {win_rate:.1%} ({wins}/{n_episodes_returns})")
    logger.info(f"  Holding time:       avg={avg_holding_time:.1f}, std={std_holding_time:.1f}")
    logger.info(f"                      min={min_holding_time:.0f}, max={max_holding_time:.0f} steps")
    logger.info(f"  Total reward (sum): {total_reward:.2f}")
    logger.info("-"*40)
    
    # Clean up
    test_env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_reward_per_step': mean_reward_per_step,
        'mean_episode_length': mean_length,
        'win_rate': win_rate,
        'total_reward': total_reward,
        'n_episodes': len(episode_rewards),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        # Holding time statistics
        'avg_holding_time': avg_holding_time,
        'std_holding_time': std_holding_time,
        'min_holding_time': min_holding_time,
        'max_holding_time': max_holding_time,
        'episode_returns': episode_returns,
    }


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("RL Risk Management Agent Training")
    logger.info("="*60)
    
    # Log date range configuration
    logger.info("\nDate Range Configuration:")
    logger.info(f"  Training:   {TRAIN_START_DATE or 'start'} to {TRAIN_END_DATE or 'end'}")
    logger.info(f"  Validation: {VAL_START_DATE or 'start'} to {VAL_END_DATE or 'end'}")
    logger.info(f"  Evaluation: {EVAL_START_DATE or 'start'} to {EVAL_END_DATE or 'end'}")
    
    # Load TRAINING data
    logger.info("\n--- Loading TRAINING Data ---")
    train_tickers_data = load_all_tickers_data(
        TICKER_LIST,
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE
    )
    
    if len(train_tickers_data) == 0:
        logger.error("No training data loaded! Please ensure dataset.csv exists and contains ticker data.")
        return
    
    # Print training data statistics
    logger.info(f"\nTraining Data Statistics:")
    for ticker, data in train_tickers_data.items():
        price_len = len(data['price'])
        start_dt = data['price'].index.min()
        end_dt = data['price'].index.max()
        num_signals = data['entry_signals'].sum() if data['entry_signals'] is not None else 0
        logger.info(f"  {ticker}: {price_len} points, {num_signals} signals | {start_dt} to {end_dt}")
    
    total_train_timesteps = sum(len(data['price']) for data in train_tickers_data.values())
    logger.info(f"  Total training timesteps: {total_train_timesteps}")
    
    # Load VALIDATION data
    logger.info("\n--- Loading VALIDATION Data ---")
    val_tickers_data = load_all_tickers_data(
        TICKER_LIST,
        start_date=VAL_START_DATE,
        end_date=VAL_END_DATE
    )
    
    if len(val_tickers_data) == 0:
        logger.warning("No validation data loaded! Evaluation will use training data range.")
        val_tickers_data = train_tickers_data
    else:
        # Print validation data statistics
        logger.info(f"\nValidation Data Statistics:")
        for ticker, data in val_tickers_data.items():
            price_len = len(data['price'])
            start_dt = data['price'].index.min()
            end_dt = data['price'].index.max()
            num_signals = data['entry_signals'].sum() if data['entry_signals'] is not None else 0
            logger.info(f"  {ticker}: {price_len} points, {num_signals} signals | {start_dt} to {end_dt}")
        
        total_val_timesteps = sum(len(data['price']) for data in val_tickers_data.values())
        logger.info(f"  Total validation timesteps: {total_val_timesteps}")
    
    # Load TEST data (for final evaluation after training)
    logger.info("\n--- Loading TEST Data ---")
    test_tickers_data = load_all_tickers_data(
        TICKER_LIST,
        start_date=EVAL_START_DATE,
        end_date=EVAL_END_DATE
    )
    
    if len(test_tickers_data) > 0:
        logger.info(f"\nTest Data Statistics:")
        for ticker, data in test_tickers_data.items():
            price_len = len(data['price'])
            start_dt = data['price'].index.min()
            end_dt = data['price'].index.max()
            num_signals = data['entry_signals'].sum() if data['entry_signals'] is not None else 0
            logger.info(f"  {ticker}: {price_len} points, {num_signals} signals | {start_dt} to {end_dt}")
        
        total_test_timesteps = sum(len(data['price']) for data in test_tickers_data.values())
        logger.info(f"  Total test timesteps: {total_test_timesteps}")
    else:
        logger.warning("No test data available in the specified date range!")
    
    # Train model
    model = train_ppo_agent(
        train_tickers_data,
        MODEL_SAVE_DIR,
        TENSORBOARD_LOG_DIR,
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=None,  # Use N_ENVS from config (auto-detected or manually set)
        val_tickers_data=val_tickers_data  # Pass validation data for evaluation
    )
    
    logger.info("\nTraining pipeline complete!")
    logger.info(f"To view TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    
    # Evaluate on TEST data after training (using ALL entry signals)
    if len(test_tickers_data) > 0 and model is not None:
        vec_normalize_path = MODEL_SAVE_DIR / "vec_normalize.pkl"
        test_results = evaluate_on_test_data(
            model,
            test_tickers_data,
            vec_normalize_path=vec_normalize_path
        )
        
        # Save test results
        if test_results:
            results_file = MODEL_SAVE_DIR / "test_evaluation_results.txt"
            with open(results_file, 'w') as f:
                f.write("TEST DATA EVALUATION RESULTS\n")
                f.write("="*40 + "\n")
                f.write(f"Test date range: {EVAL_START_DATE} to {EVAL_END_DATE}\n")
                f.write(f"Entry signals evaluated: {test_results['n_episodes']} (ALL signals in test period)\n")
                f.write(f"Mean total reward: {test_results['mean_reward']:.2f} +/- {test_results['std_reward']:.2f}\n")
                f.write(f"Mean reward/step: {test_results['mean_reward_per_step']:.4f}\n")
                f.write(f"Win rate: {test_results['win_rate']:.1%}\n")
                f.write(f"\nHolding Time Statistics:\n")
                f.write(f"  Average holding time: {test_results['avg_holding_time']:.1f} steps\n")
                f.write(f"  Std holding time: {test_results['std_holding_time']:.1f} steps\n")
                f.write(f"  Min holding time: {test_results['min_holding_time']:.0f} steps\n")
                f.write(f"  Max holding time: {test_results['max_holding_time']:.0f} steps\n")
                f.write(f"\nTotal reward (sum): {test_results['total_reward']:.2f}\n")
            logger.info(f"\nTest results saved to: {results_file}")
    else:
        logger.warning("Skipping test evaluation (no test data or model not trained)")
    
    logger.info("\n" + "="*60)
    logger.info("ALL DONE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

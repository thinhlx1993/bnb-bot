"""
Train RL Risk Management Agent using Stable-Baselines3 PPO
"""

import os
import sys
from pathlib import Path
import pickle
import multiprocessing
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import torch.nn as nn

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
TRAINING_DATA_DIR = Path("rl_training_episodes")
MODEL_SAVE_DIR = Path("models/rl_agent")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG_DIR = MODEL_SAVE_DIR / "tensorboard_logs"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Ticker list (should match backtest.py)
TICKER_LIST = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "SOLUSDT"]
INITIAL_BALANCE = 1000.0  # Default initial balance

# Training hyperparameters
TOTAL_TIMESTEPS = 5e6  # Total training steps (use early stopping)
LEARNING_RATE = 3e-4  # Initial learning rate
LEARNING_RATE_END = 1e-5  # Final learning rate (for linear decay)
USE_LR_SCHEDULE = True  # Enable learning rate scheduling
BATCH_SIZE = 256  # Batch size for stable training
N_STEPS = 2048  # Steps per update
N_EPOCHS = 4  # Optimization epochs per update (further reduced to prevent overfitting)
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda
ENT_COEF = 0.01  # Entropy coefficient (exploration) - increased to prevent premature convergence
VF_COEF = 0.25  # Value function coefficient (further reduced to prevent value loss from dominating)
MAX_GRAD_NORM = 0.5  # Maximum gradient norm for clipping
CLIP_RANGE = 0.2  # PPO clip range (standard value, keeps policy updates conservative)

# Reward normalization settings
USE_VEC_NORMALIZE = True  # Enable reward and observation normalization for stable training

# Model architecture configuration
POLICY_LAYERS = [256, 256]  # Policy network hidden layers
VALUE_LAYERS = [256, 256]   # Value network hidden layers
ACTIVATION_FN = 'tanh'  # Activation function: 'tanh', 'relu', or 'elu'
CHECKPOINT_FREQ = 10000  # Save checkpoint every N steps
EVAL_FREQ = 10000  # Evaluate every N steps (single run: all val episodes, N_VAL_WORKERS)
N_VAL_WORKERS = 16  # Workers: episodes split across workers; use fewer (e.g. 8) if EVAL_WORKER_DEVICE='cuda'
EVAL_WORKER_DEVICE = 'cuda'  # Device in each worker: 'cpu' or 'cuda' (spawn loads checkpoint per worker)
EVAL_EPISODES_FALLBACK = 50  # Episodes when no val_episodes (e.g. val_tickers_data only)

# Early stopping configuration
ENABLE_EARLY_STOPPING = False  # Enable early stopping
EARLY_STOPPING_PATIENCE = 50  # Number of evaluations without improvement before stopping
EARLY_STOPPING_MIN_DELTA = 0.0  # Minimum change to qualify as improvement
EARLY_STOPPING_MONITOR = 'mean_total_reward'  # Metric to monitor: 'mean_reward', 'mean_total_reward', 'mean_ep_length', or 'loss'
EARLY_STOPPING_MODE = 'max'  # 'max' for reward/ep_length (higher is better), 'min' for loss (lower is better)

# Training/Validation split (used if date ranges not specified)
TRAIN_SPLIT = 0.7  # 70% training, 30% validation

# ============== Date Range Configuration ==============
# Set to None to use all available data, or specify date range (YYYY-MM-DD format)
# Time-series split: TRAIN -> VALIDATION -> EVAL (chronological order)

# Training data date range  # YYYY-MM-DD format, or None for all data
TRAIN_START_DATE = "2017-01-01"  # Start date for training data (None = beginning of data)
TRAIN_END_DATE = "2024-01-01"    # End date for training data (None = use TRAIN_SPLIT)

# Validation data date range (for evaluation during training)
VAL_START_DATE = "2024-01-01"    # Start date for validation data (None = after TRAIN_END_DATE)
VAL_END_DATE = "2024-12-31"      # End date for validation data (None = end of data)

# Evaluation data date range (for final evaluation after training)
EVAL_START_DATE = "2025-01-01"   # Start date for evaluation (None = use VAL dates)
EVAL_END_DATE = "2026-01-24"              # End date for evaluation (None = end of data)

# Technical indicator settings for entry signal generation
USE_TECHNICAL_SIGNALS = True  # Use technical signals for entry points (if False, random entry)
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


def _compute_eval_metrics(episode_rewards: List[float], episode_lengths: List[int]) -> Dict[str, float]:
    """Compute mean/std reward, win rate, holding time from episode results."""
    if episode_rewards and episode_lengths:
        rewards_per_step = [r / max(l, 1) for r, l in zip(episode_rewards, episode_lengths)]
        mean_reward = np.mean(rewards_per_step)
        std_reward = np.std(rewards_per_step)
        mean_total_reward = np.mean(episode_rewards)
        std_total_reward = np.std(episode_rewards)
        wins = sum(1 for r in episode_rewards if r > 0)
        win_rate = wins / len(episode_rewards)
        avg_holding_time = np.mean(episode_lengths)
        std_holding_time = np.std(episode_lengths)
        min_holding_time = np.min(episode_lengths)
        max_holding_time = np.max(episode_lengths)
        mean_ep_length = np.mean(episode_lengths)
    else:
        mean_reward = std_reward = mean_total_reward = std_total_reward = 0.0
        win_rate = avg_holding_time = std_holding_time = min_holding_time = max_holding_time = mean_ep_length = 0.0
    return {
        "mean_reward": mean_reward, "std_reward": std_reward,
        "mean_total_reward": mean_total_reward, "std_total_reward": std_total_reward,
        "win_rate": win_rate, "mean_ep_length": mean_ep_length,
        "avg_holding_time": avg_holding_time, "std_holding_time": std_holding_time,
        "min_holding_time": min_holding_time, "max_holding_time": max_holding_time,
    }


class DynamicEvalCallback(BaseCallback):
    """
    Single evaluation every eval_freq: all validation episodes split across n_val_workers.
    One run for logging, early stopping, and best-model selection.
    """
    
    def __init__(
        self,
        val_episodes: List[Dict],
        all_tickers_data: Dict[str, Dict[str, pd.Series]],
        eval_freq: int,
        model_save_dir: Path,
        log_path: Path,
        deterministic: bool = True,
        verbose: int = 1,
        n_val_workers: int = 64,
        n_eval_episodes_fallback: int = 50,
        eval_worker_device: str = 'cpu',
    ):
        super().__init__(verbose=verbose)
        self.val_episodes = val_episodes
        self.all_tickers_data = all_tickers_data
        self.eval_freq = eval_freq
        self.model_save_dir = model_save_dir
        self.log_path = log_path
        self.deterministic = deterministic
        self.n_val_workers = n_val_workers
        self.n_eval_episodes_fallback = n_eval_episodes_fallback
        self.eval_worker_device = eval_worker_device
        self.last_mean_reward = None
        self.last_mean_total_reward = None
        self.last_mean_ep_length = None
        self.last_win_rate = None
        self.last_avg_holding_time = None
        self.best_mean_reward = float('-inf')
        self.eval_count = 0
        
        log_path.mkdir(parents=True, exist_ok=True)
    
    def _run_eval_fallback(self, n_episodes: int, seed_offset: int) -> Tuple[List[float], List[int]]:
        """Fallback when no val_episodes: run evaluate_policy with all_tickers_data."""
        n_envs = min(16, n_episodes)
        factory = create_env_factory(self.all_tickers_data, seed=123 + seed_offset)
        env = make_vec_env(factory, n_envs=n_envs, vec_env_cls=DummyVecEnv)
        try:
            rewards, lengths = evaluate_policy(
                self.model,
                env,
                n_eval_episodes=n_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )
            return rewards, lengths
        finally:
            env.close()
    
    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        
        # Single evaluation: all val episodes (64 workers) or fallback (no val_episodes)
        if len(self.val_episodes) > 0:
            n_full = len(self.val_episodes)
            n_workers = max(1, min(self.n_val_workers, n_full))
            if self.verbose >= 1:
                logger.info(f"Eval num_timesteps={self.num_timesteps} ({n_full} episodes, {n_workers} workers, device={self.eval_worker_device})...")
            # Checkpoint for workers: each worker loads this and runs its chunk (spawn avoids CUDA fork issues)
            temp_path = self.model_save_dir / "_eval_temp.zip"
            self.model.save(str(temp_path))
            try:
                chunks = np.array_split(self.val_episodes, n_workers)
                arg_tuples = [(list(chunk), str(temp_path), self.deterministic, self.eval_worker_device) for chunk in chunks]
                ctx = multiprocessing.get_context('spawn')
                with ctx.Pool(n_workers) as pool:
                    results = pool.map(_parallel_eval_worker, arg_tuples)
                all_rewards = [r for res in results for r in res[0]]
                all_lengths = [l for res in results for l in res[1]]
            finally:
                if temp_path.exists():
                    temp_path.unlink()
            episode_rewards, episode_lengths = all_rewards, all_lengths
        else:
            episode_rewards, episode_lengths = self._run_eval_fallback(
                self.n_eval_episodes_fallback, self.eval_count
            )
        
        metrics = _compute_eval_metrics(episode_rewards, episode_lengths)
        self.eval_count += 1
        
        self.last_mean_reward = metrics["mean_reward"]
        self.last_mean_total_reward = metrics["mean_total_reward"]
        self.last_mean_ep_length = metrics["mean_ep_length"]
        self.last_win_rate = metrics["win_rate"]
        self.last_avg_holding_time = metrics["avg_holding_time"]
        
        # Log to TensorBoard
        self.logger.record("eval/mean_reward", metrics["mean_reward"])
        self.logger.record("eval/mean_ep_length", metrics["mean_ep_length"])
        self.logger.record("eval/std_reward", metrics["std_reward"])
        self.logger.record("eval/mean_total_reward", metrics["mean_total_reward"])
        self.logger.record("eval/std_total_reward", metrics["std_total_reward"])
        self.logger.record("eval/win_rate", metrics["win_rate"])
        self.logger.record("eval/avg_holding_time", metrics["avg_holding_time"])
        self.logger.record("eval/std_holding_time", metrics["std_holding_time"])
        self.logger.record("eval/min_holding_time", metrics["min_holding_time"])
        self.logger.record("eval/max_holding_time", metrics["max_holding_time"])
        self.logger.dump(step=self.num_timesteps)
        
        if self.verbose >= 1:
            logger.info(
                f"Eval num_timesteps={self.num_timesteps}, "
                f"episode_reward_per_step={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f} "
                f"(total: {metrics['mean_total_reward']:.2f} +/- {metrics['std_total_reward']:.2f})"
            )
            logger.info(
                f"Episode length: {metrics['mean_ep_length']:.2f}, Win rate: {metrics['win_rate']:.1%}, "
                f"Holding: avg={metrics['avg_holding_time']:.1f}, min={metrics['min_holding_time']:.0f}, max={metrics['max_holding_time']:.0f}"
            )
        
        # Best model from same run
        if metrics["mean_total_reward"] > self.best_mean_reward:
            self.best_mean_reward = metrics["mean_total_reward"]
            self.model.save(str(self.model_save_dir / "best_model.zip"))
            if self.verbose >= 1:
                logger.info(f"New best model saved (mean total reward: {self.best_mean_reward:.2f})")
        
        return True


class EarlyStoppingCallback(BaseCallback):
    """
    Custom early stopping callback that monitors validation loss/metrics.
    Stops training if the monitored metric doesn't improve for a specified number of evaluations.
    """
    
    def __init__(
        self,
        eval_callback,  # Can be EvalCallback or DynamicEvalCallback
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = EARLY_STOPPING_MIN_DELTA,
        monitor: str = EARLY_STOPPING_MONITOR,
        mode: str = EARLY_STOPPING_MODE,
        verbose: int = 1
    ):
        """
        Initialize early stopping callback.
        
        Args:
            eval_callback: EvalCallback or DynamicEvalCallback instance to monitor
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


def load_all_tickers_data(
    tickers_list: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    results_dir: Path = Path("results"),
    strategy: str = "Combined"
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Load price, balance, and entry signal data for all tickers.
    
    Args:
        tickers_list: List of ticker symbols
        start_date: Start date for filtering (YYYY-MM-DD format, or None)
        end_date: End date for filtering (YYYY-MM-DD format, or None)
        results_dir: Results directory containing balance data
        strategy: Strategy name for balance data
    
    Returns:
        Dict of {ticker: {'price': Series, 'balance': Series, 'entry_signals': Series}}
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
                        
                        # Generate entry signals from technical indicators
                        entry_signals = None
                        if USE_TECHNICAL_SIGNALS:
                            entry_signals = generate_entry_signals(ticker_df, price_series)
                            num_signals = entry_signals.sum()
                            logger.info(f"  {ticker}: {len(price_series)} price points, {num_signals} entry signals")
                        else:
                            logger.info(f"  {ticker}: {len(price_series)} price points (random entry)")
                        
                        all_tickers_data[ticker] = {
                            'price': price_series,
                            'balance': balance_series,
                            'entry_signals': entry_signals
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


def load_episodes(data_dir: Path) -> List[Dict]:
    """
    Load training episodes from disk.
    
    Args:
        data_dir: Directory containing episodes.pkl
    
    Returns:
        List of episode dictionaries
    """
    logger.info(f"Loading episodes from {data_dir}...")
    
    episodes_file = data_dir / "episodes.pkl"
    if not episodes_file.exists():
        logger.error(f"Episodes file not found: {episodes_file}")
        logger.error("Please run prepare_rl_training_data.py first!")
        return []
    
    with open(episodes_file, 'rb') as f:
        episodes = pickle.load(f)
    
    logger.info(f"Loaded {len(episodes)} episodes")
    return episodes


def split_episodes(episodes: List[Dict], train_split: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """
    Split episodes into training and validation sets.
    
    Args:
        episodes: List of all episodes
        train_split: Proportion for training (rest for validation)
    
    Returns:
        train_episodes, val_episodes
    """
    # Simple random split (could also do time-based split)
    np.random.seed(42)
    indices = np.random.permutation(len(episodes))
    split_idx = int(len(episodes) * train_split)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_episodes = [episodes[i] for i in train_indices]
    val_episodes = [episodes[i] for i in val_indices]
    
    logger.info(f"Split episodes: {len(train_episodes)} training, {len(val_episodes)} validation")
    
    return train_episodes, val_episodes


def make_env(episode: Dict, rank: int = 0, seed: int = 0) -> RiskManagementEnv:
    """
    Create a single environment instance.
    
    Args:
        episode: Episode dictionary with price_data, balance_data, etc.
        rank: Environment rank (for vectorization)
        seed: Random seed
    
    Returns:
        Environment instance
    """
    def _init():
        env = RiskManagementEnv(
            price_data=episode['price_series'],
            balance_data=episode['balance_series'],
            entry_price=episode['entry_price'],
            entry_idx=episode['entry_idx'] - episode['entry_idx'],  # Relative to episode start
            exit_idx=episode['exit_idx'] - episode['entry_idx'],
            initial_balance=episode['initial_balance'],
            history_length=60,
            max_steps=min(episode['episode_length'] + 100, 5000),
            fee_rate=0.001
        )
        # Wrap with Monitor for statistics
        env = Monitor(env, filename=None, allow_early_resets=True)
        env.reset(seed=seed + rank)
        return env
    return _init


def _parallel_eval_worker(args: Tuple) -> Tuple[List[float], List[int]]:
    """
    Run evaluation on a chunk of episodes; used by multiprocessing.Pool (spawn).
    At validation start a checkpoint is saved; each worker loads that checkpoint and runs its chunk.
    args: (chunk_episodes, model_path, deterministic) or (..., device). device defaults to 'cpu'.
    """
    if len(args) == 4:
        chunk_episodes, model_path, deterministic, device = args
    else:
        chunk_episodes, model_path, deterministic = args
        device = 'cpu'
    if not chunk_episodes:
        return [], []
    model = PPO.load(model_path, device=device)
    rewards, lengths = [], []
    for ep in chunk_episodes:
        env = make_env(ep, rank=0, seed=0)()
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = term or trunc
        rewards.append(total_reward)
        lengths.append(steps)
    return rewards, lengths


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
        env = RiskManagementEnv(
            all_tickers_data=all_tickers_data,
            initial_balance=INITIAL_BALANCE,
            history_length=60,
            max_steps=5000,
            fee_rate=0.001
        )
        # Wrap with Monitor for statistics
        env = Monitor(env, filename=None, allow_early_resets=True)
        env.reset(seed=seed + rank)
        return env
    
    return _make_env


def create_eval_env_factory(val_episodes: List[Dict], seed: int = 42) -> callable:
    """
    Create evaluation environment factory that randomly samples episodes.
    
    Args:
        val_episodes: List of validation episodes
        seed: Random seed
    
    Returns:
        Function that returns an environment
    """
    # Store episodes list for random sampling
    episodes_list = val_episodes
    
    def _make_env(rank: int = 0):
        # Create a new random state for each environment creation
        # This ensures different episodes are selected
        local_rng = np.random.RandomState(seed + rank + int(np.random.random() * 1000000))
        # Randomly sample a validation episode
        episode_idx = local_rng.randint(0, len(episodes_list))
        episode = episodes_list[episode_idx]
        # Use different seed for each environment instance
        env_seed = seed + rank + local_rng.randint(0, 10000)
        env = make_env(episode, rank=rank, seed=env_seed)()
        # Store episode list in env for later random resets
        env._available_episodes = episodes_list
        env._episode_rng = np.random.RandomState(env_seed)
        return env
    
    return _make_env


def train_ppo_agent(
    all_tickers_data: Dict[str, Dict[str, pd.Series]],
    val_episodes: List[Dict],
    model_save_dir: Path,
    tensorboard_log_dir: Path,
    total_timesteps: int = TOTAL_TIMESTEPS,
    n_envs: int = 32,
    val_tickers_data: Optional[Dict[str, Dict[str, pd.Series]]] = None
):
    """
    Train PPO agent on all tickers' data.
    
    Args:
        all_tickers_data: Dict of {ticker: {'price': Series, 'balance': Series}} for training
        val_episodes: Validation episodes (for evaluation, legacy)
        model_save_dir: Directory to save models
        tensorboard_log_dir: Directory for TensorBoard logs
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        val_tickers_data: Dict of {ticker: {'price': Series, 'balance': Series}} for validation/evaluation
    """
    # Use validation tickers data if provided, otherwise use training data
    if val_tickers_data is None:
        val_tickers_data = all_tickers_data
    logger.info("="*60)
    logger.info("Training PPO Agent")
    logger.info("="*60)
    
    # Create vectorized environments
    logger.info(f"Creating {n_envs} parallel environments...")
    train_env_factory = create_env_factory(all_tickers_data, seed=42)
    train_env = make_vec_env(
        train_env_factory,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv  # Use DummyVecEnv for simplicity
    )
    
    # Apply reward and observation normalization for stable training
    if USE_VEC_NORMALIZE:
        logger.info("Applying VecNormalize for reward and observation normalization...")
        train_env = VecNormalize(
            train_env,
            norm_obs=False,  # Don't normalize observations (they're already normalized in env)
            norm_reward=True,  # Normalize rewards (critical for stable value function learning)
            clip_obs=10.0,  # Clip observations to prevent extreme values
            clip_reward=5.0,  # Reduced clip range for rewards (rewards are already scaled down)
            gamma=GAMMA  # Use same gamma for reward normalization
        )
        logger.info("VecNormalize enabled: rewards will be normalized (observations already normalized in env)")
    
    # Create evaluation environment
    # Use validation episodes if available, otherwise use val_tickers_data with technical signals
    if len(val_episodes) > 0:
        eval_env_factory = create_eval_env_factory(val_episodes, seed=123)
        n_eval_envs = min(EVAL_EPISODES_FALLBACK, len(val_episodes), 10)
        logger.info(f"Evaluation using {len(val_episodes)} validation episodes ({N_VAL_WORKERS} workers)")
    else:
        eval_env_factory = create_env_factory(val_tickers_data, seed=123)
        n_eval_envs = min(EVAL_EPISODES_FALLBACK, 10)
        logger.info(f"Evaluation using validation data ({VAL_START_DATE} to {VAL_END_DATE})")
    
    eval_env = make_vec_env(
        eval_env_factory,
        n_envs=n_eval_envs,
        vec_env_cls=DummyVecEnv
    )
    logger.info("Eval callback will run full validation (all episodes, parallel workers) every eval")
    
    # Create PPO model with more complex architecture
    logger.info("Creating PPO model with complex architecture...")
    
    # Map activation function string to PyTorch activation
    activation_map = {
        'tanh': torch.nn.Tanh,
        'relu': torch.nn.ReLU,
        'elu': torch.nn.ELU,
        'leaky_relu': torch.nn.LeakyReLU
    }
    activation_fn = activation_map.get(ACTIVATION_FN.lower(), torch.nn.Tanh)
    
    policy_kwargs = dict(
        # Deeper and wider network architecture
        # Separate networks for policy (pi) and value function (vf)
        # Policy network: Multiple hidden layers with decreasing width (bottleneck design)
        # Value network: Multiple hidden layers with decreasing width
        net_arch=[
            dict(pi=POLICY_LAYERS,  # Policy network layers
                 vf=VALUE_LAYERS)   # Value network layers
        ],
        # Use activation function
        activation_fn=activation_fn
    )
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU (consider using GPU for faster training with complex architecture)")
    
    # Set up learning rate (constant or scheduled)
    if USE_LR_SCHEDULE:
        lr_schedule = linear_schedule(LEARNING_RATE, LEARNING_RATE_END)
        logger.info(f"Using linear LR schedule: {LEARNING_RATE} -> {LEARNING_RATE_END}")
    else:
        lr_schedule = LEARNING_RATE
        logger.info(f"Using constant LR: {LEARNING_RATE}")
    
    model = PPO(
        "MlpPolicy",
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
        clip_range=CLIP_RANGE,  # Explicit clip range for conservative policy updates
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_log_dir),
        verbose=1,
        device=device
    )
    
    logger.info(f"Model architecture:")
    logger.info(f"  Policy network: {POLICY_LAYERS} ({len(POLICY_LAYERS)} hidden layers)")
    logger.info(f"  Value network: {VALUE_LAYERS} ({len(VALUE_LAYERS)} hidden layers)")
    logger.info(f"  Activation function: {ACTIVATION_FN}")
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
    logger.info(f"  Reward normalization: {USE_VEC_NORMALIZE}")
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(model_save_dir / "checkpoints"),
        name_prefix="ppo_risk_agent",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback - use custom callback that recreates environments each time
    eval_callback = DynamicEvalCallback(
        val_episodes=val_episodes,
        all_tickers_data=val_tickers_data,
        eval_freq=EVAL_FREQ,
        model_save_dir=model_save_dir,
        log_path=model_save_dir / "eval_logs",
        deterministic=True,
        verbose=1,
        n_val_workers=N_VAL_WORKERS,
        n_eval_episodes_fallback=EVAL_EPISODES_FALLBACK,
        eval_worker_device=EVAL_WORKER_DEVICE,
    )
    callbacks.append(eval_callback)
    
    # Close the pre-created eval_env since we'll recreate it each time
    eval_env.close()
    
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
    logger.info(f"\nStarting training for {total_timesteps} timesteps...")
    logger.info(f"Checkpoints will be saved every {CHECKPOINT_FREQ} steps")
    logger.info(f"Evaluation every {EVAL_FREQ} steps")
    logger.info(f"TensorBoard logs: {tensorboard_log_dir}")
    
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
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        final_model_path = model_save_dir / "ppo_risk_agent_interrupted.zip"
        model.save(str(final_model_path))
        logger.info(f"Model saved: {final_model_path}")
    
    # Cleanup
    train_env.close()
    # Note: eval_env was already closed when we switched to DynamicEvalCallback
    # (it creates its own environments each evaluation)
    
    logger.info("="*60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {model_save_dir}")
    logger.info(f"TensorBoard logs: {tensorboard_log_dir}")
    logger.info("="*60)
    
    return model


def evaluate_on_test_data(
    model,
    test_tickers_data: Dict[str, Dict[str, pd.Series]]
) -> Dict:
    """
    Evaluate the trained model on ALL entry signals in test data.
    
    Args:
        model: Trained PPO model
        test_tickers_data: Test data for each ticker (with entry_signals)
    
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
    
    # Run evaluation on ALL entry signals
    logger.info(f"\nRunning evaluation on ALL {n_episodes} entry signals...")
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        return_episode_rewards=True
    )
    
    # Calculate metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    # Calculate reward per step (normalized)
    rewards_per_step = [r / max(l, 1) for r, l in zip(episode_rewards, episode_lengths)]
    mean_reward_per_step = np.mean(rewards_per_step)
    
    # Calculate win rate (positive reward = win)
    wins = sum(1 for r in episode_rewards if r > 0)
    win_rate = wins / len(episode_rewards) if episode_rewards else 0
    
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
    logger.info(f"  Win rate:           {win_rate:.1%} ({wins}/{len(episode_rewards)})")
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
        'max_holding_time': max_holding_time
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
    
    # Load validation episodes from file (legacy support)
    val_episodes = []
    episodes = load_episodes(TRAINING_DATA_DIR)
    if len(episodes) > 0:
        _, val_episodes = split_episodes(episodes, TRAIN_SPLIT)
        logger.info(f"\nLoaded {len(val_episodes)} validation episodes from file")
    
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
        val_episodes,
        MODEL_SAVE_DIR,
        TENSORBOARD_LOG_DIR,
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=16,
        val_tickers_data=val_tickers_data  # Pass validation data for evaluation
    )
    
    logger.info("\nTraining pipeline complete!")
    logger.info(f"To view TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    
    # Evaluate on TEST data after training (using ALL entry signals)
    if len(test_tickers_data) > 0 and model is not None:
        test_results = evaluate_on_test_data(
            model,
            test_tickers_data
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

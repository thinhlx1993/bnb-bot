"""
RL Risk Management Integration
Wrapper to integrate trained RL agent with existing backtest system.
Uses RecurrentPPO from sb3_contrib for recurrent policies with LSTM.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import logging
from tqdm import tqdm
from sb3_contrib import RecurrentPPO  # Using RecurrentPPO for recurrent policies
from stable_baselines3.common.vec_env import DummyVecEnv

# Import custom environment
from rl_risk_env import RiskManagementEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_risk_management.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_LOAD_DIR = Path("models/rl_agent")
DEFAULT_MODEL_NAME = "best_model"  # Best model from evaluation, or "ppo_risk_agent_final"


class RLRiskManager:
    """
    RL-based risk management wrapper.
    
    Manages positions using a trained PPO agent to decide when to hold or close.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        initial_balance: float = 100.0,
        history_length: int = 60,
        max_steps: int = 5000,
        fee_rate: float = 0.001,
        deterministic: bool = True
    ):
        """
        Initialize RL risk manager.
        
        Args:
            model_path: Path to model directory or model file
            model_name: Model filename (without .zip extension)
            initial_balance: Initial account balance
            history_length: Number of historical periods in observations
            max_steps: Maximum steps per position
            fee_rate: Trading fee rate
            deterministic: Use deterministic policy (True) or stochastic (False)
        """
        self.initial_balance = initial_balance
        self.history_length = history_length
        self.max_steps = max_steps
        self.fee_rate = fee_rate
        self.deterministic = deterministic
        
        # Load model
        if model_path is None:
            model_path = MODEL_LOAD_DIR
        
        model_file = model_path / f"{model_name}.zip"
        if not model_file.exists():
            # Try alternative names
            for alt_name in ["ppo_risk_agent_final", "best_model", "ppo_risk_agent"]:
                alt_file = model_path / f"{alt_name}.zip"
                if alt_file.exists():
                    model_file = alt_file
                    logger.info(f"Using alternative model: {alt_name}")
                    break
            else:
                # Search for any .zip file in the directory
                zip_files = list(model_path.glob("*.zip"))
                if zip_files:
                    model_file = zip_files[0]
                    logger.warning(f"Using first available model: {model_file.name}")
                else:
                    raise FileNotFoundError(f"Model file not found in {model_path}")
        
        logger.info(f"Loading RL model from: {model_file}")
        try:
            # RecurrentPPO uses PyTorch and will automatically use GPU if available
            self.model = RecurrentPPO.load(str(model_file))
            logger.info(f"RL model loaded successfully (RecurrentPPO - PyTorch)")
            # Get device info
            if hasattr(self.model, 'device'):
                logger.info(f"Using device: {self.model.device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Track open positions
        self.positions = []  # [(entry_idx, entry_price, current_idx, env, peak_balance)]
    
    def _create_env_for_position(
        self,
        price_data: pd.Series,
        balance_data: pd.Series,
        entry_idx: int,
        entry_price: float,
        current_idx: int,
        exit_idx_guess: Optional[int] = None
    ) -> RiskManagementEnv:
        """
        Create environment instance for a position.
        
        Args:
            price_data: Full price series
            balance_data: Full balance series
            entry_idx: Entry index
            entry_price: Entry price
            current_idx: Current index
            exit_idx_guess: Estimated exit index (for episode length)
        
        Returns:
            Environment instance
        """
        # Estimate episode length if not provided
        if exit_idx_guess is None:
            # Use a reasonable default (e.g., 100 periods ahead, or max_steps)
            exit_idx_guess = min(current_idx + self.max_steps, len(price_data) - 1)
        
        # Extract episode window (from entry to estimated exit)
        episode_end = min(exit_idx_guess, len(price_data) - 1)
        price_window = price_data.iloc[entry_idx:episode_end + 1].copy()
        balance_window = balance_data.iloc[entry_idx:episode_end + 1].copy()
        
        # Ensure we have enough history before entry
        data_start = max(0, entry_idx - self.history_length)
        if data_start < entry_idx:
            price_window = pd.concat([
                price_data.iloc[data_start:entry_idx],
                price_window
            ])
            balance_window = pd.concat([
                balance_data.iloc[data_start:entry_idx],
                balance_window
            ])
        
        env = RiskManagementEnv(
            price_data=price_window,
            balance_data=balance_window,
            entry_price=entry_price,
            entry_idx=max(0, entry_idx - data_start),  # Relative to window start
            exit_idx=len(price_window) - 1,  # End of window
            initial_balance=self.initial_balance,
            history_length=self.history_length,
            max_steps=self.max_steps,
            fee_rate=self.fee_rate
        )
        
        return env
    
    def apply_rl_risk_management(
        self,
        entries: pd.Series,
        exits: pd.Series,
        price: pd.Series,
        balance: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Apply RL-based risk management to exits - OPTIMIZED VERSION.
        
        Instead of looping through every time step, this version:
        1. Identifies all entry points upfront
        2. For each entry, creates a mini-environment and runs until exit
        3. Uses proper environment observations for accuracy
        
        Args:
            entries: Entry signals (boolean series)
            exits: Original exit signals (boolean series)
            price: Price series
            balance: Account balance series (if None, will be inferred)
        
        Returns:
            Modified exit signals with RL decisions
        """
        exits_with_rl = exits.copy()
        
        # Create balance series if not provided
        if balance is None:
            balance = pd.Series(self.initial_balance, index=price.index)
        
        # Ensure balance is aligned with price
        if len(balance) != len(price):
            balance = balance.reindex(price.index, method='ffill')
            balance = balance.fillna(self.initial_balance)
        
        # Convert to numpy arrays for faster access
        entries_arr = entries.values
        n_periods = len(price)
        
        # Find all entry indices upfront (MUCH faster than checking each step)
        entry_indices = np.where(entries_arr)[0]
        n_entries = len(entry_indices)
        
        if n_entries == 0:
            logger.info("No entry signals found, returning original exits")
            return exits_with_rl
        
        logger.info(f"Applying RL risk management: {n_entries} entries in {n_periods} periods")
        logger.info(f"Using exit signals for episode boundaries (max_steps={self.max_steps} as safety limit only)")
        
        # Track RL decisions
        rl_exit_count = 0
        rl_hold_count = 0
        rl_early_exits = 0  # Count exits that happen before original exit signal
        rl_at_signal_exits = 0  # Count exits that happen at the same time as exit signal
        
        # Find exit signal indices upfront
        exits_arr = exits.values
        exit_indices = np.where(exits_arr)[0]
        
        # Process each entry point
        pbar = tqdm(entry_indices, desc="Processing entries", unit="entry", mininterval=1.0, maxinterval=5.0)
        for entry_idx in pbar:
            entry_price = float(price.iloc[entry_idx])
            
            # Find next exit signal after this entry
            exit_signal_idx = None
            exits_after_entry = exit_indices[exit_indices > entry_idx]
            if len(exits_after_entry) > 0:
                exit_signal_idx = exits_after_entry[0]
            
            # Determine window end: use exit signal if available, otherwise use end of data
            if exit_signal_idx is not None:
                window_end = exit_signal_idx + 1  # Include exit signal
            else:
                window_end = n_periods  # End of data
            
            # Create a mini-environment for this position
            # Window: from (entry - history) to exit signal (or end of data)
            window_start = max(0, entry_idx - self.history_length)
            window_end = min(window_end, n_periods)
            
            price_window = price.iloc[window_start:window_end].copy()
            balance_window = balance.iloc[window_start:window_end].copy()
            
            # Create exit signals series for this window (if exit signal exists)
            exit_signals_window = None
            if exit_signal_idx is not None:
                exit_signals_window = pd.Series(False, index=price_window.index)
                # Set exit signal at the relative position in window
                exit_idx_in_window = exit_signal_idx - window_start
                if 0 <= exit_idx_in_window < len(exit_signals_window):
                    exit_signals_window.iloc[exit_idx_in_window] = True
            
            # Create environment for this position
            # Use all_tickers_data mode to support exit signals
            all_tickers_data = {
                'single_ticker': {
                    'price': price_window,
                    'balance': balance_window,
                    'entry_signals': pd.Series(False, index=price_window.index),  # Not used in legacy mode
                    'exit_signals': exit_signals_window
                }
            }
            
            env = RiskManagementEnv(
                all_tickers_data=all_tickers_data,
                initial_balance=self.initial_balance,
                history_length=self.history_length,
                max_steps=self.max_steps,  # Not used for termination, but kept for compatibility
                fee_rate=self.fee_rate
            )
            
            # Manually set entry point (since we're using all_tickers_data mode)
            env.current_ticker = 'single_ticker'
            env.price_data = price_window
            env.balance_data = balance_window
            env.exit_signals = exit_signals_window
            env.entry_price = entry_price
            env.entry_idx = entry_idx - window_start  # Relative to window
            env.exit_signal_idx = exit_signal_idx - window_start if exit_signal_idx is not None else None
            
            # Set exit_idx
            if exit_signal_idx is not None:
                env.exit_idx = exit_signal_idx - window_start
            else:
                env.exit_idx = len(price_window) - 1
            
            # Set up episode windows
            env.episode_start_idx = env.entry_idx
            env.episode_end_idx = env.exit_idx
            env.episode_length = env.exit_idx - env.entry_idx + 1
            env.data_start_idx = max(0, env.entry_idx - env.history_length)
            env.price_window = price_window.iloc[env.data_start_idx:env.exit_idx + 1].copy()
            env.balance_window = balance_window.iloc[env.data_start_idx:env.exit_idx + 1].copy()
            
            # Reset episode tracking
            env.current_step = 0
            env.current_idx = 0
            env.position_open = True
            env.peak_balance = self.initial_balance
            env.max_drawdown = 0.0
            env.total_reward = 0.0
            env.episode_return = 0.0
            env.returns_history = []
            env.wins = 0
            env.total_trades = 0
            env.entry_balance = self.initial_balance
            env.episode_max_price = env.entry_price
            env.episode_min_price = env.entry_price
            
            # Pre-compute indicators for performance (avoids recalculating on every step)
            env._precomputed_indicators = env._precompute_indicators()
            
            # Get initial observation
            obs = env._get_observation()
            
            # Run RL agent for this position until it decides to exit or reaches exit signal
            done = False
            position_step = 0
            update_interval = max(1, env.episode_length // 10) if env.episode_length > 0 else 10
            
            # Initialize LSTM states for RecurrentPPO
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            while not done:
                # Query RL agent with LSTM states for RecurrentPPO
                action, lstm_states = self.model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts, 
                    deterministic=self.deterministic
                )
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                position_step += 1
                
                # Update episode_start flag for LSTM state management
                episode_starts = np.array([done], dtype=bool)
                
                # Update progress periodically
                if position_step % update_interval == 0:
                    pbar.set_postfix({
                        'entry': f"{pbar.n+1}/{n_entries}",
                        'steps': position_step,
                        'exits': rl_exit_count,
                        'holds': rl_hold_count
                    })
                
                if action == 1:  # Close action
                    # Calculate actual exit index in original price series
                    exit_idx = entry_idx + position_step
                    if exit_idx < n_periods:
                        # Check if this exit is new (not in original exits)
                        is_new_exit = not exits.iloc[exit_idx]
                        exits_with_rl.iloc[exit_idx] = True
                        
                        # Track if this is an early exit (before exit signal)
                        if exit_signal_idx is not None and exit_idx < exit_signal_idx:
                            rl_early_exits += 1
                        elif exit_signal_idx is not None and exit_idx == exit_signal_idx:
                            rl_at_signal_exits += 1
                    else:
                        exits_with_rl.iloc[n_periods - 1] = True
                    rl_exit_count += 1
                    break
                else:  # Hold
                    rl_hold_count += 1
                
                # Check if we've reached exit signal (episode will terminate via truncated)
                if done and truncated and exit_signal_idx is not None:
                    # Reached exit signal, use it as exit
                    # This means agent held until exit signal (didn't close early)
                    if exit_signal_idx < n_periods:
                        # Check if this exit is new (not in original exits)
                        is_new_exit = not exits.iloc[exit_signal_idx]
                        exits_with_rl.iloc[exit_signal_idx] = True
                        if is_new_exit:
                            # This shouldn't happen often since exit signals come from original exits
                            pass
                    # Don't increment rl_exit_count here - this is the exit signal, not agent's decision
                    break
            
            pbar.set_postfix({
                'entry': f"{pbar.n+1}/{n_entries}",
                'exits': rl_exit_count,
                'holds': rl_hold_count
            })
        
        pbar.close()
        
        # Log summary
        original_exits = exits.sum()
        new_exits = exits_with_rl.sum()
        rl_added_exits = new_exits - original_exits
        
        logger.info(f"  RL Risk Management Summary:")
        logger.info(f"    Entries processed: {n_entries}")
        logger.info(f"    Original exits: {original_exits}")
        logger.info(f"    RL-added exits: {rl_added_exits}")
        logger.info(f"    Total exits: {new_exits}")
        logger.info(f"    RL decisions - Hold: {rl_hold_count}, Close: {rl_exit_count}")
        logger.info(f"    RL early exits (before signal): {rl_early_exits}")
        logger.info(f"    RL exits at signal time: {rl_at_signal_exits}")
        logger.info(f"    RL held until signal: {n_entries - rl_exit_count}")
        
        return exits_with_rl
    


def apply_rl_risk_management(
    entries: pd.Series,
    exits: pd.Series,
    price: pd.Series,
    balance: Optional[pd.Series] = None,
    model_path: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    initial_balance: float = 100.0
) -> pd.Series:
    """
    Convenience function to apply RL risk management.
    
    Args:
        entries: Entry signals
        exits: Original exit signals
        price: Price series
        balance: Account balance series (optional)
        model_path: Path to model
        model_name: Model filename
        initial_balance: Initial account balance
    
    Returns:
        Modified exit signals
    """
    manager = RLRiskManager(
        model_path=model_path,
        model_name=model_name,
        initial_balance=initial_balance
    )
    
    return manager.apply_rl_risk_management(entries, exits, price, balance)


if __name__ == "__main__":
    # Example usage
    logger.info("RL Risk Management Module")
    logger.info("Use apply_rl_risk_management() in your backtest script")
    logger.info("Example:")
    logger.info("  from rl_risk_management import apply_rl_risk_management")
    logger.info("  exits_rl = apply_rl_risk_management(entries, exits, price, balance)")

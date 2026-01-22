"""
RL Risk Management Integration
Wrapper to integrate trained RL agent with existing backtest system.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import logging
from tqdm import tqdm
from stable_baselines3 import PPO
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
        max_steps: int = 1000,
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
            self.model = PPO.load(str(model_file))
            logger.info("RL model loaded successfully")
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
        Apply RL-based risk management to exits using a single continuous environment.
        
        Creates one environment for the entire price series and processes sequentially
        from index 0 to the end, tracking all positions within that environment.
        
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
            # Initialize with initial balance
            balance = pd.Series(self.initial_balance, index=price.index)
            logger.warning("Balance series not provided, using approximation")
        
        # Ensure balance is aligned with price
        if len(balance) != len(price):
            # Reindex balance to match price
            balance = balance.reindex(price.index, method='ffill')
            balance = balance.fillna(self.initial_balance)
        
        logger.info(f"Applying RL risk management to {len(price)} periods with single continuous environment...")
        
        # Create ONE environment for the entire price series
        # We'll update entry_price and current_idx dynamically for each position
        # The environment uses the full price/balance series, so we can point to any position
        dummy_entry_price = price.iloc[0]
        env = RiskManagementEnv(
            price_data=price,  # Full price series
            balance_data=balance,  # Full balance series
            entry_price=dummy_entry_price,  # Will be updated per position
            entry_idx=0,  # Will be updated per position
            exit_idx=len(price) - 1,  # End of series
            initial_balance=self.initial_balance,
            history_length=self.history_length,
            max_steps=self.max_steps,
            fee_rate=self.fee_rate
        )
        # Store reference to full data for dynamic updates
        env.full_price_data = price
        env.full_balance_data = balance
        obs, _ = env.reset()
        
        # Track open positions (entry_idx, entry_price, current_step_in_position)
        open_positions = []  # List of dicts: {'entry_idx': i, 'entry_price': p, 'position_step': s}
        
        # Track RL decisions
        rl_exit_count = 0
        rl_hold_count = 0
        
        # Process each time step sequentially from 0 to end
        pbar = tqdm(range(len(price)), desc="Processing periods", unit="period")
        for i in pbar:
            # Check for new entries
            if entries.iloc[i]:
                # New position opened - track it
                entry_price = price.iloc[i]
                open_positions.append({
                    'entry_idx': i,
                    'entry_price': entry_price,
                    'position_step': 0  # Steps since this position opened
                })
                logger.debug(f"  New position opened at step {i}, price ${entry_price:.2f}")
            
            # Process all open positions using the single environment
            positions_to_remove = []
            for pos in open_positions:
                entry_idx = pos['entry_idx']
                entry_price = pos['entry_price']
                position_step = pos['position_step']
                current_idx_in_price = entry_idx + position_step
                
                # Check if we've reached the end of price series
                if current_idx_in_price >= len(price):
                    # Position should be closed (reached end of data)
                    exits_with_rl.iloc[i] = True
                    positions_to_remove.append(pos)
                    continue
                
                # Update environment to reflect current position
                # Recalculate price_window and balance_window for this position
                data_start = max(0, entry_idx - self.history_length)
                data_end = min(current_idx_in_price + 1, len(price))
                
                env.entry_price = entry_price
                env.entry_idx = entry_idx - data_start  # Relative to window start
                env.current_idx = current_idx_in_price - data_start  # Relative to window start
                env.current_step = position_step
                env.episode_start_idx = entry_idx
                env.data_start_idx = data_start
                
                # Update price and balance windows for this position
                env.price_window = price.iloc[data_start:data_end].copy()
                env.balance_window = balance.iloc[data_start:data_end].copy()
                
                # Update position tracking variables
                current_balance = balance.iloc[current_idx_in_price]
                if current_balance > env.peak_balance:
                    env.peak_balance = current_balance
                
                # Calculate current drawdown
                if env.peak_balance > 0:
                    env.max_drawdown = max(env.max_drawdown, 
                                         (env.peak_balance - current_balance) / env.peak_balance)
                
                # Get observation for current position
                try:
                    obs = env._get_observation()
                    
                    # Query RL agent for action
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    
                    # Execute action
                    if action in [1, 2]:  # Close actions
                        exits_with_rl.iloc[current_idx_in_price] = True
                        positions_to_remove.append(pos)
                        rl_exit_count += 1
                        logger.debug(f"  RL agent closed position at step {current_idx_in_price} (action={action})")
                    else:  # Hold (action=0)
                        rl_hold_count += 1
                        # Increment position step for next iteration
                        pos['position_step'] += 1
                    
                except Exception as e:
                    logger.warning(f"  Error processing position at step {current_idx_in_price}: {e}")
                    continue
            
            # Remove closed positions
            for pos in positions_to_remove:
                if pos in open_positions:
                    open_positions.remove(pos)
            
            # Update progress bar with current stats
            num_open_positions = len(open_positions)
            pbar.set_postfix({
                'open_pos': num_open_positions,
                'exits': rl_exit_count,
                'holds': rl_hold_count
            })
        
        pbar.close()
        
        # Log summary
        original_exits = exits.sum()
        new_exits = exits_with_rl.sum()
        rl_added_exits = new_exits - original_exits
        
        logger.info(f"  RL Risk Management Summary:")
        logger.info(f"    Original exits: {original_exits}")
        logger.info(f"    RL-added exits: {rl_added_exits}")
        logger.info(f"    Total exits: {new_exits}")
        logger.info(f"    RL decisions - Hold: {rl_hold_count}, Close: {rl_exit_count}")
        
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

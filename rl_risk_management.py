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
        history_length: int = 30,
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
        Apply RL-based risk management to exits.
        
        This replaces or augments the rule-based apply_risk_management function.
        
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
            # Simple approximation: balance changes with price changes
            # In production, you'd track actual balance from portfolio
            logger.warning("Balance series not provided, using approximation")
        
        # Ensure balance is aligned with price
        if len(balance) != len(price):
            # Reindex balance to match price
            balance = balance.reindex(price.index, method='ffill')
            balance = balance.fillna(self.initial_balance)
        
        logger.info(f"Applying RL risk management to {len(price)} periods...")
        
        # Track positions and RL decisions
        rl_exit_count = 0
        rl_hold_count = 0
        
        # Process each time step
        for i in range(len(price)):
            # Check for new entries
            if entries.iloc[i]:
                # New position opened
                entry_price = price.iloc[i]
                current_balance = balance.iloc[i]
                
                # Create environment for this position
                try:
                    env = self._create_env_for_position(
                        price,
                        balance,
                        i,
                        entry_price,
                        i
                    )
                    env.reset()
                    
                    # Track position
                    self.positions.append({
                        'entry_idx': i,
                        'entry_price': entry_price,
                        'current_idx': i,
                        'env': env,
                        'peak_balance': current_balance,
                        'closed': False
                    })
                    logger.debug(f"  New position opened at step {i}, price ${entry_price:.2f}")
                except Exception as e:
                    logger.warning(f"  Error creating environment for new position at step {i}: {e}")
                    continue
            
            # Process open positions
            positions_to_remove = []
            for pos_idx, position in enumerate(self.positions):
                if position['closed']:
                    continue
                
                entry_idx = position['entry_idx']
                entry_price = position['entry_price']
                env = position['env']
                current_idx = i
                
                # Update position tracking
                position['current_idx'] = current_idx
                current_balance = balance.iloc[i]
                if current_balance > position['peak_balance']:
                    position['peak_balance'] = current_balance
                
                # Get observation from environment
                try:
                    # Recreate environment at current state
                    # We need to recreate because we can't directly set state
                    env = self._create_env_for_position(
                        price,
                        balance,
                        entry_idx,
                        entry_price,
                        current_idx,
                        exit_idx_guess=min(current_idx + self.max_steps, len(price) - 1)
                    )
                    obs, _ = env.reset()
                    
                    # Step environment to current position
                    steps_needed = current_idx - entry_idx
                    for step in range(min(steps_needed, self.max_steps)):
                        if step < steps_needed:
                            obs, _, terminated, truncated, _ = env.step(0)  # Hold action
                            if terminated or truncated:
                                break
                    
                    # Get current observation
                    obs = env._get_observation()
                    
                    # Update position environment
                    position['env'] = env
                    
                    # Query RL agent for action
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    
                    # Execute action
                    if action in [1, 2]:  # Close actions
                        exits_with_rl.iloc[i] = True
                        position['closed'] = True
                        rl_exit_count += 1
                        logger.debug(f"  RL agent closed position at step {i} (action={action})")
                    else:  # Hold (action=0)
                        rl_hold_count += 1
                    
                except Exception as e:
                    logger.warning(f"  Error processing position at step {i}: {e}")
                    # Fall back to original exit signal
                    continue
            
            # Clean up closed positions
            self.positions = [p for p in self.positions if not p['closed']]
        
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

"""
Custom Gymnasium Environment for RL Risk Management
This environment manages existing trading positions by deciding when to hold or close them.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RiskManagementEnv(gym.Env):
    """
    Custom Gymnasium environment for position risk management.
    
    The agent decides whether to hold or close an existing position based on:
    - Account balance status
    - Market price changes
    - Position performance history
    - Current drawdown
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        price_data: pd.Series,
        balance_data: pd.Series,
        entry_price: float,
        entry_idx: int,
        exit_idx: int,
        initial_balance: float = 100.0,
        history_length: int = 30,
        max_steps: int = 1000,
        fee_rate: float = 0.001,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the risk management environment.
        
        Args:
            price_data: Price series (pandas Series with datetime index)
            balance_data: Account balance series (aligned with price_data)
            entry_price: Entry price for this position
            entry_idx: Entry index in price_data
            exit_idx: Exit index in price_data (ground truth exit)
            initial_balance: Initial account balance
            history_length: Number of historical periods to include in observations
            max_steps: Maximum steps per episode
            fee_rate: Trading fee rate (e.g., 0.001 for 0.1%)
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.price_data = price_data
        self.balance_data = balance_data
        self.entry_price = entry_price
        self.entry_idx = entry_idx
        self.exit_idx = exit_idx
        self.initial_balance = initial_balance
        self.history_length = history_length
        self.max_steps = max_steps
        self.fee_rate = fee_rate
        self.render_mode = render_mode
        
        # Extract the episode price and balance data (entry_idx to exit_idx)
        self.episode_start_idx = entry_idx
        self.episode_end_idx = exit_idx
        self.episode_length = exit_idx - entry_idx + 1
        
        # Ensure we have enough data before entry for history
        self.data_start_idx = max(0, entry_idx - history_length)
        self.price_window = price_data.iloc[self.data_start_idx:exit_idx + 1].copy()
        self.balance_window = balance_data.iloc[self.data_start_idx:exit_idx + 1].copy()
        
        # Action space: 0 = Hold, 1 = Close (take profit), 2 = Close (stop loss)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 67 features
        # 1. Current account balance (normalized)
        # 2. Current price change % (since entry)
        # 3. Position unrealized P&L %
        # 4. Periods held (normalized)
        # 5-34. Price change history (last 30 periods)
        # 35-64. Balance change history (last 30 periods)
        # 65. Current price position relative to entry (normalized)
        # 66. Recent volatility (rolling std of returns)
        # 67. Current drawdown from peak
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(67,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.current_idx = None  # Current index in price_window
        self.position_open = True
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
        # Statistics
        self.total_reward = 0.0
        self.episode_return = 0.0
        
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector from current state.
        
        Returns:
            Observation array of 67 features
        """
        if self.current_idx is None:
            # First step, use entry point
            self.current_idx = 0
            
        # Get actual data index
        data_idx = self.episode_start_idx + self.current_idx
        
        # Ensure we don't go beyond available data
        if data_idx >= len(self.price_data):
            data_idx = len(self.price_data) - 1
        if self.current_idx >= len(self.price_window):
            self.current_idx = len(self.price_window) - 1
            
        # Current price and balance
        current_price = float(self.price_window.iloc[self.current_idx])
        current_balance = float(self.balance_window.iloc[self.current_idx])
        
        # Current price change %
        price_change_pct = (current_price - self.entry_price) / self.entry_price
        
        # Position unrealized P&L % (accounting for fees if we exit now)
        # Net return = (price_change - 2*fee_rate) since we pay fees on entry and exit
        unrealized_pnl_pct = price_change_pct - 2 * self.fee_rate
        
        # Periods held (normalized by max steps)
        periods_held = self.current_idx / max(self.max_steps, 1)
        
        # Price change history (last 30 periods)
        price_history = []
        for i in range(self.history_length):
            hist_idx = self.current_idx - (self.history_length - 1 - i)
            if hist_idx < 0:
                # Pad with entry price change if before episode start
                if self.data_start_idx + hist_idx >= 0:
                    hist_price = float(self.price_data.iloc[self.data_start_idx + hist_idx])
                    price_change = (hist_price - self.entry_price) / self.entry_price
                else:
                    price_change = 0.0  # Before entry
            else:
                hist_price = float(self.price_window.iloc[hist_idx])
                price_change = (hist_price - self.entry_price) / self.entry_price
            price_history.append(price_change)
        
        # Balance change history (last 30 periods)
        balance_history = []
        initial_balance_value = float(self.balance_data.iloc[max(0, self.data_start_idx)])
        for i in range(self.history_length):
            hist_idx = self.current_idx - (self.history_length - 1 - i)
            if hist_idx < 0:
                if self.data_start_idx + hist_idx >= 0:
                    hist_balance = float(self.balance_data.iloc[self.data_start_idx + hist_idx])
                    balance_change = (hist_balance - initial_balance_value) / initial_balance_value
                else:
                    balance_change = 0.0
            else:
                hist_balance = float(self.balance_window.iloc[hist_idx])
                balance_change = (hist_balance - initial_balance_value) / initial_balance_value
            balance_history.append(balance_change)
        
        # Current price position relative to entry (normalized)
        # Use a simple normalization: clip to [-2, 2] range and normalize
        price_relative = np.clip(price_change_pct, -0.5, 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Recent volatility (rolling std of returns over last 30 periods)
        if self.current_idx >= 1:
            recent_prices = [float(self.price_window.iloc[max(0, self.current_idx - i)]) 
                           for i in range(min(self.current_idx + 1, self.history_length))]
            if len(recent_prices) > 1:
                returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = np.std(returns) if len(returns) > 0 else 0.0
            else:
                volatility = 0.0
        else:
            volatility = 0.0
        
        # Current drawdown from peak
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        current_drawdown = (self.peak_balance - current_balance) / self.peak_balance if self.peak_balance > 0 else 0.0
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Normalize account balance (relative to initial balance)
        normalized_balance = (current_balance - self.initial_balance) / self.initial_balance
        
        # Construct observation vector
        observation = np.array([
            normalized_balance,          # 0: Current account balance (normalized)
            price_change_pct,            # 1: Current price change %
            unrealized_pnl_pct,          # 2: Position unrealized P&L %
            periods_held,                # 3: Periods held (normalized)
            *price_history,              # 4-33: Price change history (30 values)
            *balance_history,            # 34-63: Balance change history (30 values)
            price_relative,              # 64: Current price relative to entry
            volatility,                  # 65: Recent volatility
            current_drawdown             # 66: Current drawdown
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, action: int, done: bool) -> float:
        """
        Calculate reward based on action and current state.
        
        Adjusted reward function to encourage holding profitable positions longer:
        - Large positive rewards for profitable exits (scaled by return and holding time)
        - Moderate negative rewards for loss exits
        - Penalty for large drawdowns
        - Reward for holding profitable positions (incentivize patience)
        - Penalty for closing too early on profitable positions
        """
        current_price = float(self.price_window.iloc[self.current_idx])
        price_change_pct = (current_price - self.entry_price) / self.entry_price
        periods_held = self.current_idx + 1
        
        # Net return accounting for fees (if closing)
        if action in [1, 2] or done:
            position_return = price_change_pct - 2 * self.fee_rate
            is_closing = True
        else:
            # Unrealized P&L while holding
            position_return = price_change_pct - self.fee_rate  # Only entry fee so far
            is_closing = False
        
        # Reward components
        return_weight = 200.0  # Increased scale for returns (encourages profit)
        drawdown_weight = 100.0  # Penalty for drawdown
        time_reward_weight = 2.0  # Reward for holding profitable positions longer
        
        # Base reward from position return (scaled by return magnitude)
        reward = position_return * return_weight
        
        # Drawdown penalty (increases with severity)
        reward -= self.max_drawdown * drawdown_weight
        
        # Time-based rewards/penalties
        if is_closing:
            # Closing action: reward/penalty based on return and holding time
            if position_return > 0:
                # Profitable exit: reward increases with holding time (up to a point)
                # Encourage holding profitable positions but not too long
                if periods_held < 10:
                    # Penalty for closing too early on profitable position
                    early_close_penalty = (10 - periods_held) * 5.0
                    reward -= early_close_penalty
                elif periods_held > 50:
                    # Small penalty for holding too long (encourage taking profits)
                    reward -= (periods_held - 50) * 0.5
                else:
                    # Bonus for holding in the sweet spot
                    hold_bonus = periods_held * 0.5
                    reward += hold_bonus
            else:
                # Losing exit: small reward for cutting losses early
                if periods_held < 5:
                    reward += 2.0  # Small bonus for cutting losses quickly
                else:
                    # Penalty for holding losses too long
                    reward -= (periods_held - 5) * 0.3
        else:
            # Holding action: reward for holding profitable positions
            if position_return > 0:
                # Reward increases with unrealized profit and time held (up to a point)
                if periods_held < 50:
                    # Positive reward for holding profitable position
                    hold_reward = position_return * time_reward_weight * min(periods_held, 20)
                    reward += hold_reward
                else:
                    # Small penalty if holding too long without closing
                    reward -= 0.1
            elif position_return < -0.05:  # Loss > 5%
                # Small penalty for holding losing positions
                reward -= abs(position_return) * 50.0
        
        # Action-specific rewards (reduced to encourage patience)
        if action == 1:  # Close (take profit)
            if position_return > 0.02:  # Profit > 2%
                # Bonus for taking good profit
                profit_bonus = min(position_return * 50.0, 30.0)  # Cap at 30
                reward += profit_bonus
            elif position_return > 0:
                # Small profit: less bonus
                reward += position_return * 20.0
            else:
                # Penalty for closing profitable position as loss
                reward -= 30.0
        elif action == 2:  # Close (stop loss)
            if position_return < -0.05:  # Loss > 5%
                # Bonus for cutting large losses early
                reward += 10.0
            elif position_return < 0:
                # Small loss: neutral reward
                reward += position_return * 10.0
            else:
                # Penalty for closing profitable position as stop loss
                reward -= 30.0
        
        # Terminal reward adjustment (only on actual exit)
        if done:
            # Final reward based on actual outcome (more weight on final result)
            if position_return > 0.02:  # Good profit
                final_bonus = position_return * 100.0
                reward += final_bonus
            elif position_return > 0:
                final_bonus = position_return * 50.0
                reward += final_bonus
            elif position_return < -0.1:  # Large loss
                final_penalty = position_return * 150.0
                reward += final_penalty
            else:
                final_penalty = position_return * 100.0
                reward += final_penalty
        
        return float(reward)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_step = 0
        self.current_idx = 0
        self.position_open = True
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.total_reward = 0.0
        self.episode_return = 0.0
        
        observation = self._get_observation()
        info = {
            "episode_length": self.episode_length,
            "entry_price": self.entry_price,
            "entry_idx": self.entry_idx,
            "exit_idx": self.exit_idx
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=Hold, 1=Close take profit, 2=Close stop loss)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Check if position should be closed
        if action in [1, 2]:  # Close actions
            self.position_open = False
        
        # Calculate reward
        reward = self._calculate_reward(action, done=False)
        self.total_reward += reward
        
        # Move to next step
        self.current_step += 1
        self.current_idx += 1
        
        # Check termination conditions
        terminated = not self.position_open  # Closed by action
        truncated = (
            self.current_idx >= self.episode_length - 1 or  # Reached ground truth exit
            self.current_step >= self.max_steps  # Max steps exceeded
        )
        done = terminated or truncated
        
        # Get next observation
        if done:
            # Final step: use exit price
            self.current_idx = self.episode_length - 1
            if not terminated:
                # Force close if truncated
                self.position_open = False
        
        observation = self._get_observation()
        
        # Calculate final return if done
        if done:
            current_price = float(self.price_window.iloc[min(self.current_idx, len(self.price_window) - 1)])
            price_change_pct = (current_price - self.entry_price) / self.entry_price
            self.episode_return = price_change_pct - 2 * self.fee_rate
        
        info = {
            "current_step": self.current_step,
            "position_open": self.position_open,
            "total_reward": self.total_reward,
            "episode_return": self.episode_return,
            "max_drawdown": self.max_drawdown,
            "action": action,
            "periods_held": self.current_idx
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            current_price = float(self.price_window.iloc[self.current_idx])
            current_balance = float(self.balance_window.iloc[self.current_idx])
            price_change = (current_price - self.entry_price) / self.entry_price * 100
            
            print(f"Step: {self.current_step}/{self.episode_length}")
            print(f"Price: ${current_price:.2f} ({price_change:+.2f}%)")
            print(f"Balance: ${current_balance:.2f}")
            print(f"Drawdown: {self.max_drawdown:.2%}")
            print(f"Total Reward: {self.total_reward:.2f}")
            print("-" * 40)

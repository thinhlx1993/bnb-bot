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
        all_tickers_data: Optional[Dict[str, Dict[str, pd.Series]]] = None,
        price_data: Optional[pd.Series] = None,
        balance_data: Optional[pd.Series] = None,
        entry_price: Optional[float] = None,
        entry_idx: Optional[int] = None,
        exit_idx: Optional[int] = None,
        initial_balance: float = 100.0,
        history_length: int = 30,
        max_steps: int = 1000,
        fee_rate: float = 0.001,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the risk management environment.
        
        Args:
            all_tickers_data: Dict of {ticker: {'price': Series, 'balance': Series}} for all tickers
            price_data: Price series (legacy - used if all_tickers_data not provided)
            balance_data: Account balance series (legacy - used if all_tickers_data not provided)
            entry_price: Entry price (legacy - will be set on reset if all_tickers_data provided)
            entry_idx: Entry index (legacy - will be set on reset if all_tickers_data provided)
            exit_idx: Exit index (legacy - will be set on reset if all_tickers_data provided)
            initial_balance: Initial account balance
            history_length: Number of historical periods to include in observations
            max_steps: Maximum steps per episode
            fee_rate: Trading fee rate (e.g., 0.001 for 0.1%)
            render_mode: Rendering mode
        """
        super().__init__()
        
        # New approach: store all tickers' data
        if all_tickers_data is not None:
            self.all_tickers_data = all_tickers_data
            self.tickers_list = list(all_tickers_data.keys())
            self.use_all_tickers = True
            # Will be set on reset
            self.current_ticker = None
            self.price_data = None
            self.balance_data = None
            self.entry_price = None
            self.entry_idx = None
            self.exit_idx = None
        else:
            # Legacy approach: single episode data
            self.all_tickers_data = None
            self.use_all_tickers = False
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
        
        # Initialize episode tracking variables (will be set on reset)
        self.episode_start_idx = None
        self.episode_end_idx = None
        self.episode_length = None
        self.data_start_idx = None
        self.price_window = None
        self.balance_window = None
        
        # If using legacy approach (single episode), set up windows now
        if not self.use_all_tickers:
            self.episode_start_idx = entry_idx
            self.episode_end_idx = exit_idx
            self.episode_length = exit_idx - entry_idx + 1
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
        
        # Statistics for portfolio-level metrics
        self.total_reward = 0.0
        self.episode_return = 0.0
        self.returns_history = []  # Track returns for Sharpe-like calculation
        self.wins = 0  # Track wins for win rate
        self.total_trades = 0  # Track total trades
        self.entry_balance = initial_balance  # Track balance at entry
        
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector from current state.
        
        Returns:
            Observation array of 67 features
        """
        # Ensure environment is initialized (should have been reset)
        if self.price_window is None or self.balance_window is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self.current_idx is None:
            # First step, use entry point
            self.current_idx = 0
            
        # Get actual data index
        if self.episode_start_idx is None:
            self.episode_start_idx = 0
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
        Calculate reward optimized for portfolio-level metrics:
        - Maximize: return, sharpe_ratio, win_rate, annualized_return
        - Minimize: drawdown
        
        Reward components:
        1. Return maximization: Direct reward proportional to position return
        2. Sharpe ratio (risk-adjusted return): Reward consistent returns, penalize volatility
        3. Win rate: Higher reward for profitable exits
        4. Annualized return: Reward higher returns per time period
        5. Drawdown minimization: Penalty for drawdowns
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
        
        # ========== 1. RETURN MAXIMIZATION ==========
        # Direct reward proportional to return (maximize total return)
        # Increased weight to incentivize higher total returns
        # Progressive scaling: larger profits get exponentially more reward
        if position_return > 0:
            # Profitable: exponential scaling to encourage larger profits
            # Base reward + progressive bonus
            return_weight = 10.0  # Base multiplier for all returns
            progressive_bonus = (position_return ** 1.5) * 50.0  # Extra reward for larger profits
            reward = position_return * return_weight + progressive_bonus
        else:
            # Loss: linear penalty (less aggressive than profit reward)
            return_weight = 5.0
            reward = position_return * return_weight
        
        # ========== 2. SHARPE RATIO (Risk-Adjusted Return) ==========
        # Sharpe ratio = (Return - Risk-free) / StdDev
        # We encourage consistent returns (low variance) and penalize volatility
        # Track price return volatility during the position
        if periods_held > 5:
            # Calculate rolling return volatility
            recent_returns = []
            for i in range(max(0, self.current_idx - 4), self.current_idx + 1):
                if i > 0:
                    prev_price = float(self.price_window.iloc[i-1])
                    curr_price = float(self.price_window.iloc[i])
                    ret = (curr_price - prev_price) / prev_price
                    recent_returns.append(ret)
            
            if len(recent_returns) > 1:
                returns_std = np.std(recent_returns)
                # Penalize high volatility (lower Sharpe)
                if returns_std > 0.02:  # High volatility threshold
                    volatility_penalty = (returns_std - 0.02) * 1.0  # Reduced from 10.0
                    reward -= volatility_penalty
                # Reward consistent positive returns (high Sharpe)
                if position_return > 0 and returns_std < 0.01:
                    # Scale consistency bonus with profit size
                    consistency_bonus = 0.5 + position_return * 3.0
                    reward += consistency_bonus
        
        # ========== 3. WIN RATE MAXIMIZATION ==========
        # Higher reward for profitable exits (encourages more wins)
        if is_closing:
            if position_return > 0:
                # Win: Much higher reward for profitable exits with progressive scaling
                # Base win bonus + return-scaled bonus
                base_win_bonus = 2.0  # Base bonus for any profitable trade
                # Progressive scaling: small profits get 5x, large profits get even more
                scaled_bonus = position_return * 15.0
                # Extra bonus for exceptional profits (>5% returns)
                if position_return > 0.05:
                    exceptional_bonus = (position_return - 0.05) * 50.0
                    scaled_bonus += exceptional_bonus
                win_bonus = base_win_bonus + scaled_bonus
                reward += win_bonus
                self.wins += 1
            else:
                # Loss: Smaller penalty (we want to minimize losses but not over-penalize)
                loss_penalty = abs(position_return) * 2.0  # Moderate penalty
                reward -= loss_penalty
            self.total_trades += 1
            # Store return for tracking (used in future episodes if multi-trade)
            self.returns_history.append(position_return)
        
        # ========== 4. ANNUALIZED RETURN MAXIMIZATION ==========
        # Annualized return = (1 + return)^(periods_per_year / periods_held) - 1
        # We want higher returns per time period
        if is_closing and periods_held > 0:
            # Approximate annualized return (assuming ~252 trading periods per year)
            periods_per_year = 252.0
            if position_return > -1.0:  # Avoid log of negative
                annualized_return = (1 + position_return) ** (periods_per_year / periods_held) - 1
                # Reward higher annualized returns - scale with profit
                if annualized_return > 0:
                    # Progressive scaling: higher annualized returns get more reward
                    annualized_bonus = annualized_return * 2.0
                    if annualized_return > 1.0:  # >100% annualized return
                        annualized_bonus += (annualized_return - 1.0) * 5.0
                    reward += annualized_bonus
            # Penalty for holding too long with small returns (inefficient time usage)
            if periods_held > 20 and abs(position_return) < 0.01:
                inefficiency_penalty = (periods_held - 20) * 0.01  # Reduced from 0.1
                reward -= inefficiency_penalty
        
        # ========== 5. DRAWDOWN MINIMIZATION ==========
        # Penalty for drawdowns (minimize max drawdown)
        drawdown_weight = 1.0  # Reduced from 10.0 to prevent high value loss
        reward -= self.max_drawdown * drawdown_weight
        
        # ========== TIME-BASED OPTIMIZATION ==========
        # Balance between holding for profits vs. efficient time usage
        if is_closing:
            # Prevent immediate closure bug
            if periods_held < 3:
                early_close_penalty = (3 - periods_held) * 0.2  # Reduced from 2.0
                reward -= early_close_penalty
        else:
            # Holding action: encourage holding profitable positions, penalize holding losses
            if periods_held < 3:
                # Prevent immediate closure
                reward += 0.03
            elif position_return > 0.01:  # Profitable position
                # Reward holding profitable positions (scaled by profit size)
                if periods_held < 30:
                    # More reward for larger unrealized profits
                    holding_bonus = position_return * 3.0  # Reward proportional to profit
                    reward += holding_bonus
            elif position_return < -0.05:  # Large loss
                # Penalty for holding large losses too long
                if periods_held > 10:
                    reward -= abs(position_return) * 0.3
        
        # ========== ACTION-SPECIFIC GUIDANCE ==========
        if action == 1:  # Close (take profit)
            if periods_held >= 3 and position_return > 0:
                # Reward taking profits - scaled by profit size
                # Base reward + progressive bonus for larger profits
                profit_action_bonus = position_return * 8.0
                if position_return > 0.03:  # Good profit (>3%)
                    profit_action_bonus += (position_return - 0.03) * 20.0
                reward += profit_action_bonus
            elif periods_held < 3:
                reward -= 0.1
        elif action == 2:  # Close (stop loss)
            if periods_held >= 5 and position_return < -0.03:
                # Reward cutting losses at reasonable time
                reward += 0.1  # Reduced from 1.0
            elif periods_held < 3:
                reward -= 0.05  # Reduced from 0.5
        
        # ========== TERMINAL REWARD (Final Outcome) ==========
        if done:
            # Strong final reward based on outcome - much higher for profits
            if position_return > 0.05:  # Excellent profit (>5%)
                # Exponential scaling for large profits
                final_bonus = position_return * 30.0 + (position_return - 0.05) * 100.0
                reward += final_bonus
            elif position_return > 0.02:  # Good profit (>2%)
                final_bonus = position_return * 20.0
                reward += final_bonus
            elif position_return > 0:
                # Small profit - still reward it well
                final_bonus = position_return * 10.0
                reward += final_bonus
            elif position_return < -0.1:  # Large loss
                final_penalty = position_return * 5.0  # Moderate penalty
                reward += final_penalty
            else:
                final_penalty = position_return * 3.0  # Moderate penalty
                reward += final_penalty
        
        # Clip reward to prevent extreme values that cause training instability
        # Increased upper bound to allow larger rewards for profits
        # Profits can now get much larger rewards, but losses are still capped
        reward = np.clip(reward, -10.0, 50.0)  # Allow high rewards for large profits
        
        return float(reward)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # If using all tickers, randomly select ticker and entry point
        if self.use_all_tickers:
            # Set random seed if provided
            if seed is not None:
                np.random.seed(seed)
            
            # Randomly select a ticker
            self.current_ticker = np.random.choice(self.tickers_list)
            ticker_data = self.all_tickers_data[self.current_ticker]
            
            # Get price and balance series for this ticker
            price_series = ticker_data['price']
            balance_series = ticker_data['balance']
            
            # Ensure balance series exists and is aligned
            if balance_series is None or len(balance_series) == 0:
                # Create default balance series
                balance_series = pd.Series(self.initial_balance, index=price_series.index)
            
            # Randomly select entry point (must have enough history and future)
            min_entry_idx = self.history_length
            max_entry_idx = len(price_series) - self.max_steps - 1
            
            if max_entry_idx <= min_entry_idx:
                # Not enough data, use available range
                min_entry_idx = 0
                max_entry_idx = max(0, len(price_series) - 10)
            
            if max_entry_idx > min_entry_idx:
                entry_idx = np.random.randint(min_entry_idx, max_entry_idx)
            else:
                entry_idx = min_entry_idx if min_entry_idx < len(price_series) else 0
            
            # Set exit index (end of available data or max_steps ahead)
            exit_idx = min(entry_idx + self.max_steps, len(price_series) - 1)
            
            # Set entry price
            entry_price = float(price_series.iloc[entry_idx])
            
            # Update environment with selected ticker and position
            self.price_data = price_series
            self.balance_data = balance_series
            self.entry_price = entry_price
            self.entry_idx = entry_idx
            self.exit_idx = exit_idx
            
            # Set up episode windows
            self.episode_start_idx = entry_idx
            self.episode_end_idx = exit_idx
            self.episode_length = exit_idx - entry_idx + 1
            self.data_start_idx = max(0, entry_idx - self.history_length)
            self.price_window = price_series.iloc[self.data_start_idx:exit_idx + 1].copy()
            self.balance_window = balance_series.iloc[self.data_start_idx:exit_idx + 1].copy()
        
        # Reset episode tracking
        self.current_step = 0
        self.current_idx = 0
        self.position_open = True
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.total_reward = 0.0
        self.episode_return = 0.0
        
        # Reset portfolio-level metrics
        self.returns_history = []
        self.wins = 0
        self.total_trades = 0
        self.entry_balance = self.initial_balance
        
        observation = self._get_observation()
        info = {
            "episode_length": self.episode_length,
            "entry_price": self.entry_price,
            "entry_idx": self.entry_idx,
            "exit_idx": self.exit_idx
        }
        
        if self.use_all_tickers:
            info["ticker"] = self.current_ticker
        
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

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
        history_length: int = 60,
        max_steps: int = 5000,
        fee_rate: float = 0.001,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the risk management environment.
        
        Args:
            all_tickers_data: Dict of {ticker: {'price': Series, 'balance': Series, 'entry_signals': Series (optional)}}
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
            self.exit_signals = None  # Exit signals for current ticker
            self.entry_price = None
            self.entry_idx = None
            self.exit_idx = None
            self.exit_signal_idx = None  # Next exit signal index after entry
            
            # Pre-calculate entry and exit signal indices for each ticker
            self._entry_signal_indices = {}
            self._exit_signal_indices = {}
            for ticker in self.tickers_list:
                ticker_data = all_tickers_data[ticker]
                
                # Entry signals
                if 'entry_signals' in ticker_data and ticker_data['entry_signals'] is not None:
                    # Get indices where entry signals are True
                    signals = ticker_data['entry_signals']
                    signal_indices = np.where(signals.values)[0]
                    # Filter to valid range (need history before and an exit signal after)
                    # Only include entries that have an exit signal after them
                    exit_signals = ticker_data.get('exit_signals')
                    if exit_signals is not None:
                        exit_indices = np.where(exit_signals.values)[0]
                        # Only keep entry signals that have at least one exit signal after them
                        valid_indices = []
                        for entry_idx in signal_indices:
                            if entry_idx >= history_length:
                                # Check if there's an exit signal after this entry
                                exits_after = exit_indices[exit_indices > entry_idx]
                                if len(exits_after) > 0:
                                    valid_indices.append(entry_idx)
                        self._entry_signal_indices[ticker] = np.array(valid_indices) if len(valid_indices) > 0 else None
                    else:
                        # No exit signals, filter by history only
                        valid_indices = signal_indices[
                            (signal_indices >= history_length) & 
                            (signal_indices < len(signals) - 50)  # At least 50 steps after
                        ]
                        self._entry_signal_indices[ticker] = valid_indices if len(valid_indices) > 0 else None
                else:
                    self._entry_signal_indices[ticker] = None
                
                # Exit signals
                if 'exit_signals' in ticker_data and ticker_data['exit_signals'] is not None:
                    exit_signals = ticker_data['exit_signals']
                    exit_indices = np.where(exit_signals.values)[0]
                    self._exit_signal_indices[ticker] = exit_indices if len(exit_indices) > 0 else None
                else:
                    self._exit_signal_indices[ticker] = None
        else:
            # Legacy approach: single episode data
            self.all_tickers_data = None
            self.use_all_tickers = False
            self._entry_signal_indices = {}
            self._exit_signal_indices = {}
            self.price_data = price_data
            self.balance_data = balance_data
            self.exit_signals = None  # Legacy mode doesn't use exit signals
            self.entry_price = entry_price
            self.entry_idx = entry_idx
            self.exit_idx = exit_idx
            self.exit_signal_idx = None  # Legacy mode doesn't use exit signals
        
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
        
        # Action space: 0 = Hold, 1 = Close (close position)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: 60 timesteps × 6 features = 360 features
        # Features per timestep:
        # 1. MACD (normalized by entry price)
        # 2. MACD Histogram (normalized by entry price)
        # 3. RSI (normalized to [0, 1])
        # 4. EMA 25 (percent-change from entry price)
        # 5. EMA 99 (percent-change from entry price)
        # 6. Close Price (percent-change from entry price)
        # 
        # Normalization strategy:
        # - MACD/Histogram: Normalized by entry price (relative values)
        # - RSI: Normalized to [0, 1] range (RSI/100)
        # - EMA 25/99/Close: Percent-change from entry price (handles future out-of-range prices)
        self.timeseries_length = 60  # Number of historical timesteps
        self.n_features_per_timestep = 6  # MACD, Histogram, RSI, EMA25, EMA99, Close
        obs_shape = (self.timeseries_length * self.n_features_per_timestep,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,  # 60 timesteps × 6 features = 360 features
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
        
        # Track max and min prices during episode
        self.episode_max_price = None
        self.episode_min_price = None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if not enough data
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10  # Avoid division by zero
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi) / 100.0  # Normalize to [0, 1]
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD, Signal, and Histogram."""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
        
        # Convert to pandas Series for easier calculation
        price_series = pd.Series(prices)
        
        # Calculate EMAs
        exp1 = price_series.ewm(span=fast, adjust=False).mean()
        exp2 = price_series.ewm(span=slow, adjust=False).mean()
        macd_line = exp1.iloc[-1] - exp2.iloc[-1]
        
        # Calculate signal line (EMA of MACD)
        macd_series = exp1 - exp2
        signal_line = macd_series.ewm(span=signal, adjust=False).mean().iloc[-1]
        histogram = macd_line - signal_line
        
        # Normalize by entry price for stability
        entry_price = float(self.entry_price)
        macd_normalized = macd_line / entry_price if entry_price > 0 else 0.0
        signal_normalized = signal_line / entry_price if entry_price > 0 else 0.0
        histogram_normalized = histogram / entry_price if entry_price > 0 else 0.0
        
        return float(macd_normalized), float(signal_normalized), float(histogram_normalized)
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average (EMA)."""
        if len(prices) < period:
            # Not enough data, use simple average of available data
            return float(np.mean(prices)) if len(prices) > 0 else float(prices[-1]) if len(prices) > 0 else 0.0
        
        # Convert to pandas Series for easier EMA calculation
        price_series = pd.Series(prices)
        ema = price_series.ewm(span=period, adjust=False).mean().iloc[-1]
        return float(ema)
    
    def _calculate_moving_averages(self, prices: np.ndarray, short_period: int = 5, long_period: int = 20) -> Tuple[float, float, float]:
        """Calculate short and long moving averages."""
        if len(prices) < long_period:
            # Use available data
            short_ma = np.mean(prices[-min(len(prices), short_period):]) if len(prices) >= short_period else float(prices[-1])
            long_ma = np.mean(prices) if len(prices) > 0 else float(prices[-1])
        else:
            short_ma = np.mean(prices[-short_period:])
            long_ma = np.mean(prices[-long_period:])
        
        # Normalize by entry price
        entry_price = float(self.entry_price)
        short_ma_norm = (short_ma - entry_price) / entry_price if entry_price > 0 else 0.0
        long_ma_norm = (long_ma - entry_price) / entry_price if entry_price > 0 else 0.0
        ma_diff = short_ma_norm - long_ma_norm
        
        return float(short_ma_norm), float(long_ma_norm), float(ma_diff)
    
    def _calculate_price_position_in_range(self, prices: np.ndarray, lookback: int = 20) -> float:
        """Calculate price position in recent range (0 = low, 1 = high)."""
        if len(prices) < 2:
            return 0.5
        
        recent_prices = prices[-min(len(prices), lookback):]
        if len(recent_prices) < 2:
            return 0.5
        
        high = np.max(recent_prices)
        low = np.min(recent_prices)
        current = prices[-1]
        
        if high == low:
            return 0.5
        
        position = (current - low) / (high - low)
        return float(np.clip(position, 0.0, 1.0))
    
    def _calculate_trend_strength(self, prices: np.ndarray, period: int = 10) -> float:
        """Calculate trend strength as slope of price over period."""
        if len(prices) < period:
            period = len(prices)
        
        if period < 2:
            return 0.0
        
        recent_prices = prices[-period:]
        x = np.arange(len(recent_prices))
        
        # Linear regression to get slope
        slope = np.polyfit(x, recent_prices, 1)[0]
        
        # Normalize by entry price
        entry_price = float(self.entry_price)
        trend_strength = slope / entry_price if entry_price > 0 else 0.0
        
        return float(trend_strength)
    
    def _calculate_volatility_adjusted_return(self, returns: np.ndarray) -> float:
        """Calculate return divided by volatility."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        if volatility < 1e-10:
            return 0.0
        
        return float(mean_return / volatility)
    
    def _calculate_price_acceleration(self, prices: np.ndarray, period: int = 5) -> float:
        """Calculate rate of change of returns (acceleration)."""
        if len(prices) < period + 1:
            return 0.0
        
        # Calculate returns
        returns = np.diff(prices[-period-1:]) / prices[-period-1:-1]
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate rate of change of returns
        acceleration = returns[-1] - returns[0]
        
        return float(acceleration)
    
    def _calculate_distance_to_extremes(self, prices: np.ndarray, lookback: int = 20) -> Tuple[float, float]:
        """Calculate distance to recent high and low."""
        if len(prices) < 2:
            return 0.0, 0.0
        
        recent_prices = prices[-min(len(prices), lookback):]
        current = prices[-1]
        high = np.max(recent_prices)
        low = np.min(recent_prices)
        
        # Normalize by entry price
        entry_price = float(self.entry_price)
        dist_to_high = (high - current) / entry_price if entry_price > 0 else 0.0
        dist_to_low = (current - low) / entry_price if entry_price > 0 else 0.0
        
        return float(dist_to_high), float(dist_to_low)
    
    def _calculate_rate_of_change(self, prices: np.ndarray, period: int = 5) -> float:
        """Calculate momentum (rate of change) over period."""
        if len(prices) < period + 1:
            return 0.0
        
        roc = (prices[-1] - prices[-period-1]) / prices[-period-1]
        return float(roc)
    
    def _calculate_return_consistency(self, returns: np.ndarray) -> float:
        """Calculate return consistency (inverse of volatility)."""
        if len(returns) < 2:
            return 1.0
        
        volatility = np.std(returns)
        if volatility < 1e-10:
            return 1.0
        
        # Inverse volatility (higher = more consistent)
        consistency = 1.0 / (1.0 + volatility)
        return float(consistency)
        
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector with 60 timesteps of features.
        
        Returns:
            Observation array of 360 features (60 timesteps × 6 features):
            - MACD (normalized by entry price)
            - MACD Histogram (normalized by entry price)
            - RSI (normalized to [0, 1])
            - EMA 25 (percent-change from entry price)
            - EMA 99 (percent-change from entry price)
            - Close Price (percent-change from entry price)
        """
        # Ensure environment is initialized (should have been reset)
        if self.price_window is None or self.balance_window is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self.current_idx is None:
            # First step, use entry point
            self.current_idx = 0
        
        # Ensure we don't go beyond available data
        if self.current_idx >= len(self.price_window):
            self.current_idx = len(self.price_window) - 1
        
        # Get entry price for normalization
        entry_price = float(self.entry_price)
        
        # Collect 60 timesteps of features
        # Start from max(0, current_idx - timeseries_length + 1) to current_idx + 1
        lookback_start = max(0, self.current_idx - self.timeseries_length + 1)
        timestep_features = []
        
        for i in range(lookback_start, self.current_idx + 1):
            if i >= len(self.price_window):
                break
            
            # Get price history up to this timestep (need enough history for indicators)
            price_history = []
            # Need at least 99 periods for EMA 99, but use available data
            history_start = max(0, i - 99)
            for j in range(history_start, i + 1):
                if j < len(self.price_window):
                    price_history.append(float(self.price_window.iloc[j]))
            
            if len(price_history) == 0:
                # Fallback: use current price
                price_history = [float(self.price_window.iloc[min(i, len(self.price_window) - 1)])]
            
            price_array = np.array(price_history)
            current_price = float(price_array[-1])
            
            # Calculate MACD (normalized by entry price)
            macd_line, macd_signal, macd_histogram = self._calculate_macd(price_array, fast=12, slow=26, signal=9)
            
            # Calculate RSI (normalized to [0, 1])
            rsi_raw = self._calculate_rsi(price_array, period=14)
            rsi_normalized = rsi_raw  # Already normalized to [0, 1] in _calculate_rsi
            
            # Calculate EMA 25 and EMA 99 (percent-change from entry price)
            ema_25 = self._calculate_ema(price_array, period=25)
            ema_99 = self._calculate_ema(price_array, period=99)
            
            # Normalize EMAs and close price by percent-change from entry price
            # This handles future prices that may be out of training range
            if entry_price > 0:
                ema_25_pct = (ema_25 - entry_price) / entry_price
                ema_99_pct = (ema_99 - entry_price) / entry_price
                close_pct = (current_price - entry_price) / entry_price
            else:
                ema_25_pct = 0.0
                ema_99_pct = 0.0
                close_pct = 0.0
            
            # Features for this timestep: [MACD, Histogram, RSI, EMA25, EMA99, Close]
            timestep_features.append([
                float(macd_line),      # MACD (normalized by entry price)
                float(macd_histogram), # MACD Histogram (normalized by entry price)
                float(rsi_normalized), # RSI (normalized to [0, 1])
                float(ema_25_pct),     # EMA 25 (percent-change from entry)
                float(ema_99_pct),     # EMA 99 (percent-change from entry)
                float(close_pct)       # Close Price (percent-change from entry)
            ])
        
        # Pad with zeros at the beginning if we don't have enough history
        while len(timestep_features) < self.timeseries_length:
            # Use zero values for padding (represents no change from entry)
            timestep_features.insert(0, [0.0, 0.0, 0.5, 0.0, 0.0, 0.0])  # RSI defaults to 0.5 (neutral)
        
        # Take only the last N timesteps
        timestep_features = timestep_features[-self.timeseries_length:]
        
        # Flatten to 1D array: 60 timesteps × 6 features = 360 features
        observation = np.array(timestep_features, dtype=np.float32).flatten()
        
        return observation
    
    def _calculate_reward_total_return_only(self, action: int, done: bool) -> float:
        """
        Calculate reward focused on encouraging closing when price is close to max price in episode.
        
        An episode is defined from entry price to exit price (determined by signals).
        If exit signal doesn't exist, max_steps=100 is used as episode limit.
        
        Action space: 0 = Hold, 1 = Close
        
        Reward strategy:
        - Primary goal: Reward closing when current price is close to episode max price
        - Holding: Small neutral reward, but penalty if price is close to episode min price
        - Closing near max: High reward
        - Closing far from max: Low/negative reward
        
        Reward Structure:
        - Holding (action 0): Small neutral reward (0.01), but penalty if close to min price
        - Closing (action 1): Reward based on how close current price is to episode max price
        """
        current_price = float(self.price_window.iloc[self.current_idx])
        periods_held = self.current_idx + 1
        # Calculate price range for normalization
        if self.episode_max_price is None:
            self.episode_max_price = self.entry_price
        if self.episode_min_price is None:
            self.episode_min_price = self.entry_price

        price_change_pct = (current_price - self.entry_price) / self.entry_price
        position_return = price_change_pct - 2 * self.fee_rate  # Account for entry + exit fees
        price_distance_from_max = abs(self.episode_max_price - current_price)
        # Calculate price range in episode (max - min) for normalization
        price_range = self.episode_max_price - self.episode_min_price
        
        # Track statistics for portfolio-level metrics
        if action == 1 or done:  # Closing
            if position_return > 0:
                self.wins += 1
            self.total_trades += 1
            self.returns_history.append(position_return)
        
        # ========== HOLDING ACTIONS (action 0) ==========
        if action == 0 and not done:
            # Initialize reward with small neutral value to allow exploration
            reward = 0.01
            
            price_range = self.episode_max_price - self.episode_min_price
            
            # Calculate how close current price is to episode min price
            if price_range > 1e-10:  # Avoid division by zero
                # Distance from min price
                distance_from_min = current_price - self.episode_min_price
                # Normalized distance: 0 = at min, 1 = at max
                normalized_distance_from_min = distance_from_min / price_range
                # Closeness to min: 1.0 = at min, 0.0 = at max
                closeness_to_min = 1.0 - normalized_distance_from_min
            else:
                # No price movement in episode (max == min)
                # Check if current price is at or near min
                price_distance_from_min = abs(self.episode_min_price - current_price)
                if price_distance_from_min < 1e-10:
                    closeness_to_min = 1.0
                else:
                    closeness_to_min = 0.0
            
            # Apply penalty if holding when price is close to min price
            # This encourages closing before price drops too much
            if closeness_to_min >= 0.9:
                # Very close to min: Strong penalty
                penalty = -2.0 * closeness_to_min  # -1.8 to -2.0
                reward += penalty
            elif closeness_to_min >= 0.7:
                # Close to min: Moderate penalty
                penalty = -1.0 * closeness_to_min  # -0.7 to -1.0
                reward += penalty
            elif closeness_to_min >= 0.5:
                # Approaching min: Small penalty
                penalty = -0.5 * closeness_to_min  # -0.25 to -0.5
                reward += penalty
            
            # Clip holding reward to reasonable range
            reward = np.clip(reward, -2.0, 0.1)
            return float(reward)
        
        # ========== CLOSING ACTIONS (action 1 or done=True) ==========
        # Calculate how close current price is to episode max price
        # Calculate closeness to max price (0.0 = far from max, 1.0 = at max)
        if price_range > 1e-10:  # Avoid division by zero
            # Normalized distance: 0 = at max, 1 = at min
            normalized_distance = price_distance_from_max / price_range
            # Closeness: 1.0 = at max, 0.0 = at min
            closeness_to_max = 1.0 - normalized_distance
        else:
            # No price movement in episode (max == min)
            # If current price equals max (or very close), reward closing
            if price_distance_from_max < 1e-10:
                closeness_to_max = 1.0
            else:
                closeness_to_max = 0.0
        
        # ========== REWARD BASED ON CLOSENESS TO MAX PRICE ==========
        # Primary reward: proportional to closeness to max price
        # Scale: 0.0 (far from max) to 1.0 (at max)
        # For PPO stability: keep rewards in reasonable range [0, 10]
        
        if closeness_to_max >= 0.9:
            # Excellent: Closing very close to max (within 10% of range)
            reward = 10.0 * closeness_to_max  # 9.0 to 10.0
        elif closeness_to_max >= 0.7:
            # Very good: Closing near max (within 30% of range)
            reward = 7.0 * closeness_to_max  # ~4.9 to 6.3
        elif closeness_to_max >= 0.5:
            # Good: Closing above midpoint
            reward = 5.0 * closeness_to_max  # ~2.5 to 3.5
        elif closeness_to_max >= 0.3:
            # Fair: Closing below midpoint but not at bottom
            reward = 2.0 * closeness_to_max  # ~0.6 to 1.0
        elif closeness_to_max >= 0.1:
            # Poor: Closing far from max
            reward = 0.5 * closeness_to_max  # ~0.05 to 0.5
        else:
            # Very poor: Closing at or near minimum price
            reward = -1.0  # Penalty for closing at worst price
        
        # ========== BONUS FOR PROFITABLE CLOSES ==========
        # Small bonus if closing with profit (encourages profitable exits)
        if action == 1 or done:
            price_change_pct = (current_price - self.entry_price) / self.entry_price
            position_return = price_change_pct - 2 * self.fee_rate
            
            if position_return > 0:
                # Small bonus for profitable close (scaled by profit)
                profit_bonus = min(position_return * 2.0, 2.0)  # Max bonus of 2.0
                reward += profit_bonus
            elif position_return < -0.05:  # Large loss (>5%)
                # Small penalty for large losses
                reward -= 0.5
        
        # ========== FINAL REWARD CLIPPING ==========
        # Clip reward to prevent extreme values
        # For PPO with VecNormalize: raw rewards in reasonable range [-2, 10]
        # VecNormalize will normalize and clip normalized rewards to [-5, 5]
        reward = np.clip(reward, -2.0, 10.0)
        
        return float(reward)
    
    def _calculate_reward(self, action: int, done: bool) -> float:
        """
        Calculate reward based on configured reward function.
        
        This is the main entry point that routes to the appropriate reward function.
        """
        return self._calculate_reward_total_return_only(action, done)
    
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
            
            # Select entry point from technical signals if available, otherwise random
            signal_indices = self._entry_signal_indices.get(self.current_ticker)
            
            if signal_indices is not None and len(signal_indices) > 0:
                # Use technical signal as entry point
                entry_idx = int(np.random.choice(signal_indices))
            else:
                # Fallback: randomly select entry point (must have enough history and an exit signal after)
                min_entry_idx = self.history_length
                # Need to ensure there's an exit signal after entry
                max_entry_idx = len(price_series) - 10  # Leave some room for exit signal
                
                if max_entry_idx <= min_entry_idx:
                    # Not enough data, use available range
                    min_entry_idx = 0
                    max_entry_idx = max(0, len(price_series) - 10)
                
                if max_entry_idx > min_entry_idx:
                    entry_idx = np.random.randint(min_entry_idx, max_entry_idx)
                else:
                    entry_idx = min_entry_idx if min_entry_idx < len(price_series) else 0
            
            # Exit index will be set based on exit signals below
            
            # Set entry price
            entry_price = float(price_series.iloc[entry_idx])
            
            # Update environment with selected ticker and position
            self.price_data = price_series
            self.balance_data = balance_series
            
            # Get exit signals for this ticker
            ticker_data = self.all_tickers_data[self.current_ticker]
            self.exit_signals = ticker_data.get('exit_signals')
            
            # Find next exit signal after entry (if exit signals available)
            exit_signal_idx = None
            if self.exit_signals is not None and len(self.exit_signals) > 0:
                # Find exit signals after entry point
                exit_indices_after_entry = np.where(
                    (self.exit_signals.values) & 
                    (np.arange(len(self.exit_signals)) > entry_idx)
                )[0]
                if len(exit_indices_after_entry) > 0:
                    exit_signal_idx = exit_indices_after_entry[0]
            
            self.entry_price = entry_price
            self.entry_idx = entry_idx
            self.exit_signal_idx = exit_signal_idx
            
            # Set exit_idx: use exit signal only (required)
            if exit_signal_idx is not None:
                # Use exit signal as episode end
                exit_idx = exit_signal_idx
            else:
                # No exit signal found - skip this entry by setting exit to entry (will result in episode_length=1)
                # This should rarely happen if exit signals are properly generated
                logger.warning(f"No exit signal found after entry at index {entry_idx} for {self.current_ticker}, using end of data")
                exit_idx = len(price_series) - 1
            
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
        
        # Reset max/min price tracking (start with entry price)
        self.episode_max_price = self.entry_price
        self.episode_min_price = self.entry_price
        
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
        # Update max and min prices during episode (from entry point onwards)
        # Calculate entry index relative to price_window
        entry_idx_in_window = self.entry_idx - self.data_start_idx if self.data_start_idx > 0 else self.entry_idx
        entry_idx_in_window = max(0, entry_idx_in_window)
        
        # Only track prices from entry point to current position
        current_price = float(self.price_window.iloc[self.current_idx])
        
        # Update max/min prices if we're at or past the entry point
        if self.current_idx >= entry_idx_in_window:
            if self.episode_max_price is None or current_price > self.episode_max_price:
                self.episode_max_price = current_price
            if self.episode_min_price is None or current_price < self.episode_min_price:
                self.episode_min_price = current_price
        
        # Check if position should be closed
        if action == 1:  # Close action
            self.position_open = False
        
        # Calculate reward
        reward = self._calculate_reward(action, done=False)
        self.total_reward += reward
        
        # Move to next step
        self.current_step += 1
        self.current_idx += 1
        
        # Check termination conditions
        terminated = not self.position_open  # Closed by action
        
        # Check if we've reached an exit signal (only in all_tickers mode)
        exit_signal_reached = False
        if hasattr(self, 'exit_signals') and self.exit_signals is not None and hasattr(self, 'exit_signal_idx') and self.exit_signal_idx is not None:
            # Calculate actual index in price_data
            actual_idx = self.entry_idx + self.current_idx
            if actual_idx >= self.exit_signal_idx:
                exit_signal_reached = True
        
        truncated = (
            self.current_idx >= self.episode_length - 1 or  # Reached episode boundary
            exit_signal_reached  # Reached exit signal
        )
        done = terminated or truncated
        
        # Get next observation
        if done:
            # Final step: use exit price
            self.current_idx = self.episode_length - 1
            if not terminated:
                # Force close if truncated
                self.position_open = False
            
            # Calculate final return when done (ensure it's always calculated)
            current_price = float(self.price_window.iloc[min(self.current_idx, len(self.price_window) - 1)])
            price_change_pct = (current_price - self.entry_price) / self.entry_price
            self.episode_return = price_change_pct - 2 * self.fee_rate
        
        observation = self._get_observation()
        
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

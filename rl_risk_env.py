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
        
        # Observation space: 68 features (28 base + 40 timeseries)
        # Original 12 features:
        # 1. Current account balance (normalized)
        # 2. Position unrealized P&L %
        # 3. Periods held (normalized)
        # 4. Max price change in percentage (since entry)
        # 5. Min price change in percentage (since entry)
        # 6. Current price change in percentage (since entry)
        # 7. Max Balance change in percentage (since entry)
        # 8. Min Balance change in percentage (since entry)
        # 9. Current Balance change in percentage (since entry)
        # 10. Current price position relative to entry (normalized)
        # 11. Recent volatility (rolling std of returns)
        # 12. Current drawdown from peak
        # Technical indicator features (13-28):
        # 13. RSI (14 period)
        # 14. MACD line
        # 15. MACD signal line
        # 16. MACD histogram
        # 17. Short-term MA (5 periods)
        # 18. Long-term MA (20 periods)
        # 19. MA difference (short - long) normalized
        # 20. Price position in recent range (0-1, relative to high/low in last 20 periods)
        # 21. Trend strength (slope of price over last 10 periods)
        # 22. Volatility-adjusted return (return / volatility)
        # 23. Price acceleration (rate of change of returns)
        # 24. Distance to recent high (last 20 periods)
        # 25. Distance to recent low (last 20 periods)
        # 26. Rate of change (momentum over last 5 periods)
        # 27. Price relative to entry (normalized to [-1, 1])
        # 28. Return consistency (inverse of return volatility)
        # Timeseries features (29-68):
        # 29-48. Historical price sequence (last 20 prices, normalized by entry price)
        # 49-68. Historical return sequence (last 20 returns)
        self.timeseries_length = 20  # Number of historical periods to include
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28 + 2 * self.timeseries_length,),  # 28 base + 20 prices + 20 returns = 68
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
        Construct observation vector from current state.
        
        Returns:
            Observation array of 68 features (28 base + 20 historical prices + 20 historical returns)
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
        
        # Current price change % (since entry)
        price_change_pct = (current_price - self.entry_price) / self.entry_price
        
        # Position unrealized P&L % (accounting for fees if we exit now)
        # Net return = (price_change - 2*fee_rate) since we pay fees on entry and exit
        unrealized_pnl_pct = price_change_pct - 2 * self.fee_rate
        
        # Periods held (normalized by max steps)
        # Use episode_length for normalization instead of max_steps
        periods_held = self.current_idx / max(self.episode_length, 1) if hasattr(self, 'episode_length') and self.episode_length > 0 else 0.0
        
        # Calculate price change statistics (max, min, current) since entry
        # Get all price changes from entry point to current position
        price_changes = []
        initial_balance_value = float(self.balance_data.iloc[max(0, self.data_start_idx)])
        
        # Entry point relative to window start
        entry_idx_in_window = self.entry_idx - self.data_start_idx if self.data_start_idx > 0 else self.entry_idx
        entry_idx_in_window = max(0, entry_idx_in_window)
        
        # Calculate price changes from entry point to current position
        for i in range(entry_idx_in_window, self.current_idx + 1):
            if i < len(self.price_window):
                hist_price = float(self.price_window.iloc[i])
                price_change = (hist_price - self.entry_price) / self.entry_price
                price_changes.append(price_change)
        
        # Calculate max, min, and current price change
        if len(price_changes) > 0:
            max_price_change = max(price_changes)
            min_price_change = min(price_changes)
            current_price_change = price_change_pct
        else:
            # Fallback if no history yet (shouldn't happen, but safe)
            max_price_change = price_change_pct
            min_price_change = price_change_pct
            current_price_change = price_change_pct
        
        # Calculate balance change statistics (max, min, current) since entry
        # Get all balance changes from entry point to current position
        balance_changes = []
        
        # Calculate balance changes from entry point to current position
        for i in range(entry_idx_in_window, self.current_idx + 1):
            if i < len(self.balance_window):
                hist_balance = float(self.balance_window.iloc[i])
                balance_change = (hist_balance - initial_balance_value) / initial_balance_value
                balance_changes.append(balance_change)
        
        # Calculate max, min, and current balance change
        if len(balance_changes) > 0:
            max_balance_change = max(balance_changes)
            min_balance_change = min(balance_changes)
            current_balance_change = (current_balance - initial_balance_value) / initial_balance_value
        else:
            # Fallback if no history yet
            current_balance_change = (current_balance - initial_balance_value) / initial_balance_value
            max_balance_change = current_balance_change
            min_balance_change = current_balance_change
        
        # Current price position relative to entry (normalized)
        # Use a simple normalization: clip to [-2, 2] range and normalize
        price_relative = np.clip(price_change_pct, -0.5, 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Recent volatility (rolling std of returns over last 30 periods)
        if self.current_idx >= 1:
            lookback = min(self.current_idx + 1, 30)  # Use last 30 periods
            recent_prices = [float(self.price_window.iloc[max(0, self.current_idx - i)]) 
                           for i in range(lookback)]
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
        
        # Get price history for technical indicators
        # Use prices from the start of the window to current position
        price_history = []
        for i in range(max(0, self.current_idx - 60), self.current_idx + 1):
            if i < len(self.price_window):
                price_history.append(float(self.price_window.iloc[i]))
        
        if len(price_history) == 0:
            price_history = [current_price]
        
        price_array = np.array(price_history)
        
        # Calculate technical indicators
        rsi = self._calculate_rsi(price_array, period=14)
        macd_line, macd_signal, macd_histogram = self._calculate_macd(price_array, fast=12, slow=26, signal=9)
        short_ma, long_ma, ma_diff = self._calculate_moving_averages(price_array, short_period=5, long_period=20)
        price_position = self._calculate_price_position_in_range(price_array, lookback=20)
        trend_strength = self._calculate_trend_strength(price_array, period=10)
        
        # Calculate returns for volatility-adjusted metrics
        if len(price_array) > 1:
            returns = np.diff(price_array) / price_array[:-1]
        else:
            returns = np.array([0.0])
        
        volatility_adjusted_return = self._calculate_volatility_adjusted_return(returns)
        price_acceleration = self._calculate_price_acceleration(price_array, period=5)
        dist_to_high, dist_to_low = self._calculate_distance_to_extremes(price_array, lookback=20)
        rate_of_change = self._calculate_rate_of_change(price_array, period=5)
        return_consistency = self._calculate_return_consistency(returns)
        
        # ========== TIMESERIES DATA: Historical prices and returns ==========
        # Extract historical price sequence (last N periods, normalized by entry price)
        historical_prices = []
        lookback_start = max(0, self.current_idx - self.timeseries_length + 1)
        for i in range(lookback_start, self.current_idx + 1):
            if i < len(self.price_window):
                hist_price = float(self.price_window.iloc[i])
                # Normalize by entry price to make it relative
                price_normalized = (hist_price - self.entry_price) / self.entry_price
                historical_prices.append(price_normalized)
        
        # Pad with zeros if not enough history
        while len(historical_prices) < self.timeseries_length:
            historical_prices.insert(0, 0.0)
        
        # Take only the last N periods
        historical_prices = historical_prices[-self.timeseries_length:]
        
        # Extract historical return sequence (last N periods)
        historical_returns = []
        if len(price_array) > 1:
            # Calculate returns from price array
            price_returns = np.diff(price_array) / price_array[:-1]
            # Get last N returns
            lookback_returns = min(self.timeseries_length, len(price_returns))
            historical_returns = price_returns[-lookback_returns:].tolist()
        
        # Pad with zeros if not enough history
        while len(historical_returns) < self.timeseries_length:
            historical_returns.insert(0, 0.0)
        
        # Take only the last N periods
        historical_returns = historical_returns[-self.timeseries_length:]
        
        # Convert to numpy arrays
        historical_prices = np.array(historical_prices, dtype=np.float32)
        historical_returns = np.array(historical_returns, dtype=np.float32)
        
        # Construct observation vector (68 features: 28 base + 20 prices + 20 returns)
        observation = np.concatenate([
            # Base features (28)
            np.array([
                # Original 12 features
                normalized_balance,          # 0: Current account balance (normalized)
                unrealized_pnl_pct,          # 1: Position unrealized P&L %
                periods_held,                # 2: Periods held (normalized)
                max_price_change,            # 3: Max price change in percentage
                min_price_change,            # 4: Min price change in percentage
                current_price_change,        # 5: Current price change in percentage
                max_balance_change,          # 6: Max Balance change in percentage
                min_balance_change,          # 7: Min Balance change in percentage
                current_balance_change,      # 8: Current Balance change in percentage
                price_relative,              # 9: Current price relative to entry
                volatility,                  # 10: Recent volatility
                current_drawdown,            # 11: Current drawdown
                
                # Technical indicator features (13-28)
                rsi,                         # 12: RSI (14 period, normalized to [0,1])
                macd_line,                   # 13: MACD line (normalized)
                macd_signal,                 # 14: MACD signal line (normalized)
                macd_histogram,              # 15: MACD histogram (normalized)
                short_ma,                    # 16: Short-term MA (5 periods, normalized)
                long_ma,                     # 17: Long-term MA (20 periods, normalized)
                ma_diff,                     # 18: MA difference (short - long, normalized)
                price_position,              # 19: Price position in recent range [0,1]
                trend_strength,              # 20: Trend strength (slope, normalized)
                volatility_adjusted_return,  # 21: Volatility-adjusted return
                price_acceleration,          # 22: Price acceleration (rate of change of returns)
                dist_to_high,                # 23: Distance to recent high (normalized)
                dist_to_low,                 # 24: Distance to recent low (normalized)
                rate_of_change,              # 25: Rate of change (momentum over 5 periods)
                np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns) if len(returns) > 0 else 0.0,  # 26: Average return over last 10 periods
                return_consistency           # 27: Return consistency (inverse volatility)
            ], dtype=np.float32),
            # Timeseries features (29-68)
            historical_prices,               # 29-48: Historical price sequence (last 20 prices, normalized)
            historical_returns               # 49-68: Historical return sequence (last 20 returns)
        ])
        
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

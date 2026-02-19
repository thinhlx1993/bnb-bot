"""
Custom Gymnasium Environment for RL Risk Management
This environment manages existing trading positions by deciding when to hold or close them.
Technical indicators are computed via TA-Lib where available; TSI and PVT remain custom.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import talib
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Shared env config: single source of truth for train and evaluate.
# Both train_rl_agent and rl_risk_management use this so observation space and normalization stay in sync.
ENV_DEFAULT_CONFIG = {
    "history_length": 50,
    "max_steps": 500,
    "fee_rate": 0.001,
}


class RiskManagementEnv(gym.Env):
    """
    Custom Gymnasium environment for position risk management.
    
    The agent decides whether to hold or close an existing position based on:
    - Account balance status
    - Market price changes
    - Position performance history
    - Current drawdown

    The observation is the current bar only (14 market + 6 account features); RecurrentPPO
    LSTM provides temporal context across env steps.
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
        max_steps: int = 100,
        fee_rate: float = 0.001,
        render_mode: Optional[str] = None,
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
            max_steps: From entry bar, allow the bot to play this many steps (bars). Episode ends at
                min(entry_idx + max_steps - 1, end of data). Also used to normalize periods_held in the observation.
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
            
            # Pre-calculate entry signal indices: valid = enough history before + room for max_steps after
            self._entry_signal_indices = {}
            self._exit_signal_indices = {}
            for ticker in self.tickers_list:
                ticker_data = all_tickers_data[ticker]
                price_len = len(ticker_data['price'])
                
                # Entry signals: require history_length before and max_steps bars after entry
                if 'entry_signals' in ticker_data and ticker_data['entry_signals'] is not None:
                    signals = ticker_data['entry_signals']
                    signal_indices = np.where(signals.values)[0]
                    valid_mask = (
                        (signal_indices >= history_length) &
                        (signal_indices + max_steps <= price_len)
                    )
                    valid_indices = signal_indices[valid_mask]
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
        self.ohlcv_window = None  # Optional DataFrame (high, low, close, volume) for OBV/MFI/AD/PVT
        
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
        
        # Observation space: current bar only (14 market + 6 account = 20). RecurrentPPO LSTM
        # runs over env steps, so history is in the recurrence, not in the observation.
        # Market: MACD, Histogram, RSI, EMA25, EMA99, Close, CCI, MFI, OBV_norm, Williams, TSI, ROC, AD_norm, PVT_norm
        # Account: unrealized_return, balance_ratio, current_drawdown_from_entry,
        #   max_profit_from_entry_so_far, max_drawdown_from_entry_so_far, periods_held_normalized
        self.n_features_per_timestep = 14
        self.n_account_features = 6
        obs_size = self.n_features_per_timestep + self.n_account_features  # 20
        obs_shape = (obs_size,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
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
        
        # Track max and min prices during episode (used for reward only; not in obs)
        self.episode_max_price = None
        self.episode_min_price = None
        # Track from entry (observable in live: no future knowledge)
        self.max_profit_from_entry = 0.0   # max (price - entry)/entry seen so far
        self.max_drawdown_from_entry = 0.0  # max (entry - price)/entry when underwater, so far
        
        # Pre-computed indicators cache (for performance optimization)
        self._precomputed_indicators = None
        
        # Configurable reward component weights
        # These allow tuning the importance of different reward components
        self.reward_weights = {
            'closeness_to_max': 1.0,      # Base reward for closing near max price
            'profit_bonus': 1.0,          # Bonus for profitable closes
            'loss_penalty': 1.0,          # Penalty for large losses
            'drawdown_penalty': 1.0,      # Penalty for holding during drawdowns
            'max_drawdown_penalty': 1.0,  # Penalty for trades with large max drawdown
            'time_efficiency': 0.0,       # Disabled - was encouraging close at fixed duration (~12h45)
            'early_exit_bonus': 0.0,      # DISABLED - was encouraging early exits around ~12h
                                          # (the larger early_exit_factor at small periods_held encouraged closing too soon)
            'profit_efficiency': 0.0,      # DISABLED - was unfairly penalizing longer episodes
                                          # (dividing return by periods_held meant same return = smaller reward)
            'holding_base': 0.0,          # Base reward for holding
            'holding_penalty': 0.01,       # Small penalty per hold step (encourages closing instead of holding to end)
            'closeness_to_min_penalty': 2.0,  # Penalty for holding near min price
            'momentum_reward': 0.5,       # Reward for holding when price is trending up
            'unrealized_profit_reward': 0.3,  # Reward for holding profitable positions (encourages letting winners run)
            'closeness_to_max_reward': 1.0,  # Reward for holding when close to max price (potential for more)
        }
        
        # Track reward components for debugging (optional)
        self._reward_components = {}

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
        
    def _precompute_indicators(self):
        """
        Pre-compute all technical indicators for the current price window using TA-Lib
        where available. TSI and PVT have no TA-Lib equivalent and are computed manually.
        Called once per episode (in reset()) for performance.
        """
        prices = np.asarray(self.price_window.values, dtype=np.float64)
        entry_price = float(self.entry_price)
        price_series = pd.Series(prices)

        # ---- TA-Lib: MACD ----
        macd_line, signal_line, hist = talib.MACD(
            prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        macd_normalized = np.where(
            np.isfinite(macd_line),
            macd_line / entry_price if entry_price > 0 else 0.0,
            0.0,
        ).astype(np.float32)
        histogram_normalized = np.where(
            np.isfinite(hist),
            hist / entry_price if entry_price > 0 else 0.0,
            0.0,
        ).astype(np.float32)

        # ---- TA-Lib: RSI (normalize to [0, 1]) ----
        rsi_raw = talib.RSI(prices, timeperiod=14)
        rsi_values = np.where(np.isfinite(rsi_raw), np.clip(rsi_raw / 100.0, 0.0, 1.0), 0.5)
        rsi_values = rsi_values.astype(np.float32)

        # ---- TA-Lib: EMA 25, 99; close pct ----
        ema_25 = talib.EMA(prices, timeperiod=25)
        ema_99 = talib.EMA(prices, timeperiod=99)

        ema_25_pct = np.where(np.isfinite(ema_25), (ema_25 - entry_price) / entry_price, 0.0)
        ema_99_pct = np.where(np.isfinite(ema_99), (ema_99 - entry_price) / entry_price, 0.0)
        close_pct = (prices - entry_price) / entry_price

        ema_25_pct = ema_25_pct.astype(np.float32)
        ema_99_pct = ema_99_pct.astype(np.float32)
        close_pct = close_pct.astype(np.float32)

        # ---- TA-Lib: CCI (period=20). Use close for high/low when no OHLC ----
        high_cci = price_series.rolling(20, min_periods=1).max().values.astype(np.float64)
        low_cci = price_series.rolling(20, min_periods=1).min().values.astype(np.float64)
        cci_raw = talib.CCI(high_cci, low_cci, prices, timeperiod=20)
        cci_norm = np.clip(np.where(np.isfinite(cci_raw), cci_raw / 200.0, 0.0), -1.0, 1.0).astype(np.float32)

        # ---- TA-Lib: Williams %R (period=14). Rolling H/L from close when no OHLC ----
        high_wr = price_series.rolling(14, min_periods=1).max().values.astype(np.float64)
        low_wr = price_series.rolling(14, min_periods=1).min().values.astype(np.float64)
        wr = talib.WILLR(high_wr, low_wr, prices, timeperiod=14)
        williams_norm = np.where(np.isfinite(wr), np.clip(wr / 100.0, -1.0, 0.0), -0.5).astype(np.float32)

        # ---- TSI (no TA-Lib): fast=13, slow=25 ----
        pc = price_series.diff()
        pcds = pc.ewm(span=13, adjust=False).mean().ewm(span=25, adjust=False).mean()
        apcds = pc.abs().ewm(span=13, adjust=False).mean().ewm(span=25, adjust=False).mean()
        tsi = 100 * (pcds / apcds.replace(0, np.nan))
        tsi_norm = np.clip(tsi.fillna(0).values / 100.0, -1.0, 1.0).astype(np.float32)

        # ---- TA-Lib: ROC (period=12) ----
        roc_raw = talib.ROC(prices, timeperiod=12)
        roc_norm = np.clip(np.where(np.isfinite(roc_raw), roc_raw, 0.0), -1.0, 1.0).astype(np.float32)

        df = self.ohlcv_window
        high = np.asarray(df["high"].values, dtype=np.float64)
        low = np.asarray(df["low"].values, dtype=np.float64)
        close_vol = np.asarray(df["close"].values, dtype=np.float64)
        volume = np.asarray(df["volume"].values, dtype=np.float64)

        # TA-Lib OBV
        obv = talib.OBV(close_vol, volume)
        obv = np.where(np.isfinite(obv), obv, 0.0)
        obv_max = np.abs(obv).max()
        obv_norm = (obv / (obv_max + 1e-8)).astype(np.float32)

        # TA-Lib MFI (period=14)
        mfi_raw = talib.MFI(high, low, close_vol, volume, timeperiod=14)
        mfi_norm = np.where(np.isfinite(mfi_raw), np.clip(mfi_raw / 100.0, 0.0, 1.0), 0.5).astype(np.float32)

        # TA-Lib AD (Chaikin A/D)
        ad = talib.AD(high, low, close_vol, volume)
        ad = np.where(np.isfinite(ad), ad, 0.0)
        ad_max = np.abs(ad).max()
        ad_norm = (ad / (ad_max + 1e-8)).astype(np.float32)

        # PVT (no TA-Lib)
        pvt = (df["close"].pct_change().fillna(0) * df["volume"]).cumsum().values
        pvt_max = np.abs(pvt).max()
        pvt_norm = (pvt / (pvt_max + 1e-8)).astype(np.float32)

        return {
            "macd": macd_normalized,
            "histogram": histogram_normalized,
            "rsi": rsi_values,
            "ema_25": ema_25_pct,
            "ema_99": ema_99_pct,
            "close": close_pct,
            "cci": cci_norm,
            "mfi": mfi_norm,
            "obv": obv_norm,
            "williams": williams_norm,
            "tsi": tsi_norm,
            "roc": roc_norm,
            "ad": ad_norm,
            "pvt": pvt_norm,
        }
    
    def _get_account_features(self) -> np.ndarray:
        """
        Compute account/position features using only entry and current state
        (no future knowledge: no episode high/low). All metrics are from entry or
        tracked over time.
        Returns array of shape (6,) with normalized metrics.
        """
        current_price = float(self.price_window.iloc[self.current_idx])
        current_balance = float(self.balance_window.iloc[self.current_idx])
        entry_price = float(self.entry_price)
        entry_balance = float(self.entry_balance) if self.entry_balance > 0 else 1.0
        # Current unrealized return from entry (clip to [-1, 1])
        unrealized_return = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        unrealized_return = float(np.clip(unrealized_return, -1.0, 1.0))
        # Balance relative to entry (e.g. 1.0 = unchanged)
        balance_ratio = current_balance / entry_balance if entry_balance > 0 else 1.0
        balance_ratio = float(np.clip(balance_ratio, 0.0, 3.0))
        # Current drawdown from entry: how far underwater now (0 if in profit)
        if entry_price > 0 and current_price < entry_price:
            current_drawdown_from_entry = (entry_price - current_price) / entry_price
        else:
            current_drawdown_from_entry = 0.0
        current_drawdown_from_entry = float(np.clip(current_drawdown_from_entry, 0.0, 1.0))
        # Update and expose max profit from entry so far (tracked; no future knowledge)
        if entry_price > 0:
            current_return = (current_price - entry_price) / entry_price
            self.max_profit_from_entry = max(self.max_profit_from_entry, current_return)
        max_profit_so_far = float(np.clip(self.max_profit_from_entry, 0.0, 1.0))
        # Update and expose max drawdown from entry so far (tracked; no future knowledge)
        if entry_price > 0 and current_price < entry_price:
            dd_from_entry = (entry_price - current_price) / entry_price
            self.max_drawdown_from_entry = max(self.max_drawdown_from_entry, dd_from_entry)
        max_drawdown_from_entry_so_far = float(np.clip(self.max_drawdown_from_entry, 0.0, 1.0))
        # Time in position (0 = just entered, 1 = at max_steps). Normalized by max_steps for consistent scale.
        effective_max = max(self.max_steps, 1)
        periods_held_norm = float(self.current_idx) / effective_max if effective_max > 0 else 0.0
        periods_held_norm = float(np.clip(periods_held_norm, 0.0, 1.0))
        return np.array([
            unrealized_return,
            balance_ratio,
            current_drawdown_from_entry,
            max_profit_so_far,
            max_drawdown_from_entry_so_far,
            periods_held_norm,
        ], dtype=np.float32)
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation for the current bar only: 14 market features + 6 account features = 20.
        RecurrentPPO's LSTM runs over env steps, so temporal context comes from the recurrence.
        """
        # Ensure environment is initialized (should have been reset)
        if self.price_window is None or self.balance_window is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self.current_idx is None:
            self.current_idx = 0
        
        if self.current_idx >= len(self.price_window):
            self.current_idx = len(self.price_window) - 1
        
        idx = self.current_idx

        if self._precomputed_indicators is None:
            raise RuntimeError("Pre-computed indicators are None. Call reset() first so _precompute_indicators runs.")

        # Current bar only: one row of 14 market features
        market_row = np.array([
            self._precomputed_indicators['macd'][idx],
            self._precomputed_indicators['histogram'][idx],
            self._precomputed_indicators['rsi'][idx],
            self._precomputed_indicators['ema_25'][idx],
            self._precomputed_indicators['ema_99'][idx],
            self._precomputed_indicators['close'][idx],
            self._precomputed_indicators['cci'][idx],
            self._precomputed_indicators['mfi'][idx],
            self._precomputed_indicators['obv'][idx],
            self._precomputed_indicators['williams'][idx],
            self._precomputed_indicators['tsi'][idx],
            self._precomputed_indicators['roc'][idx],
            self._precomputed_indicators['ad'][idx],
            self._precomputed_indicators['pvt'][idx],
        ], dtype=np.float32)
        observation = np.append(market_row, self._get_account_features())
        return observation
    
    def _calculate_reward_long_term(self, action: int, done: bool) -> float:
        """
        Improved reward function for long-term performance optimization.
        
        Primary goal: Encourage the bot to close at episode max price within the max_steps
        (default 100 bars) window. No fixed bar count—reward is driven by how close the close price
        is to the episode max (closeness_to_max), not by when in the episode we close.
        
        Terminal-only reward: Reward is 0 on every hold step; all reward is given on the
        close step. This helps with action imbalance (many hold vs one close per episode).
        
        Key components:
        1. Hold: reward = 0 (terminal-only)
        2. Close: reward scaled by closeness to episode max, profit/loss, early-exit bonus, etc.
        3. Configurable component weights
        """
        current_price = float(self.price_window.iloc[self.current_idx])
        periods_held = self.current_idx + 1
        
        # Ensure max/min prices are initialized (should be set in reset(), but safety check)
        if self.episode_max_price is None:
            self.episode_max_price = self.entry_price
        if self.episode_min_price is None:
            self.episode_min_price = self.entry_price
        
        # Calculate price metrics
        price_change_pct = (current_price - self.entry_price) / self.entry_price
        position_return = price_change_pct - 2 * self.fee_rate  # Account for entry + exit fees
        price_range = self.episode_max_price - self.episode_min_price
        
        # Calculate current drawdown
        if self.episode_max_price > 0:
            current_drawdown = (self.episode_max_price - current_price) / self.episode_max_price
        else:
            current_drawdown = 0.0
        
        # Update maximum drawdown (episode-scoped only)
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Track statistics for evaluation only (NOT used in reward calculation)
        if action == 1 or done:  # Closing
            if position_return > 0:
                self.wins += 1
            self.total_trades += 1
            self.returns_history.append(position_return)
        
        # Reset reward components tracking
        self._reward_components = {}
        
        # ========== HOLDING ACTIONS (action 0) ==========
        # Small per-step penalty for holding (encourages closing when appropriate instead of holding to end).
        if action == 0 and not done:
            # Penalty only when holding a loss position (current price below entry)
            position_return = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0.0
            if position_return >= 0:
                self._reward_components['holding_penalty'] = 0.0
                return 0.0
            holding_penalty = self.reward_weights.get('holding_penalty', 0.02)
            # Slightly higher penalty later in episode so holding a loss near max_steps costs more
            if self.episode_length and self.episode_length > 0:
                progress = periods_held / self.episode_length
                step_penalty = holding_penalty * (0.5 + 0.5 * progress)
            else:
                step_penalty = holding_penalty
            self._reward_components['holding_penalty'] = -step_penalty
            return -float(step_penalty)
        
        # ========== CLOSING ACTIONS (action 1 or done=True) ==========
        # Calculate closeness to max price
        if price_range > 1e-10:
            price_distance_from_max = abs(self.episode_max_price - current_price)
            normalized_distance = price_distance_from_max / price_range
            closeness_to_max = 1.0 - normalized_distance
        else:
            # No price movement in episode (max == min)
            price_distance_from_max = abs(self.episode_max_price - current_price)
            if price_distance_from_max < 1e-10:
                closeness_to_max = 1.0
            else:
                closeness_to_max = 0.0
        
        # Base reward from closeness to max (scaled by weight)
        if closeness_to_max >= 0.9:
            reward = 8.0 * closeness_to_max * self.reward_weights['closeness_to_max']
        elif closeness_to_max >= 0.7:
            reward = 6.0 * closeness_to_max * self.reward_weights['closeness_to_max']
        elif closeness_to_max >= 0.5:
            reward = 4.0 * closeness_to_max * self.reward_weights['closeness_to_max']
        elif closeness_to_max >= 0.3:
            reward = 2.0 * closeness_to_max * self.reward_weights['closeness_to_max']
        elif closeness_to_max >= 0.1:
            reward = 0.5 * closeness_to_max * self.reward_weights['closeness_to_max']
        else:
            reward = -1.0 * self.reward_weights['closeness_to_max']
        
        self._reward_components['closeness_to_max'] = reward
        
        # 4. Early exit bonus (OPTIONAL - weight can be 0)
        # Small bonus for closing at good price (top 20% of range) before 70% of episode.
        # No fixed bar count: primary goal is to close at episode max price within max_steps (default 100).
        if closeness_to_max >= 0.8 and self.episode_length is not None and self.reward_weights['early_exit_bonus'] > 0:
            if periods_held < self.episode_length * 0.7:
                early_exit_factor = 1.0 - (periods_held / max(self.episode_length, periods_held))
                early_exit_bonus = early_exit_factor * 1.5 * self.reward_weights['early_exit_bonus']
                reward += early_exit_bonus
                self._reward_components['early_exit_bonus'] = early_exit_bonus
        
        # 5. Profit bonus/penalty
        if position_return > 0:
            # Profit bonus scaled by return magnitude
            profit_bonus = min(position_return * 2.5, 2.5) * self.reward_weights['profit_bonus']
            reward += profit_bonus
            self._reward_components['profit_bonus'] = profit_bonus
        elif position_return < -0.05:  # Large loss (>5%)
            # Penalty for large losses
            loss_penalty = abs(position_return) * 2.0 * self.reward_weights['loss_penalty']
            reward -= min(loss_penalty, 2.0)
            self._reward_components['loss_penalty'] = -min(loss_penalty, 2.0)
        
        # 7. Maximum drawdown penalty (episode-scoped only)
        # Penalize trades that experienced large drawdowns during the episode
        if self.max_drawdown > 0.1:  # 10% max drawdown threshold
            max_dd_penalty = min(self.max_drawdown * 2.0, 1.5) * self.reward_weights['max_drawdown_penalty']
            reward -= max_dd_penalty
            self._reward_components['max_drawdown_penalty'] = -max_dd_penalty
        
        # Final reward clipping (narrower range for better stability)
        reward = np.clip(reward, -3.0, 8.0)
        
        return float(reward)
    
    def _calculate_reward(self, action: int, done: bool) -> float:
        """
        Calculate reward based on configured reward function.
        
        This is the main entry point that routes to the appropriate reward function.
        
        To switch reward functions, change the return statement:
        - return self._calculate_reward_total_return_only(action, done)  # Original
        - return self._calculate_reward_long_term(action, done)  # Improved for long-term
        """
        return self._calculate_reward_long_term(action, done)  # Using improved long-term reward
    
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
                # Fallback: randomly select entry point (need history before and max_steps after)
                min_entry_idx = self.history_length
                max_entry_idx = len(price_series) - self.max_steps
                if max_entry_idx <= min_entry_idx:
                    min_entry_idx = 0
                    max_entry_idx = max(0, len(price_series) - self.max_steps)
                if max_entry_idx > min_entry_idx:
                    entry_idx = np.random.randint(min_entry_idx, max_entry_idx)
                else:
                    entry_idx = min_entry_idx if min_entry_idx < len(price_series) else 0
            
            # Set entry price
            entry_price = float(price_series.iloc[entry_idx])
            
            # Update environment with selected ticker and position
            self.price_data = price_series
            self.balance_data = balance_series
            
            ticker_data = self.all_tickers_data[self.current_ticker]
            self.exit_signals = ticker_data.get('exit_signals')
            self.exit_signal_idx = None  # Episode length is fixed from entry by max_steps
            
            self.entry_price = entry_price
            self.entry_idx = entry_idx
            
            # Episode: from entry, allow max_steps bars (capped by end of data)
            exit_idx = min(entry_idx + self.max_steps - 1, len(price_series) - 1)
            self.exit_idx = exit_idx
            
            # Set up episode windows
            self.episode_start_idx = entry_idx
            self.episode_end_idx = exit_idx
            self.episode_length = exit_idx - entry_idx + 1
            self.data_start_idx = max(0, entry_idx - self.history_length)
            self.price_window = price_series.iloc[self.data_start_idx:exit_idx + 1].copy()
            self.balance_window = balance_series.iloc[self.data_start_idx:exit_idx + 1].copy()
            # Optional OHLCV for CCI/Williams/TSI/ROC and OBV/MFI/AD/PVT
            ohlcv = ticker_data.get('ohlcv')
            if ohlcv is not None and len(ohlcv) > 0:
                self.ohlcv_window = ohlcv.iloc[self.data_start_idx:exit_idx + 1].copy()
            else:
                self.ohlcv_window = None
        
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
        
        # Reset max/min price tracking (start with entry price; used for reward only)
        self.episode_max_price = self.entry_price
        self.episode_min_price = self.entry_price
        # Reset entry-relative tracking (used in observation; no future knowledge)
        self.max_profit_from_entry = 0.0
        self.max_drawdown_from_entry = 0.0
        
        # Initialize reward components tracking
        self._reward_components = {}
        
        # Pre-compute indicators once per episode (major performance optimization!)
        # This avoids recalculating MACD, RSI, EMA on every step
        self._precomputed_indicators = self._precompute_indicators()
        
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
        # Ensure max/min prices are initialized (safety check)
        if self.episode_max_price is None:
            self.episode_max_price = self.entry_price
        if self.episode_min_price is None:
            self.episode_min_price = self.entry_price
        
        # Calculate entry index relative to price_window
        entry_idx_in_window = self.entry_idx - self.data_start_idx if self.data_start_idx > 0 else self.entry_idx
        entry_idx_in_window = max(0, entry_idx_in_window)
        
        # Only track prices from entry point to current position
        current_price = float(self.price_window.iloc[self.current_idx])
        
        # Update max/min prices if we're at or past the entry point
        if self.current_idx >= entry_idx_in_window:
            if current_price > self.episode_max_price:
                self.episode_max_price = current_price
            if current_price < self.episode_min_price:
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
            "periods_held": self.current_idx,
            "reward_components": self._reward_components.copy() if hasattr(self, '_reward_components') else {}
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

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

# Shared env config: single source of truth for train and evaluate.
# Both train_rl_agent and rl_risk_management use this so observation space and normalization stay in sync.
ENV_DEFAULT_CONFIG = {
    "history_length": 50,
    "max_steps": 100,
    "obs_periods_norm_steps": 100,
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
        obs_periods_norm_steps: Optional[int] = None,
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
                min(entry_idx + max_steps - 1, end of data). Default 100.
            obs_periods_norm_steps: If set, normalize periods_held in observation by this value (for eval/train
                consistency when max_steps differs). If None, use max_steps.
            fee_rate: Trading fee rate (e.g., 0.001 for 0.1%)
            render_mode: Rendering mode
        """
        super().__init__()

        # Normalization scale for periods_held in observation (must match training, e.g. 100)
        self.obs_periods_norm_steps = obs_periods_norm_steps if obs_periods_norm_steps is not None else max_steps

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
            'closeness_to_min_penalty': 1.0,  # Penalty for holding near min price
            'momentum_reward': 0.5,       # Reward for holding when price is trending up
            'unrealized_profit_reward': 0.3,  # Reward for holding profitable positions (encourages letting winners run)
            'closeness_to_max_reward': 1.0,  # Reward for holding when close to max price (potential for more)
        }
        
        # Track reward components for debugging (optional)
        self._reward_components = {}
    
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
        
    def _precompute_indicators(self):
        """
        Pre-compute all technical indicators for the current price window.
        This is called once per episode (in reset()) to avoid recalculating
        indicators on every step, which is a major performance bottleneck.
        """
        if self.price_window is None or len(self.price_window) == 0:
            return None
        
        prices = self.price_window.values
        entry_price = float(self.entry_price)
        n = len(prices)
        
        # Convert to pandas Series for efficient calculations
        price_series = pd.Series(prices)
        
        # Pre-compute MACD for all timesteps
        exp1 = price_series.ewm(span=12, adjust=False).mean()
        exp2 = price_series.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        macd_series = macd_line
        signal_line = macd_series.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Normalize MACD by entry price
        macd_normalized = (macd_line / entry_price).values if entry_price > 0 else np.zeros(n)
        histogram_normalized = (histogram / entry_price).values if entry_price > 0 else np.zeros(n)
        
        # Pre-compute RSI for all timesteps
        rsi_values = np.zeros(n)
        for i in range(n):
            if i < 14:
                rsi_values[i] = 0.5  # Neutral RSI if not enough data
            else:
                # Use sliding window approach
                window = prices[max(0, i-14):i+1]
                if len(window) >= 2:
                    deltas = np.diff(window)
                    gains = np.where(deltas > 0, deltas, 0.0)
                    losses = np.where(deltas < 0, -deltas, 0.0)
                    avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
                    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
                    if avg_loss == 0:
                        rsi_values[i] = 1.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        rsi_values[i] = rsi / 100.0  # Normalize to [0, 1]
                else:
                    rsi_values[i] = 0.5
        
        # Pre-compute EMAs
        ema_25 = price_series.ewm(span=25, adjust=False).mean()
        ema_99 = price_series.ewm(span=99, adjust=False).mean()
        
        # Normalize EMAs and prices by percent-change from entry price
        if entry_price > 0:
            ema_25_pct = ((ema_25 - entry_price) / entry_price).values
            ema_99_pct = ((ema_99 - entry_price) / entry_price).values
            close_pct = ((price_series - entry_price) / entry_price).values
        else:
            ema_25_pct = np.zeros(n)
            ema_99_pct = np.zeros(n)
            close_pct = np.zeros(n)

        # CCI (period=20): use close as typical price when no OHLC
        tp = price_series
        sma20 = tp.rolling(20, min_periods=1).mean()
        mad20 = tp.rolling(20, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp - sma20) / (0.015 * mad20.replace(0, np.nan))
        cci_norm = np.clip(cci.fillna(0).values / 200.0, -1.0, 1.0).astype(np.float32)

        # Williams %R (period=14): use rolling max/min of close when no high/low
        high = price_series.rolling(14, min_periods=1).max()
        low = price_series.rolling(14, min_periods=1).min()
        wr = -100 * (high - price_series) / (high - low).replace(0, np.nan)
        williams_norm = (wr.fillna(-50).values / 100.0).astype(np.float32)  # roughly [-1, 0]

        # TSI (fast=13, slow=25)
        pc = price_series.diff()
        pcds = pc.ewm(span=13, adjust=False).mean().ewm(span=25, adjust=False).mean()
        apcds = pc.abs().ewm(span=13, adjust=False).mean().ewm(span=25, adjust=False).mean()
        tsi = 100 * (pcds / apcds.replace(0, np.nan))
        tsi_norm = np.clip(tsi.fillna(0).values / 100.0, -1.0, 1.0).astype(np.float32)

        # ROC (period=12)
        roc = price_series.pct_change(12)
        roc_norm = np.clip(roc.fillna(0).values, -1.0, 1.0).astype(np.float32)

        # Volume-based: OBV, MFI, AD, PVT (use ohlcv_window when available)
        has_volume = (
            self.ohlcv_window is not None
            and isinstance(self.ohlcv_window, pd.DataFrame)
            and 'volume' in self.ohlcv_window.columns
        )
        if has_volume:
            df = self.ohlcv_window
            # OBV
            direction = np.sign(df['close'].diff().fillna(0))
            obv = (direction * df['volume']).cumsum().values
            obv_max = np.abs(obv).max()
            obv_norm = (obv / (obv_max + 1e-8)).astype(np.float32)
            # MFI (period=14)
            tp_df = (df['high'] + df['low'] + df['close']) / 3 if all(c in df.columns for c in ['high', 'low']) else df['close']
            if 'high' in df.columns and 'low' in df.columns:
                mf = tp_df * df['volume']
                delta = tp_df.diff()
                pos = mf.where(delta > 0, 0).rolling(14, min_periods=1).sum()
                neg = mf.where(delta < 0, 0).rolling(14, min_periods=1).sum()
                mfi = 100 - (100 / (1 + pos / neg.replace(0, np.nan)))
                mfi_norm = (mfi.fillna(50).values / 100.0).astype(np.float32)
            else:
                mfi_norm = np.full(n, 0.5, dtype=np.float32)
            # AD
            if all(c in df.columns for c in ['high', 'low']):
                cl = (df['close'] - df['low']) - (df['high'] - df['close'])
                cl = cl / (df['high'] - df['low']).replace(0, np.nan)
                ad = (cl * df['volume']).fillna(0).cumsum().values
                ad_max = np.abs(ad).max()
                ad_norm = (ad / (ad_max + 1e-8)).astype(np.float32)
            else:
                ad_norm = np.zeros(n, dtype=np.float32)
            # PVT
            pvt = (df['close'].pct_change().fillna(0) * df['volume']).cumsum().values
            pvt_max = np.abs(pvt).max()
            pvt_norm = (pvt / (pvt_max + 1e-8)).astype(np.float32)
        else:
            obv_norm = np.zeros(n, dtype=np.float32)
            mfi_norm = np.full(n, 0.5, dtype=np.float32)
            ad_norm = np.zeros(n, dtype=np.float32)
            pvt_norm = np.zeros(n, dtype=np.float32)

        return {
            'macd': macd_normalized.astype(np.float32),
            'histogram': histogram_normalized.astype(np.float32),
            'rsi': rsi_values.astype(np.float32),
            'ema_25': ema_25_pct.astype(np.float32),
            'ema_99': ema_99_pct.astype(np.float32),
            'close': close_pct.astype(np.float32),
            'cci': cci_norm,
            'mfi': mfi_norm,
            'obv': obv_norm,
            'williams': williams_norm,
            'tsi': tsi_norm,
            'roc': roc_norm,
            'ad': ad_norm,
            'pvt': pvt_norm,
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
        # Time in position (0 = just entered, 1 = at norm steps). Use obs_periods_norm_steps so eval matches training.
        effective_max = max(self.obs_periods_norm_steps, 1)
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
        
        if self._precomputed_indicators is not None:
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
        
        # Fallback: compute one bar of indicators for current_idx
        logger.warning("Pre-computed indicators not available, using slow fallback method")
        entry_price = float(self.entry_price)
        price_history = []
        history_start = max(0, idx - 99)
        for j in range(history_start, idx + 1):
            if j < len(self.price_window):
                price_history.append(float(self.price_window.iloc[j]))
        if len(price_history) == 0:
            price_history = [float(self.price_window.iloc[min(idx, len(self.price_window) - 1)])]
        price_array = np.array(price_history)
        current_price = float(price_array[-1])
        
        macd_line, _, macd_histogram = self._calculate_macd(price_array, fast=12, slow=26, signal=9)
        rsi_raw = self._calculate_rsi(price_array, period=14)
        ema_25 = self._calculate_ema(price_array, period=25)
        ema_99 = self._calculate_ema(price_array, period=99)
        if entry_price > 0:
            ema_25_pct = (ema_25 - entry_price) / entry_price
            ema_99_pct = (ema_99 - entry_price) / entry_price
            close_pct = (current_price - entry_price) / entry_price
        else:
            ema_25_pct = ema_99_pct = close_pct = 0.0
        
        market_row = np.array([
            float(macd_line), float(macd_histogram), float(rsi_raw),
            float(ema_25_pct), float(ema_99_pct), float(close_pct),
        ], dtype=np.float32)
        market_row = np.append(market_row, np.zeros(8, dtype=np.float32))  # CCI/MFI/OBV/Williams/TSI/ROC/AD/PVT placeholder
        observation = np.append(market_row, self._get_account_features())
        return observation
    
    def _calculate_reward_total_return_only(self, action: int, done: bool) -> float:
        """
        Calculate reward focused on encouraging closing when price is close to max price in episode.
        
        Episode has max_steps (default 100) bars from entry. Goal: close at episode max price.
        Terminal-only reward: hold steps get 0; all reward on close step.
        
        Action space: 0 = Hold, 1 = Close
        
        Reward strategy:
        - Holding (action 0): reward = 0
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
        
        # Track statistics for evaluation only (NOT used in reward calculation)
        if action == 1 or done:  # Closing
            if position_return > 0:
                self.wins += 1
            self.total_trades += 1
            self.returns_history.append(position_return)
        
        # Reset reward components tracking
        self._reward_components = {}
        
        # ========== HOLDING ACTIONS (action 0) ==========
        # Terminal-only reward: no per-step reward for hold. All reward on close step.
        if action == 0 and not done:
            self._reward_components['holding_base'] = 0.0
            return 0.0
        
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
        
        self._reward_components['closeness_to_max'] = reward
        
        # ========== BONUS FOR PROFITABLE CLOSES ==========
        # Small bonus if closing with profit (encourages profitable exits)
        # Note: position_return already calculated above, reuse it
        if action == 1 or done:
            if position_return > 0:
                # Small bonus for profitable close (scaled by profit)
                profit_bonus = min(position_return * 2.0, 2.0)  # Max bonus of 2.0
                reward += profit_bonus
                self._reward_components['profit_bonus'] = profit_bonus
            elif position_return < -0.05:  # Large loss (>5%)
                # Small penalty for large losses
                loss_penalty = -0.5
                reward += loss_penalty
                self._reward_components['loss_penalty'] = loss_penalty
        
        # ========== FINAL REWARD CLIPPING ==========
        # Clip reward to prevent extreme values
        # For PPO with VecNormalize: raw rewards in reasonable range [-2, 10]
        # VecNormalize will normalize and clip normalized rewards to [-5, 5]
        reward = np.clip(reward, -2.0, 10.0)
        
        return float(reward)
    
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
        # Switch between reward functions here
        return self._calculate_reward_long_term(action, done)  # Using improved long-term reward
        # return self._calculate_reward_total_return_only(action, done)  # Original reward
    
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

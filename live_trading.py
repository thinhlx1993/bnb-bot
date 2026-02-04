"""
Live Trading Module for Binance Testnet
This module executes trades on Binance testnet based on the strategy signals.

IMPORTANT: This is for TESTNET only. Never use real API keys in production without proper security.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
import logging

# Timezone configuration - UTC+7 (local time)
LOCAL_TIMEZONE = timezone(timedelta(hours=7))

# Add FinRL-Meta to sys.path
finrl_meta_path = Path("/mnt/data/FinRL-Tutorials/3-Practical/FinRL-Meta")
if str(finrl_meta_path) not in sys.path:
    sys.path.insert(0, str(finrl_meta_path))

import pandas as pd
import numpy as np
import requests
import hmac
import hashlib
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from binance.client import Client
from binance.exceptions import BinanceAPIException
from meta.data_processor import DataProcessor
from meta.data_processors._base import DataSource

# Import strategy functions from backtest (risk management, EMA, etc.)
from backtest import (
    calculate_ema, apply_risk_management,
    USE_MA99_FILTER, MA99_PERIOD, ENABLE_RISK_MANAGEMENT,
    BULLISH_EMA_FAST, BULLISH_EMA_SLOW,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_HOLDING_PERIODS,
    USE_STOP_LOSS, USE_TAKE_PROFIT, USE_MAX_HOLDING,
    INITIAL_BALANCE
)
# Shared entry/exit signal generator (same as evaluate and train when USE_BACKTEST_SIGNALS)
from entry_signal_generator import get_strategy_signals

# Import RL risk management
from rl_risk_management import RLRiskManager
from rl_risk_env import RiskManagementEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Trading Configuration
TESTNET = True  # Set to False for production (NOT RECOMMENDED without thorough testing)
TRADING_ENABLED = True  # Set to False to run in paper trading mode (no actual orders)
MIN_TRADE_AMOUNT = 10.0  # Minimum trade amount in USDT
TRADE_PERCENTAGE = 0.95  # Percentage of available balance to use per trade (95% to leave some buffer)
MAX_POSITION_SIZE = 100.0  # Maximum position size in USDT

# Binance Testnet Configuration
BINANCE_TESTNET_BASE_URL = "https://testnet.binance.vision"
BINANCE_TESTNET_API_URL = "https://testnet.binance.vision/api"


class BinanceTrader:
    """Handles live trading on Binance testnet.
    
    Supports two authentication methods:
    1. HMAC (API secret) - traditional method
    2. RSA (private key file) - more secure method
    """
    
    def __init__(self, api_key: str, api_secret: Optional[str] = None, 
                 private_key_path: Optional[str] = None, testnet: bool = True):
        """
        Initialize Binance trader.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret (for HMAC authentication)
            private_key_path: Path to RSA private key PEM file (for RSA authentication)
            testnet: Whether to use testnet (default: True)
        
        Note: Either api_secret OR private_key_path must be provided, not both.
        """
        self.testnet = testnet
        self.api_key = api_key
        self.api_secret = api_secret
        self.private_key_path = private_key_path
        self.auth_method = None
        
        # Determine authentication method
        if private_key_path and Path(private_key_path).exists():
            self.auth_method = 'RSA'
            try:
                with open(private_key_path, 'rb') as f:
                    self.private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=default_backend()
                    )
                logger.info(f"Using RSA authentication with private key: {private_key_path}")
            except Exception as e:
                logger.error(f"Failed to load private key: {e}")
                raise
        elif api_secret:
            self.auth_method = 'HMAC'
            logger.info("Using HMAC authentication with API secret")
        else:
            raise ValueError("Either api_secret or private_key_path must be provided")
        
        # Set base URL
        if testnet:
            self.base_url = "https://testnet.binance.vision"
            logger.info("Connected to Binance TESTNET")
        else:
            self.base_url = "https://api.binance.com"
            logger.warning("Connected to Binance PRODUCTION - BE CAREFUL!")
        
        # Initialize Binance client for read-only operations (uses HMAC if available)
        if api_secret:
            try:
                if testnet:
                    self.client = Client(api_key=api_key, api_secret=api_secret, testnet=True)
                else:
                    self.client = Client(api_key=api_key, api_secret=api_secret)
            except Exception as e:
                logger.warning(f"Could not initialize Binance client: {e}")
                self.client = None
        else:
            self.client = None
        
        # Track positions
        self.positions = {}  # {symbol: {'entry_price': float, 'entry_time': datetime, 'quantity': float, 'usdt_value': float}}
        self.trade_history = []
    
    def print_positions(self):
        """Print current positions with detailed information."""
        if not self.positions:
            logger.info("" + "="*60)
            logger.info("📊 CURRENT POSITIONS: None")
            logger.info("="*60)
            return
        
        logger.info("" + "="*60)
        logger.info("📊 CURRENT POSITIONS")
        logger.info("="*60)
        
        total_pnl_usdt = 0.0
        total_value_usdt = 0.0
        
        for symbol, pos in self.positions.items():
            current_price = self.get_current_price(symbol)
            entry_price = pos['entry_price']
            quantity = pos['quantity']
            entry_time = pos['entry_time']
            entry_usdt_value = pos.get('usdt_value', entry_price * quantity)
            
            # Calculate PnL
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            pnl_usdt = (current_price - entry_price) * quantity
            current_value = current_price * quantity
            
            # Calculate holding time
            holding_time = datetime.now() - entry_time
            hours = holding_time.total_seconds() / 3600
            minutes = (holding_time.total_seconds() % 3600) / 60
            
            # Accumulate totals
            total_pnl_usdt += pnl_usdt
            total_value_usdt += current_value
            
            # Format PnL with color indicators
            pnl_sign = "+" if pnl_pct >= 0 else ""
            pnl_indicator = "🟢" if pnl_pct >= 0 else "🔴"
            
            logger.info(f"{symbol}:")
            logger.info(f"  Quantity:     {quantity:.8f}")
            logger.info(f"  Entry Price:  ${entry_price:.2f}")
            logger.info(f"  Current Price: ${current_price:.2f}")
            logger.info(f"  Entry Value:  ${entry_usdt_value:.2f}")
            logger.info(f"  Current Value: ${current_value:.2f}")
            logger.info(f"  PnL:          {pnl_indicator} {pnl_sign}{pnl_pct:.2f}% (${pnl_sign}{pnl_usdt:.2f})")
            logger.info(f"  Holding Time: {int(hours)}h {int(minutes)}m")
            # Convert entry_time to UTC+7 for display
            if isinstance(entry_time, datetime):
                if entry_time.tzinfo is None:
                    entry_time_utc7 = entry_time.replace(tzinfo=timezone.utc).astimezone(LOCAL_TIMEZONE)
                else:
                    entry_time_utc7 = entry_time.astimezone(LOCAL_TIMEZONE)
                logger.info(f"  Entry Time:   {entry_time_utc7.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
            else:
                logger.info(f"  Entry Time:   {entry_time}")
        
        # Print summary
        total_pnl_sign = "+" if total_pnl_usdt >= 0 else ""
        total_pnl_indicator = "🟢" if total_pnl_usdt >= 0 else "🔴"
        logger.info("" + "-"*60)
        logger.info("SUMMARY:")
        logger.info(f"  Total Positions: {len(self.positions)}")
        logger.info(f"  Total Value:     ${total_value_usdt:.2f}")
        logger.info(f"  Total PnL:       {total_pnl_indicator} {total_pnl_sign}{total_pnl_usdt:.2f} USDT")
        logger.info("="*60)
    
    def _sign_request_hmac(self, params: dict) -> str:
        """Sign request using HMAC-SHA256."""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _sign_request_rsa(self, params: dict) -> str:
        """Sign request using RSA private key."""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature_bytes = self.private_key.sign(
            query_string.encode('utf-8'),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        signature = base64.b64encode(signature_bytes).decode('utf-8')
        return signature
    
    def _sign_request(self, params: dict) -> str:
        """Sign request using the configured authentication method."""
        if self.auth_method == 'RSA':
            return self._sign_request_rsa(params)
        elif self.auth_method == 'HMAC':
            return self._sign_request_hmac(params)
        else:
            raise ValueError(f"Unknown authentication method: {self.auth_method}")
        
    def _make_api_request(self, method: str, endpoint: str, params: dict = None) -> Optional[Dict]:
        """Make an authenticated API request."""
        if params is None:
            params = {}
        
        # Add timestamp
        params['timestamp'] = int(time.time() * 1000)
        
        # Sign request
        signature = self._sign_request(params)
        params['signature'] = signature
        
        # Make request
        url = f"{self.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self.api_key}
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    logger.error(f"Error response: {error_data}")
                except:
                    logger.error(f"Error response text: {e.response.text}")
            return None
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balances."""
        try:
            if self.client:
                # Use client if available (HMAC)
                account = self.client.get_account()
            else:
                # Use direct API call (RSA)
                account = self._make_api_request('GET', '/api/v3/account')
                if not account:
                    return {}
            
            balances = {}
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                if free > 0 or locked > 0:
                    balances[asset] = {'free': free, 'locked': locked, 'total': free + locked}
            return balances
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            if self.client:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
            else:
                # Public endpoint, no auth needed
                url = f"{self.base_url}/api/v3/ticker/price"
                response = requests.get(url, params={'symbol': symbol}, timeout=10)
                response.raise_for_status()
                ticker = response.json()
            
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        Fetch klines (candlestick data) from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '15m', '1h')
            limit: Number of candles to fetch (max 1000)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            df = df.set_index('timestamp')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        """
        Place a market order.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            quantity: Quantity to trade
        
        Returns:
            Order result or None if failed
        """
        if not TRADING_ENABLED:
            logger.info(f"[PAPER TRADING] Would {side} {quantity} {symbol}")
            return {'status': 'PAPER_TRADE', 'side': side, 'quantity': quantity, 'symbol': symbol}
        
        try:
            if self.client:
                # Use client if available (HMAC)
                if side == 'BUY':
                    order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
                else:
                    order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
            else:
                # Use direct API call (RSA)
                params = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'MARKET',
                    'quantity': quantity
                }
                order = self._make_api_request('POST', '/api/v3/order', params)
            
            if order:
                price = order.get('fills', [{}])[0].get('price', 'N/A') if isinstance(order, dict) else 'N/A'
                logger.info(f"Order placed: {side} {quantity} {symbol} at {price}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def calculate_trade_quantity(self, symbol: str, usdt_amount: float) -> float:
        """
        Calculate trade quantity based on USDT amount.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            usdt_amount: Amount in USDT to trade
        
        Returns:
            Quantity in base currency
        """
        try:
            # Get current price
            price = self.get_current_price(symbol)
            if price == 0:
                return 0.0
            
            # Get symbol info for precision
            if self.client:
                exchange_info = self.client.get_exchange_info()
            else:
                # Public endpoint, no auth needed
                url = f"{self.base_url}/api/v3/exchangeInfo"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                exchange_info = response.json()
            
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found")
                return 0.0
            
            # Get step size and precision
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                precision = len(str(step_size).rstrip('0').split('.')[-1])
            else:
                precision = 8  # Default precision
            
            # Calculate quantity
            quantity = usdt_amount / price
            
            # Round down to step size
            if lot_size_filter:
                quantity = (quantity // step_size) * step_size
            
            # Round to precision
            quantity = round(quantity, precision)
            
            return quantity
        except Exception as e:
            logger.error(f"Error calculating trade quantity: {e}")
            return 0.0
    
    def buy(self, symbol: str, usdt_amount: Optional[float] = None) -> bool:
        """
        Execute a buy order.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            usdt_amount: Amount in USDT to spend (None = use available balance)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get available balance
            balances = self.get_account_balance()
            usdt_balance = balances.get('USDT', {}).get('free', 0.0)
            
            if usdt_amount is None:
                usdt_amount = usdt_balance * TRADE_PERCENTAGE
            
            if usdt_amount < MIN_TRADE_AMOUNT:
                logger.warning(f"Insufficient balance: {usdt_amount} USDT < {MIN_TRADE_AMOUNT} USDT")
                return False
            
            if usdt_amount > MAX_POSITION_SIZE:
                usdt_amount = MAX_POSITION_SIZE
                logger.info(f"Capping trade size to {MAX_POSITION_SIZE} USDT")
            
            # Calculate quantity
            quantity = self.calculate_trade_quantity(symbol, usdt_amount)
            
            if quantity == 0:
                logger.error(f"Could not calculate quantity for {symbol}")
                return False
            
            # Place order
            order = self.place_market_order(symbol, 'BUY', quantity)
            
            if order:
                # Update position tracking
                entry_price = self.get_current_price(symbol)
                self.positions[symbol] = {
                    'entry_price': entry_price,
                    'entry_time': datetime.now(),
                    'quantity': quantity,
                    'usdt_value': usdt_amount
                }
                
                # Log trade
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': quantity,
                    'price': entry_price,
                    'usdt_value': usdt_amount
                })
                
                logger.info(f"BUY executed: {quantity} {symbol} at ${entry_price:.2f}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error executing buy: {e}")
            return False
    
    def sell(self, symbol: str, quantity: Optional[float] = None) -> bool:
        """
        Execute a sell order.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            quantity: Quantity to sell (None = sell all)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get position
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            if quantity is None:
                quantity = position['quantity']
            
            # Place order
            order = self.place_market_order(symbol, 'SELL', quantity)
            
            if order:
                # Get exit price
                exit_price = self.get_current_price(symbol)
                entry_price = position['entry_price']
                
                # Calculate PnL
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                pnl_usdt = (exit_price - entry_price) * quantity
                
                # Log trade
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': 'SELL',
                    'quantity': quantity,
                    'price': exit_price,
                    'entry_price': entry_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usdt': pnl_usdt
                })
                
                logger.info(f"SELL executed: {quantity} {symbol} at ${exit_price:.2f}")
                logger.info(f"PnL: {pnl_pct:.2f}% (${pnl_usdt:.2f})")
                
                # Remove position
                del self.positions[symbol]
                
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error executing sell: {e}")
            return False
    
    def check_risk_management(self, symbol: str) -> Optional[str]:
        """
        Check if risk management conditions are met for exiting a position.
        Uses rule-based risk management as fallback.
        
        Returns:
            'STOP_LOSS', 'TAKE_PROFIT', 'MAX_HOLDING', or None
        """
        if symbol not in self.positions:
            return None
        
        if not ENABLE_RISK_MANAGEMENT:
            return None
        
        position = self.positions[symbol]
        current_price = self.get_current_price(symbol)
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        
        # Calculate returns
        returns = (current_price - entry_price) / entry_price
        
        # Check stop loss
        if USE_STOP_LOSS and returns <= STOP_LOSS_PCT:
            return 'STOP_LOSS'
        
        # Check take profit
        if USE_TAKE_PROFIT and returns >= TAKE_PROFIT_PCT:
            return 'TAKE_PROFIT'
        
        # Check max holding period
        if USE_MAX_HOLDING:
            # Convert MAX_HOLDING_PERIODS to actual time based on interval
            # This is a simplified version - you may need to adjust based on your interval
            holding_time = datetime.now() - entry_time
            # Assuming 15m intervals, adjust as needed
            max_holding_time = timedelta(minutes=MAX_HOLDING_PERIODS * 15)
            if holding_time >= max_holding_time:
                return 'MAX_HOLDING'
        
        return None
    
    def check_rl_agent_decision(
        self, 
        symbol: str, 
        price_history: pd.Series,
        balance_history: pd.Series,
        rl_manager: Optional[RLRiskManager] = None
    ) -> bool:
        """
        Query RL agent to decide whether to hold or close a position.
        
        Args:
            symbol: Trading pair symbol
            price_history: Historical price series (should include entry point and current)
            balance_history: Historical balance series
            rl_manager: RLRiskManager instance (if None, will use rule-based fallback)
        
        Returns:
            True if RL agent wants to close, False if hold
        """
        if symbol not in self.positions:
            return False
        
        if rl_manager is None:
            # Fallback to rule-based risk management
            return self.check_risk_management(symbol) is not None
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        
        # Find entry index in price_history
        # We need to find where entry_time matches in price_history index
        entry_idx = None
        for idx, timestamp in enumerate(price_history.index):
            if isinstance(timestamp, pd.Timestamp):
                # Compare timestamps (allow some tolerance)
                time_diff = abs((timestamp - entry_time).total_seconds())
                if time_diff < 900:  # Within 15 minutes (one candle)
                    entry_idx = idx
                    break
        
        if entry_idx is None:
            logger.warning(f"Could not find entry point in price history for {symbol}, using rule-based fallback")
            return self.check_risk_management(symbol) is not None
        
        # Ensure we have enough data
        history_length = rl_manager.history_length
        current_idx = len(price_history) - 1
        
        if current_idx - entry_idx < 1:
            logger.warning(f"Not enough price history for {symbol} (need at least 1 period after entry)")
            return self.check_risk_management(symbol) is not None
        
        # Ensure we have enough history before entry
        window_start = max(0, entry_idx - history_length)
        price_window = price_history.iloc[window_start:current_idx + 1].copy()
        balance_window = balance_history.iloc[window_start:current_idx + 1].copy()
        
        # Ensure balance_window has same length as price_window
        if len(balance_window) != len(price_window):
            # Pad or trim balance_window
            if len(balance_window) < len(price_window):
                # Pad with last value
                last_balance = balance_window.iloc[-1] if len(balance_window) > 0 else INITIAL_BALANCE
                padding = pd.Series([last_balance] * (len(price_window) - len(balance_window)), 
                                   index=price_window.index[len(balance_window):])
                balance_window = pd.concat([balance_window, padding])
            else:
                balance_window = balance_window.iloc[:len(price_window)]
        
        # Create exit signals (none for now, agent will decide)
        exit_signals = pd.Series(False, index=price_window.index)
        
        # Create environment
        all_tickers_data = {
            'single_ticker': {
                'price': price_window,
                'balance': balance_window,
                'entry_signals': pd.Series(False, index=price_window.index),
                'exit_signals': exit_signals
            }
        }
        
        env = RiskManagementEnv(
            all_tickers_data=all_tickers_data,
            initial_balance=rl_manager.initial_balance,
            history_length=history_length,
            max_steps=rl_manager.max_steps,
            fee_rate=rl_manager.fee_rate
        )
        
        # Manually set entry point
        env.current_ticker = 'single_ticker'
        env.price_data = price_window
        env.balance_data = balance_window
        env.exit_signals = exit_signals
        env.entry_price = entry_price
        env.entry_idx = entry_idx - window_start
        env.exit_signal_idx = None  # No exit signal, agent decides
        env.exit_idx = len(price_window) - 1
        
        # Set up episode windows
        env.episode_start_idx = env.entry_idx
        env.episode_end_idx = env.exit_idx
        env.episode_length = env.exit_idx - env.episode_start_idx + 1
        env.data_start_idx = max(0, env.entry_idx - env.history_length)
        env.price_window = price_window.iloc[env.data_start_idx:env.exit_idx + 1].copy()
        env.balance_window = balance_window.iloc[env.data_start_idx:env.exit_idx + 1].copy()
        
        # Reset environment state
        env.current_step = 0
        env.current_idx = current_idx - window_start  # Current position in window (relative to episode start)
        env.position_open = True
        env.peak_balance = rl_manager.initial_balance
        env.max_drawdown = 0.0
        env.total_reward = 0.0
        env.episode_return = 0.0
        env.returns_history = []
        env.wins = 0
        env.total_trades = 0
        env.entry_balance = rl_manager.initial_balance
        
        # Track max/min prices from entry point
        entry_idx_in_window = env.entry_idx
        prices_since_entry = price_window.iloc[entry_idx_in_window:env.current_idx + 1].values
        if len(prices_since_entry) > 0:
            env.episode_max_price = float(np.max(prices_since_entry))
            env.episode_min_price = float(np.min(prices_since_entry))
        else:
            env.episode_max_price = entry_price
            env.episode_min_price = entry_price
        
        try:
            # Get observation for current state
            obs = env._get_observation()
            
            # Query RL agent
            action, _ = rl_manager.model.predict(obs, deterministic=True)
            
            # Action 0 = Hold, Action 1 = Close
            return action == 1
        except Exception as e:
            logger.error(f"Error getting RL agent decision for {symbol}: {e}")
            logger.warning("Falling back to rule-based risk management")
            return self.check_risk_management(symbol) is not None


def fetch_live_data(trader: BinanceTrader, ticker_list: list, time_interval: str, lookback_periods: int = 500) -> pd.DataFrame:
    """
    Fetch live data directly from Binance API (real-time).
    
    Args:
        trader: BinanceTrader instance with API access
        ticker_list: List of tickers (e.g., ['BTCUSDT'])
        time_interval: Time interval (e.g., '15m')
        lookback_periods: Number of periods to fetch (max 1000)
    
    Returns:
        DataFrame with OHLCV data in FinRL format
    """
    all_data = []
    
    try:
        for ticker in ticker_list:
            # Fetch klines directly from Binance (real-time data)
            if trader.client:
                # Use client if available (HMAC)
                klines = trader.client.get_klines(symbol=ticker, interval=time_interval, limit=lookback_periods)
            else:
                # Use direct API call (RSA) - public endpoint, no auth needed
                url = f"{trader.base_url}/api/v3/klines"
                params = {
                    'symbol': ticker,
                    'interval': time_interval,
                    'limit': min(lookback_periods, 1000)  # Binance max is 1000
                }
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                klines = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['time'] = df['timestamp']  # FinRL format uses 'time'
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['tic'] = ticker  # FinRL format uses 'tic' for ticker
            
            # Select columns in FinRL format
            df = df[['time', 'tic', 'open', 'high', 'low', 'close', 'volume']]
            
            all_data.append(df)
        
        if not all_data:
            logger.error("No data fetched from Binance")
            return pd.DataFrame()
        
        # Combine all tickers
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Log latest candle info (convert to UTC+7)
        latest_time = combined_df['time'].max()
        # Convert to UTC+7 if timezone-aware, otherwise assume UTC
        if isinstance(latest_time, pd.Timestamp):
            if latest_time.tz is None:
                latest_time_utc7 = latest_time.tz_localize('UTC').tz_convert('Asia/Bangkok')
            else:
                latest_time_utc7 = latest_time.tz_convert('Asia/Bangkok')
        else:
            latest_time_utc7 = pd.to_datetime(latest_time).tz_localize('UTC').tz_convert('Asia/Bangkok')
        
        min_time = combined_df['time'].min()
        if isinstance(min_time, pd.Timestamp):
            if min_time.tz is None:
                min_time_utc7 = min_time.tz_localize('UTC').tz_convert('Asia/Bangkok')
            else:
                min_time_utc7 = min_time.tz_convert('Asia/Bangkok')
        else:
            min_time_utc7 = pd.to_datetime(min_time).tz_localize('UTC').tz_convert('Asia/Bangkok')
        
        max_time = combined_df['time'].max()
        if isinstance(max_time, pd.Timestamp):
            if max_time.tz is None:
                max_time_utc7 = max_time.tz_localize('UTC').tz_convert('Asia/Bangkok')
            else:
                max_time_utc7 = max_time.tz_convert('Asia/Bangkok')
        else:
            max_time_utc7 = pd.to_datetime(max_time).tz_localize('UTC').tz_convert('Asia/Bangkok')
        
        logger.info(f"📊 Fetched data: Latest candle at {latest_time_utc7.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
        logger.info(f"📊 Current time: {datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')} (Local UTC+7)")
        logger.info(f"📊 Data range: {min_time_utc7.strftime('%Y-%m-%d %H:%M:%S')} to {max_time_utc7.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error fetching live data from Binance: {e}")
        logger.error(f"Falling back to FinRL-Meta data source...")
        
        # Fallback to FinRL-Meta if direct API fails
        try:
            end_date = datetime.now()
            if 'm' in time_interval:
                minutes = int(time_interval.replace('m', ''))
                start_date = end_date - timedelta(minutes=minutes * lookback_periods)
            elif 'h' in time_interval:
                hours = int(time_interval.replace('h', ''))
                start_date = end_date - timedelta(hours=hours * lookback_periods)
            elif 'd' in time_interval:
                days = int(time_interval.replace('d', ''))
                start_date = end_date - timedelta(days=days * lookback_periods)
            else:
                start_date = end_date - timedelta(days=30)
            
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            dp = DataProcessor(
                data_source=DataSource.binance,
                start_date=start_date_str,
                end_date=end_date_str,
                time_interval=time_interval
            )
            dp.download_data(ticker_list=ticker_list)
            dp.clean_data()
            return dp.dataframe
        except Exception as e2:
            logger.error(f"Error with FinRL-Meta fallback: {e2}")
            return pd.DataFrame()


def generate_signals(df: pd.DataFrame, ticker: str, strategy: str = "Combined") -> Tuple[pd.Series, pd.Series]:
    """
    Generate buy/sell signals for a ticker using shared entry_signal_generator.
    Same logic as evaluate_rl_agent and train_rl_agent (backtest divergence strategies).

    Args:
        df: DataFrame with OHLCV data
        ticker: Ticker symbol
        strategy: Strategy name (default "Combined")

    Returns:
        entries, exits: Boolean series for entry and exit signals
    """
    ticker_df = df[df['tic'] == ticker].copy()

    if ticker_df.empty:
        return pd.Series(), pd.Series()

    ticker_df['time'] = pd.to_datetime(ticker_df['time'])
    ticker_df = ticker_df.set_index('time').sort_index()

    price = ticker_df['close']
    return get_strategy_signals(ticker_df, price, strategy=strategy)


def run_live_trading(
    api_key: str,
    api_secret: Optional[str] = None,
    private_key_path: Optional[str] = None,
    ticker_list: list = None,
    time_interval: str = "15m",
    check_interval_seconds: int = 15,
    use_rl_agent: bool = True,
    rl_model_path: Optional[Path] = None,
    rl_model_name: str = "best_model"
):
    """
    Run live trading loop.
    
    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        private_key_path: Path to RSA private key (for RSA auth)
        ticker_list: List of tickers to trade
        time_interval: Time interval for strategy (e.g., '15m')
        check_interval_seconds: How often to check for signals (in seconds)
            - Recommended: 15-30 seconds for 15m candles
            - Too frequent (< 10s): More API calls, potential rate limiting
            - Too slow (> 60s): Might miss signals or risk management triggers
        use_rl_agent: Whether to use RL agent for hold/close decisions (default: True)
        rl_model_path: Path to RL model directory (default: models/rl_agent)
        rl_model_name: RL model filename without .zip extension (default: best_model)
    """
    if ticker_list is None:
        ticker_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "SOLUSDT"]
    
    logger.info("="*60)
    logger.info("Starting Live Trading Bot")
    logger.info("="*60)
    logger.info(f"Tickers: {ticker_list}")
    logger.info(f"Interval: {time_interval}")
    logger.info(f"Check Interval: {check_interval_seconds} seconds")
    logger.info(f"Trading Enabled: {TRADING_ENABLED}")
    logger.info(f"Testnet: {TESTNET}")
    logger.info(f"RL Agent: {'Enabled' if use_rl_agent else 'Disabled (using rule-based)'}")
    logger.info("")
    logger.info(f"Note: Checking every {check_interval_seconds}s allows quick detection of:")
    logger.info(f"  - New candle signals (candles update every {time_interval})")
    logger.info(f"  - RL agent hold/close decisions (if enabled)")
    logger.info(f"  - Risk management triggers (stop loss, take profit)")
    
    # Initialize trader
    trader = BinanceTrader(
        api_key=api_key,
        api_secret=api_secret,
        private_key_path=private_key_path,
        testnet=TESTNET
    )
    
    # Load RL agent if enabled
    rl_manager = None
    if use_rl_agent:
        try:
            if rl_model_path is None:
                rl_model_path = Path("models/rl_agent")
            
            logger.info(f"Loading RL model from: {rl_model_path}/{rl_model_name}.zip")
            rl_manager = RLRiskManager(
                model_path=rl_model_path,
                model_name=rl_model_name,
                initial_balance=INITIAL_BALANCE,
                deterministic=True  # Use deterministic policy for live trading
            )
            logger.info(f"✅ RL model loaded successfully (device: {rl_manager.model.device})")
        except Exception as e:
            logger.error(f"❌ Failed to load RL model: {e}")
            logger.warning("Falling back to rule-based risk management")
            use_rl_agent = False
            rl_manager = None
    
    # Display initial balance
    balances = trader.get_account_balance()
    logger.info(f"Initial Balance: {balances}")
    
    # Display initial positions (if any)
    trader.print_positions()
    
    # Track last signal state to avoid duplicate trades
    last_signals = {ticker: {'entry': False, 'exit': False} for ticker in ticker_list}
    
    try:
        while True:
            logger.info(f"{'='*60}")
            logger.info(f"Checking signals at {datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
            logger.info(f"{'='*60}")
            
            # Fetch latest data (real-time from Binance)
            df = fetch_live_data(trader, ticker_list, time_interval)
            
            if df.empty:
                logger.warning("No data fetched, skipping this cycle")
                time.sleep(check_interval_seconds)
                continue
            
            # Process each ticker
            for ticker in ticker_list:
                logger.info(f"Processing {ticker}...")
                
                # Check if position exists
                if ticker in trader.positions:
                    # Get price history for RL agent decision
                    ticker_df = df[df['tic'] == ticker].copy()
                    if not ticker_df.empty:
                        ticker_df = ticker_df.set_index('time').sort_index()
                        price_history = ticker_df['close']
                        
                        # Create balance history (simplified - use current balance)
                        # In a real scenario, you'd track balance over time
                        current_balance = trader.get_account_balance().get('USDT', {}).get('total', INITIAL_BALANCE)
                        balance_history = pd.Series(current_balance, index=price_history.index)
                        
                        # Check RL agent decision if enabled
                        if use_rl_agent and rl_manager is not None:
                            try:
                                should_close = trader.check_rl_agent_decision(
                                    ticker, price_history, balance_history, rl_manager
                                )
                                if should_close:
                                    logger.info(f"🤖 RL Agent decision: CLOSE position for {ticker}")
                                    trader.sell(ticker)
                                    last_signals[ticker]['exit'] = True
                                    continue
                                else:
                                    logger.info(f"🤖 RL Agent decision: HOLD position for {ticker}")
                            except Exception as e:
                                logger.error(f"Error querying RL agent for {ticker}: {e}")
                                logger.warning("Falling back to rule-based risk management")
                                # Fall through to rule-based check
                        
                        # Fallback to rule-based risk management
                        risk_exit = trader.check_risk_management(ticker)
                        if risk_exit:
                            logger.warning(f"Risk management exit triggered: {risk_exit}")
                            trader.sell(ticker)
                            last_signals[ticker]['exit'] = True
                            continue
                
                # Generate signals
                entries, exits = generate_signals(df, ticker)
                
                if entries.empty or exits.empty:
                    logger.warning(f"No signals generated for {ticker}")
                    continue
                
                # Check latest signal (most recent)
                latest_entry = entries.iloc[-1] if len(entries) > 0 else False
                latest_exit = exits.iloc[-1] if len(exits) > 0 else False
                
                # Debug: Show signal status with timestamps
                total_buy_signals = entries.sum()
                total_sell_signals = exits.sum()
                
                # Find the most recent buy/sell signal timestamps (convert to UTC+7)
                if total_buy_signals > 0:
                    last_buy_time = entries[entries == True].index[-1] if entries.sum() > 0 else None
                    if last_buy_time is not None:
                        # Convert pandas Timestamp to UTC+7
                        if isinstance(last_buy_time, pd.Timestamp):
                            if last_buy_time.tz is None:
                                last_buy_time_utc7 = last_buy_time.tz_localize('UTC').tz_convert('Asia/Bangkok')
                            else:
                                last_buy_time_utc7 = last_buy_time.tz_convert('Asia/Bangkok')
                        else:
                            last_buy_time_utc7 = pd.to_datetime(last_buy_time).tz_localize('UTC').tz_convert('Asia/Bangkok')
                        logger.info(f"  📈 Last BUY signal at: {last_buy_time_utc7.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
                if total_sell_signals > 0:
                    last_sell_time = exits[exits == True].index[-1] if exits.sum() > 0 else None
                    if last_sell_time is not None:
                        # Convert pandas Timestamp to UTC+7
                        if isinstance(last_sell_time, pd.Timestamp):
                            if last_sell_time.tz is None:
                                last_sell_time_utc7 = last_sell_time.tz_localize('UTC').tz_convert('Asia/Bangkok')
                            else:
                                last_sell_time_utc7 = last_sell_time.tz_convert('Asia/Bangkok')
                        else:
                            last_sell_time_utc7 = pd.to_datetime(last_sell_time).tz_localize('UTC').tz_convert('Asia/Bangkok')
                        logger.info(f"  📉 Last SELL signal at: {last_sell_time_utc7.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
                
                # Show latest candle info (convert to UTC+7)
                latest_candle_time = entries.index[-1] if len(entries) > 0 else None
                if latest_candle_time is not None:
                    # Convert pandas Timestamp to UTC+7
                    if isinstance(latest_candle_time, pd.Timestamp):
                        if latest_candle_time.tz is None:
                            latest_candle_time_utc7 = latest_candle_time.tz_localize('UTC').tz_convert('Asia/Bangkok')
                        else:
                            latest_candle_time_utc7 = latest_candle_time.tz_convert('Asia/Bangkok')
                    else:
                        latest_candle_time_utc7 = pd.to_datetime(latest_candle_time).tz_localize('UTC').tz_convert('Asia/Bangkok')
                    logger.info(f"  📊 Latest candle: {latest_candle_time_utc7.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
                logger.info(f"  Signal Status: {total_buy_signals} buy signals, {total_sell_signals} sell signals in history")
                logger.info(f"  Latest Entry Signal (most recent candle): {latest_entry}")
                logger.info(f"  Last signal processed flag: entry={last_signals[ticker]['entry']}, exit={last_signals[ticker]['exit']}")
                
                # Execute buy signal
                if latest_entry and not last_signals[ticker]['entry']:
                    if ticker not in trader.positions:
                        logger.info(f"✅ BUY signal detected for {ticker} - Opening position...")
                        success = trader.buy(ticker)
                        if success:
                            last_signals[ticker]['entry'] = True
                            logger.info(f"✅ Position opened successfully for {ticker}")
                        else:
                            logger.error(f"❌ Failed to open position for {ticker}")
                    else:
                        logger.info(f"⚠️  BUY signal but position already exists for {ticker}")
                elif latest_entry and last_signals[ticker]['entry']:
                    logger.info(f"ℹ️  BUY signal still active for {ticker} (already processed)")
                elif not latest_entry:
                    logger.info(f"ℹ️  No active BUY signal for {ticker}")
                
                # Execute sell signal
                if latest_exit and not last_signals[ticker]['exit']:
                    if ticker in trader.positions:
                        logger.info(f"✅ SELL signal detected for {ticker} - Closing position...")
                        success = trader.sell(ticker)
                        if success:
                            last_signals[ticker]['exit'] = True
                            logger.info(f"✅ Position closed successfully for {ticker}")
                        else:
                            logger.error(f"❌ Failed to close position for {ticker}")
                    else:
                        logger.info(f"⚠️  SELL signal but no position for {ticker}")
                elif latest_exit and last_signals[ticker]['exit']:
                    logger.info(f"ℹ️  SELL signal still active for {ticker} (already processed)")
                elif not latest_exit:
                    logger.info(f"ℹ️  No active SELL signal for {ticker}")
                
                # Reset signal flags if signal changed
                if not latest_entry:
                    last_signals[ticker]['entry'] = False
                if not latest_exit:
                    last_signals[ticker]['exit'] = False
            
            # Display current positions (detailed)
            trader.print_positions()
            
            # Display balance
            balances = trader.get_account_balance()
            usdt_balance = balances.get('USDT', {}).get('total', 0.0)
            logger.info(f"💰 USDT Balance: ${usdt_balance:.2f}")
            
            # Wait before next check
            logger.info(f"Waiting {check_interval_seconds} seconds before next check...")
            time.sleep(check_interval_seconds)
    
    except KeyboardInterrupt:
        logger.info("Stopping trading bot...")
        
        # Close all positions
        logger.info("Closing all positions...")
        for symbol in list(trader.positions.keys()):
            trader.sell(symbol)
        
        # Display final balance
        balances = trader.get_account_balance()
        logger.info(f"Final Balance: {balances}")
        
        # Display trade history
        logger.info(f"Trade History ({len(trader.trade_history)} trades):")
        for trade in trader.trade_history:
            logger.info(f"  {trade}")
    
    except Exception as e:
        logger.error(f"Error in trading loop: {e}", exc_info=True)


if __name__ == "__main__":
    # Configuration - SET YOUR API KEYS HERE
    # Get testnet API keys from: https://testnet.binance.vision/
    # 
    # For RSA Authentication (Recommended):
    #   - Only need API_KEY (from Binance testnet)
    #   - Only need PRIVATE_KEY_PATH (your private key PEM file)
    #   - NO API_SECRET needed!
    #
    # For HMAC Authentication (Legacy):
    #   - Need API_KEY and API_SECRET
    #   - No private key needed
    from dotenv import load_dotenv
    load_dotenv()
    
    API_KEY = os.getenv("BINANCE_API_KEY", "your_testnet_api_key_here")
    API_SECRET = os.getenv("BINANCE_API_SECRET", None)  # Only needed for HMAC auth
    PRIVATE_KEY_PATH = os.getenv("BINANCE_PRIVATE_KEY_PATH", "test-prv-key.pem")  # Only needed for RSA auth
    
    # Trading configuration
    TICKER_LIST = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "SOLUSDT"]  # All tickers to trade
    TIME_INTERVAL = "15m"  # Strategy interval
    # Check interval: How often to check for signals and risk management
    # - For 15m candles: 15-30 seconds is good (catches new candles quickly)
    # - Too frequent (< 10s): More API calls, potential rate limiting
    # - Too slow (> 60s): Might miss signals or risk management triggers
    CHECK_INTERVAL_SECONDS = 60  # Check for signals every 60 seconds
    
    # RL Agent configuration
    USE_RL_AGENT = True  # Enable RL agent for hold/close decisions
    RL_MODEL_PATH = Path("models/rl_agent")  # Path to RL model directory
    RL_MODEL_NAME = "best_model_03012026"  # Model filename without .zip extension
    
    # Validate API key
    if API_KEY == "your_testnet_api_key_here":
        logger.error("Please set your Binance testnet API key!")
        logger.error("Get testnet keys from: https://testnet.binance.vision/")
        logger.error("")
        logger.error("For RSA Authentication (Recommended):")
        logger.error("  export BINANCE_API_KEY='your_api_key'")
        logger.error("  export BINANCE_PRIVATE_KEY_PATH='test-prv-key.pem'")
        logger.error("")
        logger.error("For HMAC Authentication (Legacy):")
        logger.error("  export BINANCE_API_KEY='your_api_key'")
        logger.error("  export BINANCE_API_SECRET='your_api_secret'")
        sys.exit(1)
    
    # Validate authentication method
    if not API_SECRET and not PRIVATE_KEY_PATH:
        logger.error("Authentication method not specified!")
        logger.error("")
        logger.error("For RSA Authentication (Recommended - NO API SECRET NEEDED):")
        logger.error("  export BINANCE_PRIVATE_KEY_PATH='test-prv-key.pem'")
        logger.error("")
        logger.error("For HMAC Authentication (Legacy):")
        logger.error("  export BINANCE_API_SECRET='your_api_secret'")
        sys.exit(1)
    
    # Validate private key file if using RSA
    if PRIVATE_KEY_PATH:
        if not Path(PRIVATE_KEY_PATH).exists():
            logger.error(f"Private key file not found: {PRIVATE_KEY_PATH}")
            logger.error("Make sure the path is correct and the file exists")
            sys.exit(1)
        logger.info(f"Using RSA authentication with private key: {PRIVATE_KEY_PATH}")
        logger.info("Note: No API secret needed for RSA authentication!")
    else:
        logger.info("Using HMAC authentication with API secret")
    
    # Run live trading
    run_live_trading(
        api_key=API_KEY,
        api_secret=API_SECRET,
        private_key_path=PRIVATE_KEY_PATH,
        ticker_list=TICKER_LIST,
        time_interval=TIME_INTERVAL,
        check_interval_seconds=CHECK_INTERVAL_SECONDS,
        use_rl_agent=USE_RL_AGENT,
        rl_model_path=RL_MODEL_PATH,
        rl_model_name=RL_MODEL_NAME
    )

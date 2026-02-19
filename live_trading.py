"""
Live Trading Module for Binance Testnet
This module executes trades on Binance testnet based on the strategy signals.

IMPORTANT: This is for TESTNET only. Never use real API keys in production without proper security.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
import logging

from dotenv import load_dotenv
load_dotenv()

# Timezone: all logic uses UTC. LOCAL_TIMEZONE is only for log output.
LOCAL_TIMEZONE = timezone(timedelta(hours=7))  # Logs only: show timestamps in your timezone

# Add FinRL-Meta to sys.path (configurable via FINRL_META_PATH in .env)
finrl_meta_path = Path(os.getenv("FINRL_META_PATH", "/mnt/data/FinRL-Tutorials/3-Practical/FinRL-Meta"))
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

# Import strategy functions from backtest (risk management, etc.)
from backtest import (
    ENABLE_RISK_MANAGEMENT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_HOLDING_PERIODS,
    USE_STOP_LOSS, USE_TAKE_PROFIT, USE_MAX_HOLDING,
    INITIAL_BALANCE
)
# Shared entry/exit signal generator (same as evaluate and train when USE_BACKTEST_SIGNALS)
from entry_signal_generator import get_strategy_signals

# Import RL risk management (same env config as train/evaluate for observation consistency)
from rl_risk_management import RLRiskManager
from rl_risk_env import RiskManagementEnv, ENV_DEFAULT_CONFIG

from live_trading_db import (
    init_db,
    insert_signal,
    insert_position,
    close_position,
    get_open_positions,
    reset_db,
)

# Configure logging - add FileHandler explicitly because basicConfig is a no-op
# when backtest/rl_risk_management (imported above) already configured the root logger
Path("logs").mkdir(parents=True, exist_ok=True)
_root = logging.getLogger()
_live_handler = logging.FileHandler('logs/live_trading.log', encoding='utf-8')
_live_handler.setLevel(logging.INFO)
_live_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
_root.addHandler(_live_handler)
logger = logging.getLogger(__name__)

# Trading Configuration
TESTNET = True  # Set to False for production (NOT RECOMMENDED without thorough testing)
TRADING_ENABLED = True  # Set to False to run in paper trading mode (no actual orders)
MIN_TRADE_AMOUNT = 10.0  # Minimum trade amount in USDT
TRADE_PERCENTAGE = 0.95  # Percentage of available balance to use per trade (95% to leave some buffer)
MAX_POSITION_SIZE = 100.0  # Maximum position size in USDT
MIN_TICKER_PRICE = 0.01  # Skip tickers with price below this (avoids very low-priced / dust pairs)

# Binance Testnet Configuration
BINANCE_TESTNET_BASE_URL = "https://testnet.binance.vision"
BINANCE_TESTNET_API_URL = "https://testnet.binance.vision/api"

# RL agent: ensure this many bars before entry so technical indicators have enough history
ENTRY_LOOKBACK_STEPS = 100


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
        # Cache exchange info to avoid rate limits (NOTIONAL check, etc.)
        self._exchange_info_cache: Optional[Dict] = None
        self._exchange_info_ts: float = 0.0
        self._exchange_info_ttl_sec: float = 300.0

    def _get_exchange_info(self) -> Optional[Dict]:
        """Fetch exchange info (cached for _exchange_info_ttl_sec)."""
        now = time.time()
        if self._exchange_info_cache is not None and (now - self._exchange_info_ts) < self._exchange_info_ttl_sec:
            return self._exchange_info_cache
        try:
            if self.client:
                self._exchange_info_cache = self.client.get_exchange_info()
            else:
                url = f"{self.base_url}/api/v3/exchangeInfo"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                self._exchange_info_cache = response.json()
            self._exchange_info_ts = now
            return self._exchange_info_cache
        except Exception as e:
            logger.debug(f"Failed to fetch exchange info: {e}")
            return self._exchange_info_cache  # use stale if any

    def _get_min_notional(self, symbol: str) -> float:
        """
        Minimum notional (quote asset, e.g. USDT) required for a market order.
        Returns 0.0 if not found (no check).
        """
        info = self._get_exchange_info()
        if not info:
            return 0.0
        symbol_info = next((s for s in info.get('symbols', []) if s.get('symbol') == symbol), None)
        if not symbol_info:
            return 0.0
        min_for_market = 0.0
        min_any = 0.0
        for f in symbol_info.get('filters', []):
            if f.get('filterType') not in ('NOTIONAL', 'MIN_NOTIONAL'):
                continue
            try:
                val = float(f.get('minNotional', 0) or 0)
            except (TypeError, ValueError):
                continue
            if val <= 0:
                continue
            if f.get('applyToMarket', False):
                min_for_market = val
                break
            if min_any == 0:
                min_any = val
        return min_for_market or min_any

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
            # Use actual value at entry (quantity * entry_price) so Entry Value and Current Value are comparable
            entry_value_actual = entry_price * quantity
            current_value = current_price * quantity
            
            # Calculate PnL from value change so it matches Entry Value vs Current Value
            pnl_pct = ((current_value - entry_value_actual) / entry_value_actual) * 100 if entry_value_actual else 0.0
            pnl_usdt = current_value - entry_value_actual
            
            # All times in UTC (server/DB). Normalize naive entry_time as UTC for comparison.
            entry_utc = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            holding_time = now_utc - entry_utc
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
            logger.info(f"  Entry Value:  ${entry_value_actual:.2f}")
            logger.info(f"  Current Value: ${current_value:.2f}")
            logger.info(f"  PnL:          {pnl_indicator} {pnl_sign}{pnl_pct:.2f}% (${pnl_sign}{pnl_usdt:.2f})")
            logger.info(f"  Holding Time: {int(hours)}h {int(minutes)}m")
            # Logs: show entry time in local timezone
            if isinstance(entry_time, datetime):
                entry_utc = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
                entry_local = entry_utc.astimezone(LOCAL_TIMEZONE)
                logger.info(f"  Entry Time:   {entry_local.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
            else:
                logger.info(f"  Entry Time:   {entry_time}")
        
        # Print summary (positions = on exchange, any balance; bot-tracked open count is in DB only)
        total_pnl_sign = "+" if total_pnl_usdt >= 0 else ""
        total_pnl_indicator = "🟢" if total_pnl_usdt >= 0 else "🔴"
        logger.info("" + "-"*60)
        logger.info("SUMMARY:")
        logger.info(f"  Total Positions (on exchange): {len(self.positions)}")
        logger.info(f"  Total Value:     ${total_value_usdt:.2f}")
        logger.info(f"  Total PnL:       {total_pnl_indicator} {total_pnl_sign}{total_pnl_usdt:.2f} USDT")
        logger.info("="*60)
    
    def _sign_query_string_hmac(self, query_string: str) -> str:
        """Sign a query string using HMAC-SHA256 (string must match exactly what is sent)."""
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _sign_query_string_rsa(self, query_string: str) -> str:
        """Sign a query string using RSA (string must match exactly what is sent)."""
        signature_bytes = self.private_key.sign(
            query_string.encode('utf-8'),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return base64.b64encode(signature_bytes).decode('utf-8')

    def _sign_query_string(self, query_string: str) -> str:
        """Sign a raw query string using the configured authentication method."""
        if self.auth_method == 'RSA':
            return self._sign_query_string_rsa(query_string)
        elif self.auth_method == 'HMAC':
            return self._sign_query_string_hmac(query_string)
        else:
            raise ValueError(f"Unknown authentication method: {self.auth_method}")

    def _make_api_request(self, method: str, endpoint: str, params: dict = None) -> Optional[Dict]:
        """Make an authenticated API request.
        Uses URL-encoded query string for signing so it matches the request (fixes symbols with non-ASCII chars).
        POST body is urlencoded to safely handle base64 signature (+, /).
        """
        if params is None:
            params = {}
        # Ensure all param values are strings for consistent encoding (urlencode converts int to str)
        params = {k: v if isinstance(v, str) else str(v) for k, v in dict(params).items()}
        params['timestamp'] = str(int(time.time() * 1000))
        # Sign the exact query string we will send (URL-encoded), so non-ASCII symbols work
        query_string_to_sign = urlencode(sorted(params.items()))
        signature = self._sign_query_string(query_string_to_sign)
        params['signature'] = signature

        url = f"{self.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self.api_key}

        try:
            if method.upper() == 'GET':
                query_string = urlencode(sorted(params.items()))
                full_url = f"{url}?{query_string}"
                response = requests.get(full_url, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                headers['Content-Type'] = 'application/x-www-form-urlencoded'
                body = urlencode(sorted(params.items()))
                response = requests.post(url, headers=headers, data=body, timeout=10)
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
    
    def get_my_trades(self, symbol: str, limit: int = 100) -> list:
        """Get trade history for a symbol (signed). Returns list of dicts with price, qty, time, isBuyer."""
        try:
            if self.client:
                trades = self.client.get_my_trades(symbol=symbol, limit=limit)
            else:
                data = self._make_api_request('GET', '/api/v3/myTrades', {'symbol': symbol, 'limit': limit})
                trades = data if data else []
            return trades
        except Exception as e:
            logger.debug(f"get_my_trades {symbol}: {e}")
            return []
    
    def sync_positions_from_exchange(self, ticker_list: list) -> None:
        """
        Set self.positions from exchange balances (source of truth).
        Only considers symbols in ticker_list. Entry price/time from recent buy trades (myTrades).
        DB is not read; use this instead of restoring from DB.
        """
        self.positions.clear()
        balances = self.get_account_balance()
        if not balances:
            return
        for symbol in ticker_list:
            base = symbol.replace('USDT', '')
            if base == symbol:
                continue
            total = balances.get(base, {}).get('total', 0) or 0
            if total <= 0:
                continue
            quantity = float(total)
            current_price = self.get_current_price(symbol)
            usdt_value = quantity * current_price if current_price else 0.0
            entry_price = current_price
            entry_time = datetime.now(timezone.utc)
            trades = self.get_my_trades(symbol, limit=200)
            buys = [t for t in trades if t.get('isBuyer') is True]
            if buys:
                buys_sorted = sorted(buys, key=lambda t: t['time'], reverse=True)
                cum_qty = 0.0
                cost_sum = 0.0
                earliest_time = None
                for t in buys_sorted:
                    qty = float(t['qty'])
                    price = float(t['price'])
                    cost_sum += qty * price
                    cum_qty += qty
                    if earliest_time is None or t['time'] < earliest_time:
                        earliest_time = t['time']
                    if cum_qty >= quantity:
                        break
                if cum_qty > 0:
                    entry_price = cost_sum / cum_qty
                    entry_time = datetime.fromtimestamp(earliest_time / 1000.0, tz=timezone.utc)
            self.positions[symbol] = {
                'entry_price': entry_price,
                'entry_time': entry_time,
                'quantity': quantity,
                'usdt_value': usdt_value,
            }
            logger.info(f"Synced position from exchange: {symbol} qty={quantity:.8f} entry≈{entry_price:.4f}")
    
    def get_all_usdt_pairs(self) -> list:
        """Fetch all USDT spot trading pairs from exchange (public endpoint)."""
        try:
            if self.client:
                exchange_info = self.client.get_exchange_info()
            else:
                url = f"{self.base_url}/api/v3/exchangeInfo"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                exchange_info = response.json()
            
            pairs = [
                s['symbol'] for s in exchange_info['symbols']
                if s.get('quoteAsset') == 'USDT'
                and s.get('status') == 'TRADING'
            ]
            return sorted(pairs)
        except Exception as e:
            logger.error(f"Error fetching USDT pairs: {e}")
            return []
    
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

        # Enforce NOTIONAL filter: order value must be >= min notional (avoids -1013)
        price = self.get_current_price(symbol)
        if price and price > 0:
            notional = quantity * price
            min_notional = self._get_min_notional(symbol)
            if min_notional > 0 and notional < min_notional:
                logger.info(
                    f"Order skipped (NOTIONAL): {symbol} {side} {notional:.2f} USDT < min {min_notional:.2f} USDT"
                )
                return None

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
            
            if order and isinstance(order, dict):
                fills = order.get('fills') or []
                fill_price = fills[0].get('price', 'N/A') if fills else 'N/A'
                logger.info(f"Order placed: {side} {quantity} {symbol} at {fill_price}")
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
                    'entry_time': datetime.now(timezone.utc),
                    'quantity': quantity,
                    'usdt_value': usdt_amount
                }
                
                # Log trade
                self.trade_history.append({
                    'timestamp': datetime.now(timezone.utc),
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
                    'timestamp': datetime.now(timezone.utc),
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

            # Order was skipped or failed (e.g. NOTIONAL filter). If below min notional, drop from tracking.
            price = self.get_current_price(symbol)
            min_notional = self._get_min_notional(symbol)
            if price and min_notional > 0 and (quantity * price) < min_notional:
                logger.info(
                    f"Dust removed from tracking: {symbol} ({(quantity * price):.2f} USDT < min {min_notional:.2f})"
                )
                del self.positions[symbol]
                return True  # so caller can close_position() and stop retrying
            
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
            entry_utc = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            holding_time = now_utc - entry_utc
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
        rl_manager: Optional[RLRiskManager] = None,
        ohlcv_df: Optional[pd.DataFrame] = None,
    ) -> bool:
        """
        Query RL agent to decide whether to hold or close a position.
        Uses same observation construction and VecNormalize as evaluate_rl_agent/train.

        Args:
            symbol: Trading pair symbol
            price_history: Historical price series (should include entry point and current)
            balance_history: Historical balance series
            rl_manager: RLRiskManager instance (if None, will use rule-based fallback)
            ohlcv_df: OHLCV DataFrame (same index as price_history; cols: open, high, low, close, volume)
                      for volume-based obs (OBV/MFI/AD/PVT). Must match train/eval.

        Returns:
            True if RL agent wants to close, False if hold
        """
        if symbol not in self.positions:
            return False

        if rl_manager is None:
            return self.check_risk_management(symbol) is not None

        position = self.positions[symbol]
        entry_price = position['entry_price']
        entry_time = position['entry_time']

        # All times in UTC. Normalize entry_time and index timestamps to UTC-aware for comparison.
        entry_utc = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
        def _to_utc_aware(ts):
            if isinstance(ts, pd.Timestamp):
                return ts.tz_localize(timezone.utc) if ts.tzinfo is None else ts.tz_convert(timezone.utc)
            # datetime
            return ts.replace(tzinfo=timezone.utc) if getattr(ts, 'tzinfo', None) is None else ts.astimezone(timezone.utc)

        entry_idx = None
        for idx, timestamp in enumerate(price_history.index):
            if isinstance(timestamp, pd.Timestamp):
                ts_utc = _to_utc_aware(timestamp)
                time_diff = abs((ts_utc - entry_utc).total_seconds())
                if time_diff < 900:
                    entry_idx = idx
                    break

        if entry_idx is None:
            logger.warning(f"Could not find entry point in price history for {symbol}, using rule-based fallback")
            return self.check_risk_management(symbol) is not None

        history_length = rl_manager.history_length
        current_idx = len(price_history) - 1

        # Use effective entry = entry - ENTRY_LOOKBACK_STEPS so we have enough bars before entry for indicators
        effective_entry_idx = max(0, entry_idx - ENTRY_LOOKBACK_STEPS)
        if current_idx - effective_entry_idx < 1:
            logger.warning(f"Not enough price history for {symbol} (need at least 1 period after effective entry)")
            return self.check_risk_management(symbol) is not None

        window_start = max(0, effective_entry_idx - history_length)
        price_window = price_history.iloc[window_start:current_idx + 1].copy()
        balance_window = balance_history.iloc[window_start:current_idx + 1].copy()

        if len(balance_window) != len(price_window):
            if len(balance_window) < len(price_window):
                last_balance = balance_window.iloc[-1] if len(balance_window) > 0 else INITIAL_BALANCE
                padding = pd.Series([last_balance] * (len(price_window) - len(balance_window)),
                                   index=price_window.index[len(balance_window):])
                balance_window = pd.concat([balance_window, padding])
            else:
                balance_window = balance_window.iloc[:len(price_window)]

        exit_signals = pd.Series(False, index=price_window.index)

        all_tickers_data = {
            'single_ticker': {
                'price': price_window,
                'balance': balance_window,
                'entry_signals': pd.Series(False, index=price_window.index),
                'exit_signals': exit_signals
            }
        }

        # OHLCV for volume-based observation features (OBV/MFI/AD/PVT) - must match training/eval
        if ohlcv_df is not None and len(ohlcv_df) >= current_idx + 1:
            ohlcv_full_window = ohlcv_df.iloc[window_start:current_idx + 1].copy()
        else:
            ohlcv_full_window = None

        env = RiskManagementEnv(
            all_tickers_data=all_tickers_data,
            initial_balance=rl_manager.initial_balance,
            history_length=history_length,
            max_steps=rl_manager.max_steps,
            obs_periods_norm_steps=ENV_DEFAULT_CONFIG["obs_periods_norm_steps"],
            fee_rate=rl_manager.fee_rate
        )

        env.current_ticker = 'single_ticker'
        env.price_data = price_window
        env.balance_data = balance_window
        env.exit_signals = exit_signals
        env.entry_price = entry_price
        env.entry_idx = effective_entry_idx - window_start
        env.exit_signal_idx = None
        env.exit_idx = len(price_window) - 1

        env.episode_start_idx = env.entry_idx
        env.episode_end_idx = env.exit_idx
        env.episode_length = env.exit_idx - env.episode_start_idx + 1
        env.data_start_idx = max(0, env.entry_idx - env.history_length)
        env.price_window = price_window.iloc[env.data_start_idx:env.exit_idx + 1].copy()
        env.balance_window = balance_window.iloc[env.data_start_idx:env.exit_idx + 1].copy()

        if ohlcv_full_window is not None and len(ohlcv_full_window) > env.exit_idx:
            env.ohlcv_window = ohlcv_full_window.iloc[env.data_start_idx:env.exit_idx + 1].copy()
        else:
            env.ohlcv_window = None

        env.current_step = 0
        env.current_idx = current_idx - window_start
        env.position_open = True
        env.peak_balance = rl_manager.initial_balance
        env.max_drawdown = 0.0
        env.total_reward = 0.0
        env.episode_return = 0.0
        env.returns_history = []
        env.wins = 0
        env.total_trades = 0
        env.entry_balance = rl_manager.initial_balance

        # Episode min/max use real entry (actual holding period), not effective entry
        real_entry_idx_in_window = entry_idx - window_start
        prices_since_entry = price_window.iloc[real_entry_idx_in_window:env.current_idx + 1].values
        if len(prices_since_entry) > 0:
            env.episode_max_price = float(np.max(prices_since_entry))
            env.episode_min_price = float(np.min(prices_since_entry))
        else:
            env.episode_max_price = entry_price
            env.episode_min_price = entry_price

        env._precomputed_indicators = env._precompute_indicators()

        try:
            obs = env._get_observation()

            if rl_manager.vec_normalize is not None:
                obs = rl_manager.vec_normalize.normalize_obs(obs.reshape(1, -1))[0]

            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            action, _ = rl_manager.model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=rl_manager.deterministic
            )
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
            if not klines:
                logger.debug("No data for %s, skipping", ticker)
                continue
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
            if len(df) == 0:
                logger.debug("No rows for %s, skipping", ticker)
                continue
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


def test_positions(
    api_key: str,
    api_secret: Optional[str] = None,
    private_key_path: Optional[str] = None,
    symbol: str = "BNBUSDT",
    usdt_amount: float = None,
):
    """
    Test open and close position by executing a buy-sell roundtrip on Binance testnet.
    Uses MIN_TRADE_AMOUNT (10 USDT) if usdt_amount not specified.
    """
    amount = usdt_amount if usdt_amount is not None and usdt_amount >= MIN_TRADE_AMOUNT else MIN_TRADE_AMOUNT
    
    logger.info("="*60)
    logger.info("POSITION TEST MODE - Buy then Sell roundtrip")
    logger.info("="*60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Amount: {amount} USDT")
    logger.info(f"Trading Enabled: {TRADING_ENABLED}")
    logger.info("")
    
    trader = BinanceTrader(
        api_key=api_key,
        api_secret=api_secret,
        private_key_path=private_key_path,
        testnet=TESTNET
    )
    
    init_db()
    
    # Show initial balance
    balances = trader.get_account_balance()
    usdt_before = balances.get('USDT', {}).get('total', 0.0)
    logger.info(f"USDT before: ${usdt_before:.2f}")
    
    # Step 1: Open position (BUY)
    logger.info("")
    logger.info("Step 1: Opening position (BUY)...")
    success_buy = trader.buy(symbol, usdt_amount=amount)
    
    if not success_buy:
        logger.error("FAILED: Buy order failed")
        return False
    
    pos = trader.positions.get(symbol)
    if not pos:
        logger.error("FAILED: Position not tracked after buy")
        return False
    
    logger.info(f"Position opened: {pos['quantity']} {symbol} @ ${pos['entry_price']:.4f}")
    
    # Persist to DB (same as live loop)
    insert_position(symbol, pos['entry_price'], pos['quantity'], pos['usdt_value'])
    
    # Brief pause
    time.sleep(2)
    
    # Step 2: Close position (SELL)
    logger.info("")
    logger.info("Step 2: Closing position (SELL)...")
    success_sell = trader.sell(symbol)
    
    if not success_sell:
        logger.error("FAILED: Sell order failed")
        return False
    
    if symbol in trader.positions:
        logger.error("FAILED: Position still tracked after sell")
        return False
    
    close_position(symbol, 'test')
    
    # Show final balance
    balances = trader.get_account_balance()
    usdt_after = balances.get('USDT', {}).get('total', 0.0)
    logger.info("")
    logger.info(f"USDT after:  ${usdt_after:.2f}")
    logger.info("="*60)
    logger.info("PASSED: Open and close position test completed successfully")
    logger.info("="*60)
    return True


def reset_testnet_account(
    api_key: str,
    api_secret: Optional[str] = None,
    private_key_path: Optional[str] = None,
    clear_local_db: bool = True,
) -> bool:
    """
    Reset testnet account: sell all open positions (convert to USDT) and optionally clear local DB.
    Note: Binance testnet does not provide an API to reset balance to a fixed amount; this only
    closes all positions so your balance is 100% USDT.
    """
    trader = BinanceTrader(
        api_key=api_key,
        api_secret=api_secret,
        private_key_path=private_key_path,
        testnet=TESTNET,
    )
    ticker_list = trader.get_all_usdt_pairs()
    if not ticker_list:
        logger.warning("Could not fetch USDT pairs, using default list")
        ticker_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "SOLUSDT"]
    trader.sync_positions_from_exchange(ticker_list)
    if not trader.positions:
        logger.info("No open positions on exchange.")
    else:
        logger.info(f"Closing {len(trader.positions)} position(s)...")
        for symbol in list(trader.positions.keys()):
            if trader.sell(symbol):
                close_position(symbol, "reset_account")
                logger.info(f"  Closed {symbol}")
            else:
                logger.warning(f"  Failed to close {symbol}")
    balances = trader.get_account_balance()
    usdt = balances.get("USDT", {}).get("total", 0.0)
    logger.info(f"USDT balance after reset: ${usdt:.2f}")
    if clear_local_db:
        reset_db()
        logger.info("Local DB cleared (signals and positions history).")
    return True


def run_live_trading(
    api_key: str,
    api_secret: Optional[str] = None,
    private_key_path: Optional[str] = None,
    ticker_list: list = None,
    time_interval: str = "15m",
    check_interval_seconds: int = 15,
    signal_lookback_candles: int = 6,
    use_rl_agent: bool = True,
    rl_model_path: Optional[Path] = None,
    rl_model_name: str = "best_model",
    close_on_strategy_sell: bool = True,
    close_on_rl_agent: bool = True,
    close_on_risk_management: bool = True,
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
        signal_lookback_candles: Number of most recent candles to consider for entry/exit.
            - 6 (default): Act if signal appeared in last 6 candles so 24/7 bot doesn't miss due to timing.
            - Use 1 for strict "only current candle" behavior.
        use_rl_agent: Whether to use RL agent for hold/close decisions (default: True)
        rl_model_path: Path to RL model directory (default: models/rl_agent)
        rl_model_name: RL model filename without .zip extension (default: best_model)
        close_on_strategy_sell: If True, close position when strategy generates SELL signal (default: True)
        close_on_rl_agent: If True, close position when RL agent says close (default: True)
        close_on_risk_management: If True, close on stop loss / take profit / max holding (default: True)
    """
    # Initialize trader first (needed to fetch exchange info when using all tickers)
    trader = BinanceTrader(
        api_key=api_key,
        api_secret=api_secret,
        private_key_path=private_key_path,
        testnet=TESTNET
    )
    
    if ticker_list is None:
        ticker_list = trader.get_all_usdt_pairs()
        if not ticker_list:
            logger.error("Failed to fetch USDT pairs from exchange, falling back to default list")
            ticker_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "SOLUSDT"]
        else:
            logger.info(f"Using all {len(ticker_list)} USDT pairs from exchange")
    
    logger.info("="*60)
    logger.info("Starting Live Trading Bot")
    logger.info("="*60)
    logger.info(f"Tickers: {len(ticker_list)} pairs" + (f" (first 10: {ticker_list[:10]}...)" if len(ticker_list) > 10 else f" - {ticker_list}"))
    logger.info(f"Interval: {time_interval}")
    logger.info(f"Check Interval: {check_interval_seconds} seconds")
    logger.info(f"Trading Enabled: {TRADING_ENABLED}")
    logger.info(f"Testnet: {TESTNET}")
    logger.info(f"RL Agent: {'Enabled' if use_rl_agent else 'Disabled (using rule-based)'}")
    logger.info(f"Close triggers: strategy_sell={close_on_strategy_sell}, rl_agent={close_on_rl_agent}, risk_management={close_on_risk_management}")
    logger.info("")
    logger.info(f"Note: Checking every {check_interval_seconds}s allows quick detection of:")
    logger.info(f"  - New candle signals (candles update every {time_interval})")
    logger.info(f"  - RL agent hold/close decisions (if enabled)")
    logger.info(f"  - Risk management triggers (stop loss, take profit)")
    
    # Load RL agent if enabled
    rl_manager = None
    if use_rl_agent:
        try:
            if rl_model_path is None:
                rl_model_path = Path("models/rl_agent")
            
            logger.info(f"Loading RL model from: {rl_model_path}/best_model/{rl_model_name}.zip")
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
    
    # DB: positions synced from exchange; ensure every exchange position has an open row in DB (bot-only).
    init_db()
    trader.sync_positions_from_exchange(ticker_list)
    open_in_db = {p["ticker"] for p in get_open_positions()}
    for symbol, pos in trader.positions.items():
        if symbol not in open_in_db:
            insert_position(symbol, pos["entry_price"], pos["quantity"], pos["usdt_value"])
            open_in_db.add(symbol)
            logger.info(f"Synced position to DB: {symbol} qty={pos['quantity']:.8f} entry≈{pos['entry_price']:.4f}")

    # Display initial balance
    balances = trader.get_account_balance()
    
    # Display initial positions (if any)
    trader.print_positions()
    
    # Track last signal state to avoid duplicate trades (entry=True for restored positions)
    last_signals = {ticker: {'entry': False, 'exit': False} for ticker in ticker_list}
    for ticker in trader.positions:
        last_signals[ticker]["entry"] = True
    
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
            tickers_with_data = df["tic"].unique().tolist()

            # Process each ticker (skip tickers that have no data this cycle)
            for ticker in ticker_list:
                if ticker not in tickers_with_data:
                    logger.debug("Skipping %s (no data this cycle)", ticker)
                    continue
                # Skip tickers with price < MIN_TICKER_PRICE (use latest close from data when available)
                ticker_df = df[df['tic'] == ticker]
                if not ticker_df.empty:
                    price = float(ticker_df['close'].iloc[-1])
                else:
                    price = trader.get_current_price(ticker)
                if price < MIN_TICKER_PRICE:
                    logger.debug(f"Skipping {ticker} (price {price:.4f} < {MIN_TICKER_PRICE})")
                    continue

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
                        
                        # Check RL agent decision if enabled and close_on_rl_agent
                        if use_rl_agent and close_on_rl_agent and rl_manager is not None:
                            try:
                                should_close = trader.check_rl_agent_decision(
                                    ticker, price_history, balance_history, rl_manager,
                                    ohlcv_df=ticker_df
                                )
                                if should_close:
                                    logger.info(f"🤖 RL Agent decision: CLOSE position for {ticker}")
                                    if trader.sell(ticker):
                                        close_position(ticker, 'rl')
                                        insert_signal(ticker, 'sell', 'rl', None)
                                    last_signals[ticker]['exit'] = True
                                    continue
                                else:
                                    logger.info(f"🤖 RL Agent decision: HOLD position for {ticker}")
                            except Exception as e:
                                logger.error(f"Error querying RL agent for {ticker}: {e}")
                                logger.warning("Falling back to rule-based risk management")
                                # Fall through to rule-based check
                        
                        # Rule-based risk management (stop loss / take profit / max holding)
                        if close_on_risk_management:
                            risk_exit = trader.check_risk_management(ticker)
                            if risk_exit:
                                logger.warning(f"Risk management exit triggered: {risk_exit}")
                                if trader.sell(ticker):
                                    close_position(ticker, 'risk_management')
                                    insert_signal(ticker, 'sell', 'risk_management', None)
                                last_signals[ticker]['exit'] = True
                                continue
                
                # Generate signals
                entries, exits = generate_signals(df, ticker)
                
                if entries.empty or exits.empty:
                    logger.warning(f"No signals generated for {ticker}")
                    continue
                
                # Debug: Show signal status with timestamps
                total_buy_signals = entries.sum()
                total_sell_signals = exits.sum()
                last_buy_time = None
                last_sell_time = None
                
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
                if total_sell_signals > 0 and last_sell_time is not None:
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
                # Active signal = most recent signal by time (BUY stays active until a SELL appears)
                latest_entry = (last_buy_time is not None) and (last_sell_time is None or last_buy_time > last_sell_time)
                latest_exit = (last_sell_time is not None) and (last_buy_time is None or last_sell_time > last_buy_time)
                logger.info(f"  Signal Status: {total_buy_signals} buy signals, {total_sell_signals} sell signals in history")
                logger.info(f"  Latest Entry Signal (most recent signal is BUY): {latest_entry}")
                logger.info(f"  Last signal processed flag: entry={last_signals[ticker]['entry']}, exit={last_signals[ticker]['exit']}")
                
                candle_time_iso = pd.Timestamp(latest_candle_time).isoformat() if latest_candle_time is not None else None
                
                # Execute buy signal: BUY active and no SELL from RL (we're not in position) → open
                if latest_entry and not last_signals[ticker]['entry']:
                    if ticker not in trader.positions:
                        logger.info(f"✅ BUY signal detected for {ticker} - Opening position...")
                        success = trader.buy(ticker)
                        if success:
                            pos = trader.positions[ticker]
                            insert_position(ticker, pos['entry_price'], pos['quantity'], pos['usdt_value'])
                            insert_signal(ticker, 'buy', 'strategy', candle_time_iso)
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
                
                # Execute sell signal (strategy) — only if close_on_strategy_sell
                if close_on_strategy_sell and latest_exit and not last_signals[ticker]['exit']:
                    if ticker in trader.positions:
                        logger.info(f"✅ SELL signal detected for {ticker} - Closing position...")
                        success = trader.sell(ticker)
                        if success:
                            close_position(ticker, 'strategy')
                            insert_signal(ticker, 'sell', 'strategy', candle_time_iso)
                            last_signals[ticker]['exit'] = True
                            logger.info(f"✅ Position closed successfully for {ticker}")
                        else:
                            logger.error(f"❌ Failed to close position for {ticker}")
                    else:
                        logger.info(f"⚠️  SELL signal but no position for {ticker}")
                elif close_on_strategy_sell and latest_exit and last_signals[ticker]['exit']:
                    logger.info(f"ℹ️  SELL signal still active for {ticker} (already processed)")
                elif not close_on_strategy_sell and latest_exit:
                    logger.info(f"ℹ️  SELL signal for {ticker} (close on strategy disabled)")
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
        
        # Display final balance - only traded tickers (USDT + base assets from ticker_list)
        balances = trader.get_account_balance()
        relevant_assets = {'USDT'}.union(
            t.replace('USDT', '') for t in ticker_list if t.endswith('USDT')
        )
        filtered_balances = {k: v for k, v in balances.items() if k in relevant_assets}
        logger.info(f"Final Balance ({len(filtered_balances)} assets): {filtered_balances}")
        
        # Display trade history
        logger.info(f"Trade History ({len(trader.trade_history)} trades):")
        for trade in trader.trade_history:
            logger.info(f"  {trade}")
    
    except Exception as e:
        logger.error(f"Error in trading loop: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Trading on Binance Testnet")
    parser.add_argument("--test-positions", action="store_true",
                        help="Test open/close position: run buy-sell roundtrip then exit")
    parser.add_argument("--test-symbol", type=str, default="BNBUSDT",
                        help="Symbol for --test-positions (default: BNBUSDT)")
    parser.add_argument("--test-amount", type=float, default=None,
                        help="USDT amount for --test-positions (default: MIN_TRADE_AMOUNT=10)")
    parser.add_argument("--reset-account", action="store_true",
                        help="Reset testnet account: sell all positions (to USDT) and clear local DB")
    args, _ = parser.parse_known_args()
    
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
    API_KEY = os.getenv("BINANCE_API_KEY", "your_testnet_api_key_here")
    API_SECRET = os.getenv("BINANCE_API_SECRET", None)  # Only needed for HMAC auth
    PRIVATE_KEY_PATH = os.getenv("BINANCE_PRIVATE_KEY_PATH", "test-prv-key.pem")  # Only needed for RSA auth
    
    # Run position test mode and exit
    if args.test_positions:
        if API_KEY == "your_testnet_api_key_here":
            logger.error("Please set BINANCE_API_KEY to run position test")
            sys.exit(1)
        if not API_SECRET and not PRIVATE_KEY_PATH:
            logger.error("Provide API_SECRET or BINANCE_PRIVATE_KEY_PATH")
            sys.exit(1)
        if PRIVATE_KEY_PATH and not Path(PRIVATE_KEY_PATH).exists():
            logger.error(f"Private key not found: {PRIVATE_KEY_PATH}")
            sys.exit(1)
        ok = test_positions(
            api_key=API_KEY,
            api_secret=API_SECRET,
            private_key_path=PRIVATE_KEY_PATH,
            symbol=args.test_symbol,
            usdt_amount=args.test_amount,
        )
        sys.exit(0 if ok else 1)
    
    # Reset testnet account: sell all positions and clear local DB
    if args.reset_account:
        if API_KEY == "your_testnet_api_key_here":
            logger.error("Please set BINANCE_API_KEY to reset account")
            sys.exit(1)
        if not API_SECRET and not PRIVATE_KEY_PATH:
            logger.error("Provide API_SECRET or BINANCE_PRIVATE_KEY_PATH")
            sys.exit(1)
        if PRIVATE_KEY_PATH and not Path(PRIVATE_KEY_PATH).exists():
            logger.error(f"Private key not found: {PRIVATE_KEY_PATH}")
            sys.exit(1)
        reset_testnet_account(
            api_key=API_KEY,
            api_secret=API_SECRET,
            private_key_path=PRIVATE_KEY_PATH if (PRIVATE_KEY_PATH and Path(PRIVATE_KEY_PATH).exists()) else None,
            clear_local_db=True,
        )
        logger.info("Test account reset complete.")
        sys.exit(0)
    
    # Trading configuration
    # TICKER_LIST in .env: comma-separated symbols, e.g. TICKER_LIST=BTCUSDT,ETHUSDT,BNBUSDT
    # Leave unset or empty to trade all USDT pairs from exchange
    _ticker_env = (os.getenv("TICKER_LIST") or "").strip()
    TICKER_LIST = [s.strip() for s in _ticker_env.split(",") if s.strip()] if _ticker_env else None
    TIME_INTERVAL = "15m"  # Strategy interval
    # Check interval: How often to check for signals and risk management
    # - For 15m candles: 15-30 seconds is good (catches new candles quickly)
    # - Too frequent (< 10s): More API calls, potential rate limiting
    # - Too slow (> 60s): Might miss signals or risk management triggers
    CHECK_INTERVAL_SECONDS = 60  # Check for signals every 60 seconds
    # Consider entry/exit active if it occurred in the last N candles. With 24/7 running we still
    # can miss the exact candle (exchange delay, check timing). Use 4–6 so we don't miss signals.
    SIGNAL_LOOKBACK_CANDLES = 6  # e.g. 6 x 15m = 1.5h window to catch recent BUY/SELL

    # RL Agent configuration
    USE_RL_AGENT = True  # Enable RL agent for hold/close decisions
    RL_MODEL_PATH = Path("models/rl_agent")  # Path to RL model directory (parent of best_model folder)
    RL_MODEL_NAME = "best_model"  # Model filename without .zip extension

    # Close triggers: enable/disable each way to close a position (env: CLOSE_ON_STRATEGY_SELL, CLOSE_ON_RL_AGENT, CLOSE_ON_RISK_MANAGEMENT; 1/true=on, 0/false=off)
    def _env_bool(name: str, default: bool) -> bool:
        v = os.getenv(name)
        if v is None:
            return default
        return v.strip().lower() in ("1", "true", "yes", "on")
    CLOSE_ON_STRATEGY_SELL = _env_bool("CLOSE_ON_STRATEGY_SELL", True)
    CLOSE_ON_RL_AGENT = _env_bool("CLOSE_ON_RL_AGENT", True)
    CLOSE_ON_RISK_MANAGEMENT = _env_bool("CLOSE_ON_RISK_MANAGEMENT", True)
    
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
        signal_lookback_candles=SIGNAL_LOOKBACK_CANDLES,
        use_rl_agent=USE_RL_AGENT,
        rl_model_path=RL_MODEL_PATH,
        rl_model_name=RL_MODEL_NAME,
        close_on_strategy_sell=CLOSE_ON_STRATEGY_SELL,
        close_on_rl_agent=CLOSE_ON_RL_AGENT,
        close_on_risk_management=CLOSE_ON_RISK_MANAGEMENT,
    )

"""
Binance trader: API client, auth (HMAC/RSA), orders, positions, risk checks, RL agent decision.
"""

import time
import logging
from pathlib import Path
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pandas as pd
import numpy as np
import requests
import hmac
import hashlib
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from binance.client import Client
from binance.exceptions import BinanceAPIException

from live_trading.config import (
    TRADING_ENABLED,
    MIN_TRADE_AMOUNT,
    TRADE_PERCENTAGE,
    MAX_POSITION_SIZE,
    TICKER_BLACKLIST,
    LOCAL_TIMEZONE,
    ENABLE_RISK_MANAGEMENT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_HOLDING_PERIODS,
    USE_STOP_LOSS,
    USE_TAKE_PROFIT,
    USE_MAX_HOLDING,
    INITIAL_BALANCE,
)
from rl_risk_management import RLRiskManager
from rl_risk_env import RiskManagementEnv, ENV_DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class BinanceTrader:
    """Handles live trading on Binance testnet.

    Supports two authentication methods:
    1. HMAC (API secret) - traditional method
    2. RSA (private key file) - more secure method
    """

    def __init__(
        self,
        api_key: str,
        api_secret: Optional[str] = None,
        private_key_path: Optional[str] = None,
        testnet: bool = True,
    ):
        if private_key_path and Path(private_key_path).exists():
            self.auth_method = "RSA"
            try:
                with open(private_key_path, "rb") as f:
                    self.private_key = serialization.load_pem_private_key(
                        f.read(), password=None, backend=default_backend()
                    )
                logger.info(f"Using RSA authentication with private key: {private_key_path}")
            except Exception as e:
                logger.error(f"Failed to load private key: {e}")
                raise
        elif api_secret:
            self.auth_method = "HMAC"
            logger.info("Using HMAC authentication with API secret")
        else:
            raise ValueError("Either api_secret or private_key_path must be provided")

        self.testnet = testnet
        self.api_key = api_key
        self.api_secret = api_secret
        self.private_key_path = private_key_path

        if testnet:
            self.base_url = "https://testnet.binance.vision"
            logger.info("Connected to Binance TESTNET")
        else:
            self.base_url = "https://api.binance.com"
            logger.warning("Connected to Binance PRODUCTION - BE CAREFUL!")

        if api_secret:
            try:
                self.client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet) if testnet else Client(api_key=api_key, api_secret=api_secret)
            except Exception as e:
                logger.warning(f"Could not initialize Binance client: {e}")
                self.client = None
        else:
            self.client = None

        self.positions = {}
        self.trade_history = []
        self._exchange_info_cache: Optional[Dict] = None
        self._exchange_info_ts: float = 0.0
        self._exchange_info_ttl_sec: float = 300.0

    def _get_exchange_info(self) -> Optional[Dict]:
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
            return self._exchange_info_cache

    def _get_min_notional(self, symbol: str) -> float:
        info = self._get_exchange_info()
        if not info:
            return 0.0
        symbol_info = next((s for s in info.get("symbols", []) if s.get("symbol") == symbol), None)
        if not symbol_info:
            return 0.0
        min_for_market = 0.0
        min_any = 0.0
        for f in symbol_info.get("filters", []):
            if f.get("filterType") not in ("NOTIONAL", "MIN_NOTIONAL"):
                continue
            try:
                val = float(f.get("minNotional", 0) or 0)
            except (TypeError, ValueError):
                continue
            if val <= 0:
                continue
            if f.get("applyToMarket", False):
                min_for_market = val
                break
            if min_any == 0:
                min_any = val
        return min_for_market or min_any

    def print_positions(self):
        if not self.positions:
            logger.info("" + "=" * 60)
            logger.info("📊 CURRENT POSITIONS: None")
            logger.info("=" * 60)
            return
        logger.info("" + "=" * 60)
        logger.info("📊 CURRENT POSITIONS")
        logger.info("=" * 60)
        total_pnl_usdt = 0.0
        total_value_usdt = 0.0
        for symbol, pos in self.positions.items():
            current_price = self.get_current_price(symbol)
            entry_price = pos["entry_price"]
            quantity = pos["quantity"]
            entry_time = pos["entry_time"]
            entry_value_actual = entry_price * quantity
            current_value = current_price * quantity
            pnl_pct = ((current_value - entry_value_actual) / entry_value_actual) * 100 if entry_value_actual else 0.0
            pnl_usdt = current_value - entry_value_actual
            entry_utc = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            holding_time = now_utc - entry_utc
            hours = holding_time.total_seconds() / 3600
            minutes = (holding_time.total_seconds() % 3600) / 60
            total_pnl_usdt += pnl_usdt
            total_value_usdt += current_value
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
            if isinstance(entry_time, datetime):
                entry_utc = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
                entry_local = entry_utc.astimezone(LOCAL_TIMEZONE)
                logger.info(f"  Entry Time:   {entry_local.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
            else:
                logger.info(f"  Entry Time:   {entry_time}")
        total_pnl_sign = "+" if total_pnl_usdt >= 0 else ""
        total_pnl_indicator = "🟢" if total_pnl_usdt >= 0 else "🔴"
        logger.info("" + "-" * 60)
        logger.info("SUMMARY:")
        logger.info(f"  Total Positions (on exchange): {len(self.positions)}")
        logger.info(f"  Total Value:     ${total_value_usdt:.2f}")
        logger.info(f"  Total PnL:       {total_pnl_indicator} {total_pnl_sign}{total_pnl_usdt:.2f} USDT")
        logger.info("=" * 60)

    def _sign_query_string_hmac(self, query_string: str) -> str:
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _sign_query_string_rsa(self, query_string: str) -> str:
        signature_bytes = self.private_key.sign(
            query_string.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256()
        )
        return base64.b64encode(signature_bytes).decode("utf-8")

    def _sign_query_string(self, query_string: str) -> str:
        if self.auth_method == "RSA":
            return self._sign_query_string_rsa(query_string)
        if self.auth_method == "HMAC":
            return self._sign_query_string_hmac(query_string)
        raise ValueError(f"Unknown authentication method: {self.auth_method}")

    def _make_api_request(self, method: str, endpoint: str, params: dict = None) -> Optional[Dict]:
        if params is None:
            params = {}
        params = {k: v if isinstance(v, str) else str(v) for k, v in dict(params).items()}
        params["timestamp"] = str(int(time.time() * 1000))
        query_string_to_sign = urlencode(sorted(params.items()))
        signature = self._sign_query_string(query_string_to_sign)
        params["signature"] = signature
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        try:
            if method.upper() == "GET":
                query_string = urlencode(sorted(params.items()))
                full_url = f"{url}?{query_string}"
                response = requests.get(full_url, headers=headers, timeout=10)
            elif method.upper() == "POST":
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                body = urlencode(sorted(params.items()))
                response = requests.post(url, headers=headers, data=body, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    logger.error(f"Error response: {e.response.json()}")
                except Exception:
                    logger.error(f"Error response text: {e.response.text}")
            return None

    def get_account_balance(self) -> Dict[str, float]:
        try:
            if self.client:
                account = self.client.get_account()
            else:
                account = self._make_api_request("GET", "/api/v3/account")
                if not account:
                    return {}
            balances = {}
            for balance in account["balances"]:
                asset = balance["asset"]
                free = float(balance["free"])
                locked = float(balance["locked"])
                if free > 0 or locked > 0:
                    balances[asset] = {"free": free, "locked": locked, "total": free + locked}
            return balances
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}

    def get_my_trades(self, symbol: str, limit: int = 100) -> list:
        try:
            if self.client:
                return self.client.get_my_trades(symbol=symbol, limit=limit)
            data = self._make_api_request("GET", "/api/v3/myTrades", {"symbol": symbol, "limit": limit})
            return data if data else []
        except Exception as e:
            logger.debug(f"get_my_trades {symbol}: {e}")
            return []

    def sync_positions_from_exchange(self, ticker_list: list) -> None:
        self.positions.clear()
        balances = self.get_account_balance()
        if not balances:
            return
        for symbol in ticker_list:
            base = symbol.replace("USDT", "")
            if base == symbol:
                continue
            total = balances.get(base, {}).get("total", 0) or 0
            if total <= 0:
                continue
            quantity = float(total)
            current_price = self.get_current_price(symbol)
            usdt_value = quantity * current_price if current_price else 0.0
            entry_price = current_price
            entry_time = datetime.now(timezone.utc)
            trades = self.get_my_trades(symbol, limit=200)
            buys = [t for t in trades if t.get("isBuyer") is True]
            if buys:
                buys_sorted = sorted(buys, key=lambda t: t["time"], reverse=True)
                cum_qty = 0.0
                cost_sum = 0.0
                earliest_time = None
                for t in buys_sorted:
                    qty = float(t["qty"])
                    price = float(t["price"])
                    cost_sum += qty * price
                    cum_qty += qty
                    if earliest_time is None or t["time"] < earliest_time:
                        earliest_time = t["time"]
                    if cum_qty >= quantity:
                        break
                if cum_qty > 0:
                    entry_price = cost_sum / cum_qty
                    entry_time = datetime.fromtimestamp(earliest_time / 1000.0, tz=timezone.utc)
            self.positions[symbol] = {
                "entry_price": entry_price,
                "entry_time": entry_time,
                "quantity": quantity,
                "usdt_value": usdt_value,
            }
            logger.info(f"Synced position from exchange: {symbol} qty={quantity:.8f} entry≈{entry_price:.4f}")

    def get_all_usdt_pairs(self, include_blacklist: bool = False) -> list:
        try:
            if self.client:
                exchange_info = self.client.get_exchange_info()
            else:
                url = f"{self.base_url}/api/v3/exchangeInfo"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                exchange_info = response.json()
            pairs = [
                s["symbol"]
                for s in exchange_info["symbols"]
                if s.get("quoteAsset") == "USDT"
                and s.get("status") == "TRADING"
                and (include_blacklist or s["symbol"] not in TICKER_BLACKLIST)
            ]
            return sorted(pairs)
        except Exception as e:
            logger.error(f"Error fetching USDT pairs: {e}")
            return []

    def get_current_price(self, symbol: str) -> float:
        try:
            if self.client:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
            else:
                url = f"{self.base_url}/api/v3/ticker/price"
                response = requests.get(url, params={"symbol": symbol}, timeout=10)
                response.raise_for_status()
                ticker = response.json()
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base",
                    "taker_buy_quote", "ignore",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
            return df
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

    def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        if not TRADING_ENABLED:
            logger.info(f"[PAPER TRADING] Would {side} {quantity} {symbol}")
            return {"status": "PAPER_TRADE", "side": side, "quantity": quantity, "symbol": symbol}
        price = self.get_current_price(symbol)
        if price and price > 0:
            notional = quantity * price
            min_notional = self._get_min_notional(symbol)
            if min_notional > 0 and notional < min_notional:
                logger.info(f"Order skipped (NOTIONAL): {symbol} {side} {notional:.2f} USDT < min {min_notional:.2f} USDT")
                return None
        try:
            if self.client:
                if side == "BUY":
                    order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
                else:
                    order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
            else:
                params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": quantity}
                order = self._make_api_request("POST", "/api/v3/order", params)
            if order and isinstance(order, dict):
                fills = order.get("fills") or []
                fill_price = fills[0].get("price", "N/A") if fills else "N/A"
                logger.info(f"Order placed: {side} {quantity} {symbol} at {fill_price}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def calculate_trade_quantity(self, symbol: str, usdt_amount: float) -> float:
        try:
            price = self.get_current_price(symbol)
            if price == 0:
                return 0.0
            if self.client:
                exchange_info = self.client.get_exchange_info()
            else:
                url = f"{self.base_url}/api/v3/exchangeInfo"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                exchange_info = response.json()
            symbol_info = next((s for s in exchange_info["symbols"] if s["symbol"] == symbol), None)
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found")
                return 0.0
            lot_size_filter = next((f for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE"), None)
            precision = 8
            if lot_size_filter:
                step_size = float(lot_size_filter["stepSize"])
                precision = len(str(step_size).rstrip("0").split(".")[-1])
            quantity = usdt_amount / price
            if lot_size_filter:
                quantity = (quantity // step_size) * step_size
            quantity = round(quantity, precision)
            return quantity
        except Exception as e:
            logger.error(f"Error calculating trade quantity: {e}")
            return 0.0

    def buy(self, symbol: str, usdt_amount: Optional[float] = None) -> bool:
        try:
            balances = self.get_account_balance()
            usdt_balance = balances.get("USDT", {}).get("free", 0.0)
            if usdt_amount is None:
                usdt_amount = usdt_balance * TRADE_PERCENTAGE
            if usdt_amount < MIN_TRADE_AMOUNT:
                logger.warning(f"Insufficient balance: {usdt_amount} USDT < {MIN_TRADE_AMOUNT} USDT")
                return False
            if usdt_amount > MAX_POSITION_SIZE:
                usdt_amount = MAX_POSITION_SIZE
                logger.info(f"Capping trade size to {MAX_POSITION_SIZE} USDT")
            quantity = self.calculate_trade_quantity(symbol, usdt_amount)
            if quantity == 0:
                logger.error(f"Could not calculate quantity for {symbol}")
                return False
            order = self.place_market_order(symbol, "BUY", quantity)
            if order:
                entry_price = self.get_current_price(symbol)
                self.positions[symbol] = {
                    "entry_price": entry_price,
                    "entry_time": datetime.now(timezone.utc),
                    "quantity": quantity,
                    "usdt_value": usdt_amount,
                }
                self.trade_history.append({
                    "timestamp": datetime.now(timezone.utc),
                    "symbol": symbol,
                    "side": "BUY",
                    "quantity": quantity,
                    "price": entry_price,
                    "usdt_value": usdt_amount,
                })
                logger.info(f"BUY executed: {quantity} {symbol} at ${entry_price:.2f}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error executing buy: {e}")
            return False

    def sell(self, symbol: str, quantity: Optional[float] = None) -> bool:
        try:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return False
            position = self.positions[symbol]
            if quantity is None:
                quantity = position["quantity"]
            order = self.place_market_order(symbol, "SELL", quantity)
            if order:
                exit_price = self.get_current_price(symbol)
                entry_price = position["entry_price"]
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                pnl_usdt = (exit_price - entry_price) * quantity
                self.trade_history.append({
                    "timestamp": datetime.now(timezone.utc),
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": quantity,
                    "price": exit_price,
                    "entry_price": entry_price,
                    "pnl_pct": pnl_pct,
                    "pnl_usdt": pnl_usdt,
                })
                logger.info(f"SELL executed: {quantity} {symbol} at ${exit_price:.2f}")
                logger.info(f"PnL: {pnl_pct:.2f}% (${pnl_usdt:.2f})")
                del self.positions[symbol]
                return True
            price = self.get_current_price(symbol)
            min_notional = self._get_min_notional(symbol)
            if price and min_notional > 0 and (quantity * price) < min_notional:
                logger.info(f"Dust removed from tracking: {symbol} ({(quantity * price):.2f} USDT < min {min_notional:.2f})")
                del self.positions[symbol]
                return True
            return False
        except Exception as e:
            logger.error(f"Error executing sell: {e}")
            return False

    def check_risk_management(self, symbol: str) -> Optional[str]:
        if symbol not in self.positions:
            return None
        if not ENABLE_RISK_MANAGEMENT:
            return None
        position = self.positions[symbol]
        current_price = self.get_current_price(symbol)
        entry_price = position["entry_price"]
        entry_time = position["entry_time"]
        returns = (current_price - entry_price) / entry_price
        if USE_STOP_LOSS and returns <= STOP_LOSS_PCT:
            return "STOP_LOSS"
        if USE_TAKE_PROFIT and returns >= TAKE_PROFIT_PCT:
            return "TAKE_PROFIT"
        if USE_MAX_HOLDING:
            entry_utc = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            holding_time = now_utc - entry_utc
            max_holding_time = timedelta(minutes=MAX_HOLDING_PERIODS * 15)
            if holding_time >= max_holding_time:
                return "MAX_HOLDING"
        return None

    def check_rl_agent_decision(
        self,
        symbol: str,
        price_history: pd.Series,
        balance_history: pd.Series,
        rl_manager: Optional[RLRiskManager] = None,
        ohlcv_df: Optional[pd.DataFrame] = None,
    ) -> bool:
        if symbol not in self.positions:
            return False
        if rl_manager is None:
            return self.check_risk_management(symbol) is not None
        position = self.positions[symbol]
        entry_price = position["entry_price"]
        entry_time = position["entry_time"]
        entry_utc = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)

        def _to_utc_aware(ts):
            if isinstance(ts, pd.Timestamp):
                return ts.tz_localize(timezone.utc) if ts.tzinfo is None else ts.tz_convert(timezone.utc)
            return ts.replace(tzinfo=timezone.utc) if getattr(ts, "tzinfo", None) is None else ts.astimezone(timezone.utc)

        entry_idx = None
        for idx, timestamp in enumerate(price_history.index):
            ts_utc = _to_utc_aware(timestamp)
            time_diff = abs((ts_utc - entry_utc).total_seconds())
            if time_diff < 900:
                entry_idx = idx
                break
        assert entry_idx is not None, "entry bar must be present in price_history"

        history_length = rl_manager.history_length
        current_idx = len(price_history) - 1
        window_start = max(0, entry_idx - history_length)
        price_window = price_history.iloc[window_start : current_idx + 1].copy()
        balance_window = balance_history.iloc[window_start : current_idx + 1].copy()
        if len(balance_window) != len(price_window):
            if len(balance_window) < len(price_window):
                last_balance = balance_window.iloc[-1] if len(balance_window) > 0 else INITIAL_BALANCE
                padding = pd.Series(
                    [last_balance] * (len(price_window) - len(balance_window)),
                    index=price_window.index[len(balance_window) :],
                )
                balance_window = pd.concat([balance_window, padding])
            else:
                balance_window = balance_window.iloc[: len(price_window)]
        exit_signals = pd.Series(False, index=price_window.index)
        all_tickers_data = {
            "single_ticker": {
                "price": price_window,
                "balance": balance_window,
                "entry_signals": pd.Series(False, index=price_window.index),
                "exit_signals": exit_signals,
            }
        }

        ohlcv_full_window = ohlcv_df.iloc[window_start : current_idx + 1].copy()

        env = RiskManagementEnv(
            all_tickers_data=all_tickers_data,
            initial_balance=rl_manager.initial_balance,
            history_length=history_length,
            max_steps=rl_manager.max_steps,
            obs_periods_norm_steps=ENV_DEFAULT_CONFIG["obs_periods_norm_steps"],
            fee_rate=rl_manager.fee_rate,
        )
        env.current_ticker = "single_ticker"
        env.price_data = price_window
        env.balance_data = balance_window
        env.exit_signals = exit_signals
        env.entry_price = entry_price
        env.entry_idx = entry_idx - window_start
        env.exit_signal_idx = None
        env.exit_idx = len(price_window) - 1
        env.episode_start_idx = env.entry_idx
        env.episode_end_idx = env.exit_idx
        env.episode_length = env.exit_idx - env.episode_start_idx + 1
        env.data_start_idx = max(0, env.entry_idx - env.history_length)
        env.price_window = price_window.iloc[env.data_start_idx : env.exit_idx + 1].copy()
        env.balance_window = balance_window.iloc[env.data_start_idx : env.exit_idx + 1].copy()
        env.ohlcv_window = ohlcv_full_window.iloc[env.data_start_idx : env.exit_idx + 1].copy()
        env.current_step = 0
        env.current_idx = 0
        env.position_open = True
        env.peak_balance = rl_manager.initial_balance
        env.max_drawdown = 0.0
        env.total_reward = 0.0
        env.episode_return = 0.0
        env.returns_history = []
        env.wins = 0
        env.total_trades = 0
        env.entry_balance = rl_manager.initial_balance
        env.episode_max_price = entry_price
        env.episode_min_price = entry_price
        env._precomputed_indicators = env._precompute_indicators()
        last_step_in_window = env.exit_idx - env.data_start_idx
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        obs = env._get_observation()
        obs = rl_manager.vec_normalize.normalize_obs(obs.reshape(1, -1))[0]
        while True:
            action, lstm_states = rl_manager.model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=rl_manager.deterministic
            )
            if env.current_idx >= last_step_in_window:
                return action == 1
            obs, _reward, terminated, truncated, _info = env.step(action)
            obs = rl_manager.vec_normalize.normalize_obs(obs.reshape(1, -1))[0]
            episode_starts = np.array([terminated or truncated], dtype=bool)

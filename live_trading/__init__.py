"""
Live Trading package for Binance Testnet.
Submodules: config, binance_trader, data, runner.
"""

from live_trading.config import setup_logging, INITIAL_BALANCE, TESTNET
from live_trading.binance_trader import BinanceTrader
from live_trading.runner import run_live_trading, test_positions, reset_testnet_account

# Ensure logging is configured when package is used
setup_logging()

__all__ = [
    "BinanceTrader",
    "run_live_trading",
    "test_positions",
    "reset_testnet_account",
    "INITIAL_BALANCE",
    "TESTNET",
]

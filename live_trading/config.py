"""Live trading configuration and constants."""

import os
import logging
from pathlib import Path
from datetime import timezone, timedelta

from dotenv import load_dotenv

load_dotenv()

# Add FinRL-Meta to sys.path (configurable via FINRL_META_PATH in .env)
_finrl_meta_path = Path(os.getenv("FINRL_META_PATH", "/mnt/data/FinRL-Tutorials/3-Practical/FinRL-Meta"))
if str(_finrl_meta_path) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_finrl_meta_path))

# Re-export from backtest for live trading and dashboard
from backtest import (
    ENABLE_RISK_MANAGEMENT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_HOLDING_PERIODS,
    USE_STOP_LOSS,
    USE_TAKE_PROFIT,
    USE_MAX_HOLDING,
    INITIAL_BALANCE,
)

# Timezone: all logic uses UTC. LOCAL_TIMEZONE is only for log output.
LOCAL_TIMEZONE = timezone(timedelta(hours=7))

# Trading Configuration
TESTNET = True  # Set to False for production (NOT RECOMMENDED without thorough testing)
TRADING_ENABLED = True  # Set to False to run in paper trading mode (no actual orders)
MIN_TRADE_AMOUNT = 100.0  # Minimum trade amount in USDT
TRADE_PERCENTAGE = 0.95  # Percentage of available balance to use per trade
MAX_POSITION_SIZE = 100.0  # Maximum position size in USDT
MIN_TICKER_PRICE = 0.1  # Skip tickers with price below this

# Tickers to exclude from trading (no new positions; existing positions still force-closed if invalid)
TICKER_BLACKLIST = frozenset({
    "ACAUSDT", "BTTCUSDT", "CHESSUSDT", "DATAUSDT",
    "GHSTUSDT", "NKNUSDT", "PEPEUSDT", "SHIBUSDT",
})

# Binance Testnet Configuration
BINANCE_TESTNET_BASE_URL = "https://testnet.binance.vision"
BINANCE_TESTNET_API_URL = "https://testnet.binance.vision/api"

# RL agent: ensure this many bars before entry so technical indicators have enough history
ENTRY_LOOKBACK_STEPS = 500

# Only open a position if the entry signal is within this many candles of the latest candle
ENTRY_NEAR_CURRENT_CANDLES = 2


def setup_logging() -> logging.Logger:
    """Configure file logging for live trading. Idempotent."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if not any(h for h in root.handlers if getattr(h, "baseFilename", "").endswith("live_trading.log")):
        handler = logging.FileHandler("logs/live_trading.log", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root.addHandler(handler)
    return logging.getLogger("live_trading")

"""
Live Trading CLI for Binance Testnet.
Run: python live_trading.py

Implementation lives in the live_trading package (live_trading/).
"""

import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from live_trading.config import setup_logging
from live_trading.runner import run_live_trading, test_positions, reset_testnet_account

logger = setup_logging()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Trading on Binance Testnet")
    parser.add_argument("--test-positions", action="store_true", help="Test open/close position: run buy-sell roundtrip then exit")
    parser.add_argument("--test-symbol", type=str, default="BNBUSDT", help="Symbol for --test-positions (default: BNBUSDT)")
    parser.add_argument("--test-amount", type=float, default=None, help="USDT amount for --test-positions (default: MIN_TRADE_AMOUNT)")
    parser.add_argument("--reset-account", action="store_true", help="Reset testnet account: sell all positions (to USDT) and clear local DB")
    args, _ = parser.parse_known_args()

    API_KEY = os.getenv("BINANCE_API_KEY", "your_testnet_api_key_here")
    API_SECRET = os.getenv("BINANCE_API_SECRET", None)
    PRIVATE_KEY_PATH = os.getenv("BINANCE_PRIVATE_KEY_PATH", "test-prv-key.pem")

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
        ok = test_positions(api_key=API_KEY, api_secret=API_SECRET, private_key_path=PRIVATE_KEY_PATH, symbol=args.test_symbol, usdt_amount=args.test_amount)
        sys.exit(0 if ok else 1)

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

    _ticker_env = (os.getenv("TICKER_LIST") or "").strip()
    TICKER_LIST = [s.strip() for s in _ticker_env.split(",") if s.strip()] if _ticker_env else None
    TIME_INTERVAL = "15m"
    CHECK_INTERVAL_SECONDS = 60
    SIGNAL_LOOKBACK_CANDLES = 6
    USE_RL_AGENT = True
    RL_MODEL_PATH = Path("models/rl_agent")
    RL_MODEL_NAME = "best_model"
    CLOSE_ON_STRATEGY_SELL = _env_bool("CLOSE_ON_STRATEGY_SELL", True)
    CLOSE_ON_RL_AGENT = _env_bool("CLOSE_ON_RL_AGENT", True)
    CLOSE_ON_RISK_MANAGEMENT = _env_bool("CLOSE_ON_RISK_MANAGEMENT", True)

    if API_KEY == "your_testnet_api_key_here":
        logger.error("Please set your Binance testnet API key!")
        logger.error("Get testnet keys from: https://testnet.binance.vision/")
        logger.error("For RSA: export BINANCE_API_KEY='...' BINANCE_PRIVATE_KEY_PATH='test-prv-key.pem'")
        logger.error("For HMAC: export BINANCE_API_KEY='...' BINANCE_API_SECRET='...'")
        sys.exit(1)
    if not API_SECRET and not PRIVATE_KEY_PATH:
        logger.error("Provide API_SECRET or BINANCE_PRIVATE_KEY_PATH")
        sys.exit(1)
    if PRIVATE_KEY_PATH and not Path(PRIVATE_KEY_PATH).exists():
        logger.error(f"Private key file not found: {PRIVATE_KEY_PATH}")
        sys.exit(1)
    if PRIVATE_KEY_PATH:
        logger.info(f"Using RSA authentication with private key: {PRIVATE_KEY_PATH}")
    else:
        logger.info("Using HMAC authentication with API secret")

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


if __name__ == "__main__":
    main()

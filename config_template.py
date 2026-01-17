"""
Configuration Template for Live Trading
Copy this file to config.py and fill in your API keys.

IMPORTANT: 
- Never commit config.py to version control
- Use testnet API keys for testing
- Only use production keys after thorough testing
"""

# Binance Testnet API Keys
# Get your testnet keys from: https://testnet.binance.vision/
BINANCE_API_KEY = "your_testnet_api_key_here"

# Authentication Method 1: RSA (Private Key) - RECOMMENDED
# Use this if you're using RSA private key authentication (NO API SECRET NEEDED!)
# Generate a key pair: openssl genrsa -out test-prv-key.pem 2048
# Register the public key with Binance testnet
# You'll get an API_KEY from Binance - that's all you need (no secret!)
BINANCE_PRIVATE_KEY_PATH = "test-prv-key.pem"  # Path to your RSA private key PEM file

# Authentication Method 2: HMAC (API Secret) - LEGACY
# Use this only if you have an API secret key (older method)
# BINANCE_API_SECRET = "your_testnet_api_secret_here"

# Note: Use EITHER PRIVATE_KEY_PATH (RSA - Recommended) OR API_SECRET (HMAC - Legacy), not both
# RSA authentication does NOT require an API secret!

# Trading Configuration
TICKER_LIST = ["BTCUSDT", "ETHUSDT"]  # Tickers to trade
TIME_INTERVAL = "15m"  # Strategy interval (must match main.py)
CHECK_INTERVAL_SECONDS = 60  # How often to check for signals (in seconds)

# Risk Settings
MIN_TRADE_AMOUNT = 10.0  # Minimum trade amount in USDT
TRADE_PERCENTAGE = 0.95  # Percentage of available balance to use per trade
MAX_POSITION_SIZE = 1000.0  # Maximum position size in USDT

# Trading Mode
TRADING_ENABLED = True  # Set to False for paper trading (no actual orders)
TESTNET = True  # Set to False for production (NOT RECOMMENDED without testing)

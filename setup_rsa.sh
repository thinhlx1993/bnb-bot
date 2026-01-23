#!/bin/bash
# Quick setup script for RSA authentication
# Note: RSA authentication does NOT require an API secret!

echo "Setting up RSA authentication for Binance testnet..."
echo "Note: RSA authentication does NOT require an API secret!"
echo ""

# Set your API key (from Binance testnet - this is all you need!)
export BINANCE_API_KEY=""

# Set path to private key (should be in current directory)
export BINANCE_PRIVATE_KEY_PATH="test-prv-key.pem"

# Verify private key exists
if [ ! -f "$BINANCE_PRIVATE_KEY_PATH" ]; then
    echo "ERROR: Private key file not found: $BINANCE_PRIVATE_KEY_PATH"
    echo "Please make sure test-prv-key.pem is in the current directory"
    exit 1
fi

echo "✓ API Key set: $BINANCE_API_KEY"
echo "✓ Private key found: $BINANCE_PRIVATE_KEY_PATH"
echo "✓ No API secret needed for RSA authentication!"
echo ""
echo "Configuration complete! You can now run:"
echo "  python live_trading.py"
echo ""
echo "Or run this script with --run flag to start immediately:"
echo "  ./setup_rsa.sh --run"

# Optionally run the bot
if [ "$1" == "--run" ]; then
    echo ""
    echo "Starting live trading bot..."
    python live_trading.py
fi

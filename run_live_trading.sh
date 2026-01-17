#!/bin/bash
# Simple script to run live trading with environment variables

# Set your Binance testnet API keys here or use environment variables
export BINANCE_API_KEY="${BINANCE_API_KEY:-your_testnet_api_key_here}"
export BINANCE_API_SECRET="${BINANCE_API_SECRET:-your_testnet_api_secret_here}"

# Run the live trading script
python live_trading.py

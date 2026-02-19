"""
Live trading loop, position test, and account reset.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from live_trading.config import (
    TESTNET,
    TRADING_ENABLED,
    TICKER_BLACKLIST,
    ENTRY_LOOKBACK_STEPS,
    ENTRY_NEAR_CURRENT_CANDLES,
    INITIAL_BALANCE,
    LOCAL_TIMEZONE,
    MIN_TRADE_AMOUNT,
    MIN_TICKER_PRICE,
)
from live_trading.binance_trader import BinanceTrader
from live_trading.data import fetch_live_data, generate_signals
from live_trading_db import (
    init_db,
    insert_signal,
    insert_position,
    close_position,
    get_open_positions,
    reset_db,
)
from rl_risk_management import RLRiskManager
from utils.time_utils import interval_to_timedelta

logger = logging.getLogger(__name__)


def test_positions(
    api_key: str,
    api_secret: Optional[str] = None,
    private_key_path: Optional[str] = None,
    symbol: str = "BNBUSDT",
    usdt_amount: Optional[float] = None,
) -> bool:
    """Test open and close position by executing a buy-sell roundtrip on Binance testnet."""
    amount = usdt_amount if usdt_amount is not None and usdt_amount >= MIN_TRADE_AMOUNT else MIN_TRADE_AMOUNT
    logger.info("=" * 60)
    logger.info("POSITION TEST MODE - Buy then Sell roundtrip")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Amount: {amount} USDT")
    logger.info(f"Trading Enabled: {TRADING_ENABLED}")
    logger.info("")
    trader = BinanceTrader(
        api_key=api_key,
        api_secret=api_secret,
        private_key_path=private_key_path,
        testnet=TESTNET,
    )
    init_db()
    balances = trader.get_account_balance()
    usdt_before = balances.get("USDT", {}).get("total", 0.0)
    logger.info(f"USDT before: ${usdt_before:.2f}")
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
    insert_position(symbol, pos["entry_price"], pos["quantity"], pos["usdt_value"])
    time.sleep(2)
    logger.info("")
    logger.info("Step 2: Closing position (SELL)...")
    success_sell = trader.sell(symbol)
    if not success_sell:
        logger.error("FAILED: Sell order failed")
        return False
    if symbol in trader.positions:
        logger.error("FAILED: Position still tracked after sell")
        return False
    close_position(symbol, "test")
    balances = trader.get_account_balance()
    usdt_after = balances.get("USDT", {}).get("total", 0.0)
    logger.info("")
    logger.info(f"USDT after:  ${usdt_after:.2f}")
    logger.info("=" * 60)
    logger.info("PASSED: Open and close position test completed successfully")
    logger.info("=" * 60)
    return True


def reset_testnet_account(
    api_key: str,
    api_secret: Optional[str] = None,
    private_key_path: Optional[str] = None,
    clear_local_db: bool = True,
) -> bool:
    """Reset testnet account: sell all open positions (convert to USDT) and optionally clear local DB."""
    trader = BinanceTrader(
        api_key=api_key,
        api_secret=api_secret,
        private_key_path=private_key_path,
        testnet=TESTNET,
    )
    ticker_list = trader.get_all_usdt_pairs(include_blacklist=True)
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
    ticker_list: Optional[list] = None,
    time_interval: str = "15m",
    check_interval_seconds: int = 15,
    signal_lookback_candles: int = 6,
    use_rl_agent: bool = True,
    rl_model_path: Optional[Path] = None,
    rl_model_name: str = "best_model",
    close_on_strategy_sell: bool = True,
    close_on_rl_agent: bool = True,
    close_on_risk_management: bool = True,
) -> None:
    """Run live trading loop."""
    trader = BinanceTrader(
        api_key=api_key,
        api_secret=api_secret,
        private_key_path=private_key_path,
        testnet=TESTNET,
    )
    if ticker_list is None:
        ticker_list = trader.get_all_usdt_pairs()
        if not ticker_list:
            logger.error("Failed to fetch USDT pairs from exchange, falling back to default list")
            ticker_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "SOLUSDT"]
        else:
            logger.info(f"Using all {len(ticker_list)} USDT pairs from exchange")
    ticker_list = [t for t in ticker_list if t not in TICKER_BLACKLIST]
    if not ticker_list:
        logger.error("No tickers left after blacklist filter")
        sys.exit(1)
    logger.info("=" * 60)
    logger.info("Starting Live Trading Bot")
    logger.info("=" * 60)
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
                deterministic=True,
            )
            logger.info(f"✅ RL model loaded successfully (device: {rl_manager.model.device})")
        except Exception as e:
            logger.error(f"❌ Failed to load RL model: {e}")
            logger.warning("Falling back to rule-based risk management")
            use_rl_agent = False
            rl_manager = None
    init_db()
    trader.sync_positions_from_exchange(ticker_list)
    open_in_db = {p["ticker"] for p in get_open_positions()}
    for symbol, pos in trader.positions.items():
        if symbol not in open_in_db:
            insert_position(symbol, pos["entry_price"], pos["quantity"], pos["usdt_value"])
            open_in_db.add(symbol)
            logger.info(f"Synced position to DB: {symbol} qty={pos['quantity']:.8f} entry≈{pos['entry_price']:.4f}")
    trader.get_account_balance()
    trader.print_positions()
    last_signals = {ticker: {"entry": False, "exit": False} for ticker in ticker_list}
    for ticker in trader.positions:
        last_signals[ticker]["entry"] = True
    try:
        while True:
            logger.info("=" * 60)
            logger.info(f"Checking signals at {datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
            logger.info("=" * 60)
            df = fetch_live_data(trader, ticker_list, time_interval, lookback_periods=ENTRY_LOOKBACK_STEPS)
            if df.empty:
                logger.warning("No data fetched, skipping this cycle")
                time.sleep(check_interval_seconds)
                continue
            for ticker in ticker_list:
                ticker_df = df[df["tic"] == ticker]
                if not ticker_df.empty:
                    price = float(ticker_df["close"].iloc[-1])
                else:
                    price = trader.get_current_price(ticker)
                if not price:
                    logger.error(f"Failed to get price for {ticker}")
                    continue
                if price < MIN_TICKER_PRICE:
                    logger.debug(f"Skipping {ticker} (price {price:.4f} < {MIN_TICKER_PRICE})")
                    continue
                logger.info(f"Processing {ticker}...")
                if ticker in trader.positions:
                    ticker_df = df[df["tic"] == ticker].copy()
                    if not ticker_df.empty:
                        ticker_df = ticker_df.set_index("time").sort_index()
                        price_history = ticker_df["close"]
                        current_balance = trader.get_account_balance().get("USDT", {}).get("total", INITIAL_BALANCE)
                        balance_history = pd.Series(current_balance, index=price_history.index)
                        if use_rl_agent and close_on_rl_agent and rl_manager is not None:
                            try:
                                should_close = trader.check_rl_agent_decision(
                                    ticker, price_history, balance_history, rl_manager, ohlcv_df=ticker_df
                                )
                                if should_close:
                                    logger.info(f"🤖 RL Agent decision: CLOSE position for {ticker}")
                                    if trader.sell(ticker):
                                        close_position(ticker, "rl")
                                        insert_signal(ticker, "sell", "rl", None)
                                    last_signals[ticker]["exit"] = True
                                    continue
                                logger.info(f"🤖 RL Agent decision: HOLD position for {ticker}")
                            except Exception as e:
                                logger.error(f"Error querying RL agent for {ticker}: {e}")
                                logger.warning("Falling back to rule-based risk management")
                        if close_on_risk_management:
                            risk_exit = trader.check_risk_management(ticker)
                            if risk_exit:
                                logger.warning(f"Risk management exit triggered: {risk_exit}")
                                if trader.sell(ticker):
                                    close_position(ticker, "risk_management")
                                    insert_signal(ticker, "sell", "risk_management", None)
                                last_signals[ticker]["exit"] = True
                                continue
                entries, exits = generate_signals(df, ticker)
                if entries.empty or exits.empty:
                    logger.warning(f"No signals generated for {ticker}")
                    continue
                total_buy_signals = entries.sum()
                total_sell_signals = exits.sum()
                last_buy_time = entries[entries == True].index[-1] if total_buy_signals > 0 else None
                last_sell_time = exits[exits == True].index[-1] if total_sell_signals > 0 else None
                if last_buy_time is not None:
                    lb = last_buy_time.tz_localize("UTC").tz_convert("Asia/Bangkok") if getattr(last_buy_time, "tz", None) is None else last_buy_time.tz_convert("Asia/Bangkok")
                    logger.info(f"  📈 Last BUY signal at: {lb.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
                if last_sell_time is not None:
                    ls = last_sell_time.tz_localize("UTC").tz_convert("Asia/Bangkok") if getattr(last_sell_time, "tz", None) is None else last_sell_time.tz_convert("Asia/Bangkok")
                    logger.info(f"  📉 Last SELL signal at: {ls.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
                latest_candle_time = entries.index[-1] if len(entries) > 0 else None
                if latest_candle_time is not None:
                    lc = latest_candle_time.tz_localize("UTC").tz_convert("Asia/Bangkok") if getattr(latest_candle_time, "tz", None) is None else latest_candle_time.tz_convert("Asia/Bangkok")
                    logger.info(f"  📊 Latest candle: {lc.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)")
                latest_entry = (last_buy_time is not None) and (last_sell_time is None or last_buy_time > last_sell_time)
                latest_exit = (last_sell_time is not None) and (last_buy_time is None or last_sell_time > last_buy_time)
                entry_near_current = False
                if latest_entry and last_buy_time is not None and latest_candle_time is not None:
                    latest_ts = pd.Timestamp(latest_candle_time)
                    buy_ts = pd.Timestamp(last_buy_time)
                    if latest_ts.tz is not None and buy_ts.tz is None:
                        buy_ts = buy_ts.tz_localize("UTC")
                    elif latest_ts.tz is None and buy_ts.tz is not None:
                        latest_ts = latest_ts.tz_localize("UTC")
                    interval_td = interval_to_timedelta(time_interval)
                    diff_seconds = (latest_ts - buy_ts).total_seconds()
                    max_age_seconds = (ENTRY_NEAR_CURRENT_CANDLES * interval_td).total_seconds()
                    entry_near_current = 0 <= diff_seconds <= max_age_seconds
                if latest_entry and not entry_near_current:
                    logger.info(f"  ⏭️  BUY signal too old for {ticker} (entry not within last {ENTRY_NEAR_CURRENT_CANDLES} candles); skipping open")
                logger.info(f"  Signal Status: {total_buy_signals} buy signals, {total_sell_signals} sell signals in history")
                logger.info(f"  Latest Entry Signal (most recent signal is BUY): {latest_entry}")
                logger.info(f"  Entry near current (within {ENTRY_NEAR_CURRENT_CANDLES} candles): {entry_near_current}")
                logger.info(f"  Last signal processed flag: entry={last_signals[ticker]['entry']}, exit={last_signals[ticker]['exit']}")
                candle_time_iso = pd.Timestamp(latest_candle_time).isoformat() if latest_candle_time is not None else None
                if latest_entry and entry_near_current and not last_signals[ticker]["entry"]:
                    if ticker not in trader.positions:
                        logger.info(f"✅ BUY signal detected for {ticker} - Opening position...")
                        success = trader.buy(ticker)
                        if success:
                            pos = trader.positions[ticker]
                            insert_position(ticker, pos["entry_price"], pos["quantity"], pos["usdt_value"])
                            insert_signal(ticker, "buy", "strategy", candle_time_iso)
                            last_signals[ticker]["entry"] = True
                            logger.info(f"✅ Position opened successfully for {ticker}")
                        else:
                            logger.error(f"❌ Failed to open position for {ticker}")
                    else:
                        logger.info(f"⚠️  BUY signal but position already exists for {ticker}")
                elif latest_entry and last_signals[ticker]["entry"]:
                    logger.info(f"ℹ️  BUY signal still active for {ticker} (already processed)")
                elif not latest_entry:
                    logger.info(f"ℹ️  No active BUY signal for {ticker}")
                if close_on_strategy_sell and latest_exit and not last_signals[ticker]["exit"]:
                    if ticker in trader.positions:
                        logger.info(f"✅ SELL signal detected for {ticker} - Closing position...")
                        success = trader.sell(ticker)
                        if success:
                            close_position(ticker, "strategy")
                            insert_signal(ticker, "sell", "strategy", candle_time_iso)
                            last_signals[ticker]["exit"] = True
                            logger.info(f"✅ Position closed successfully for {ticker}")
                        else:
                            logger.error(f"❌ Failed to close position for {ticker}")
                    else:
                        logger.info(f"⚠️  SELL signal but no position for {ticker}")
                elif close_on_strategy_sell and latest_exit and last_signals[ticker]["exit"]:
                    logger.info(f"ℹ️  SELL signal still active for {ticker} (already processed)")
                elif not close_on_strategy_sell and latest_exit:
                    logger.info(f"ℹ️  SELL signal for {ticker} (close on strategy disabled)")
                elif not latest_exit:
                    logger.info(f"ℹ️  No active SELL signal for {ticker}")
                if not latest_entry:
                    last_signals[ticker]["entry"] = False
                if not latest_exit:
                    last_signals[ticker]["exit"] = False
            trader.print_positions()
            balances = trader.get_account_balance()
            usdt_balance = balances.get("USDT", {}).get("total", 0.0)
            logger.info(f"💰 USDT Balance: ${usdt_balance:.2f}")
            logger.info(f"Waiting {check_interval_seconds} seconds before next check...")
            time.sleep(check_interval_seconds)
    except KeyboardInterrupt:
        logger.info("Stopping trading bot...")
        logger.info("Closing all positions...")
        for symbol in list(trader.positions.keys()):
            trader.sell(symbol)
        balances = trader.get_account_balance()
        relevant_assets = {"USDT"}.union(t.replace("USDT", "") for t in ticker_list if t.endswith("USDT"))
        filtered_balances = {k: v for k, v in balances.items() if k in relevant_assets}
        logger.info(f"Final Balance ({len(filtered_balances)} assets): {filtered_balances}")
        logger.info(f"Trade History ({len(trader.trade_history)} trades):")
        for trade in trader.trade_history:
            logger.info(f"  {trade}")
    except Exception as e:
        logger.error(f"Error in trading loop: {e}", exc_info=True)

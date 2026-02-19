"""
Simple web dashboard to visualize live trading account performance.
Based on live_trading.py - reads from live_trading.db and optionally Binance API.

Run: streamlit run trade_dashboard.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import streamlit as st

# Import live trading components
from live_trading_db import get_conn, get_open_positions, init_db
from live_trading import BinanceTrader, INITIAL_BALANCE, TESTNET

Path("logs").mkdir(parents=True, exist_ok=True)
init_db()


def format_duration(opened_at_str: str) -> str:
    """Return human-readable duration from opened_at to now (e.g. '2h 15m', '3d 5h')."""
    try:
        opened = datetime.fromisoformat(opened_at_str.replace("Z", "+00:00"))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - opened
        if delta < timedelta(0):
            return "0m"
        total_seconds = int(delta.total_seconds())
        days, r = divmod(total_seconds, 86400)
        hours, r = divmod(r, 3600)
        minutes, _ = divmod(r, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        return " ".join(parts)
    except Exception:
        return "-"


def get_closed_positions():
    """Fetch all closed positions from DB."""
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT id, ticker, entry_price, quantity, usdt_value, opened_at, closed_at, exit_reason
            FROM positions WHERE closed_at IS NOT NULL
            ORDER BY closed_at DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_signals_history(limit: int = 100):
    """Fetch recent signals from DB."""
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT id, ticker, signal_type, source, candle_time, created_at
            FROM signals ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def compute_trades_with_pnl(trader, closed_positions):
    """
    Match closed positions with Binance myTrades to compute realized PnL.
    Returns list of dicts with entry/exit info and PnL.
    """
    if not trader or not closed_positions:
        return []
    results = []
    for pos in closed_positions:
        ticker = pos["ticker"]
        entry_price = pos["entry_price"]
        quantity = pos["quantity"]
        usdt_value = pos["usdt_value"]
        opened_at = pos["opened_at"]
        closed_at = pos["closed_at"]
        exit_reason = pos.get("exit_reason", "")
        trades = trader.get_my_trades(ticker, limit=50)
        if not trades:
            results.append({
                **pos,
                "exit_price": None,
                "pnl_usdt": None,
                "pnl_pct": None,
            })
            continue
        # Parse timestamps: opened_at/closed_at are ISO strings
        try:
            closed_ts = datetime.fromisoformat(closed_at.replace("Z", "+00:00")).timestamp() * 1000
            opened_ts = datetime.fromisoformat(opened_at.replace("Z", "+00:00")).timestamp() * 1000
        except Exception:
            results.append({**pos, "exit_price": None, "pnl_usdt": None, "pnl_pct": None})
            continue
        sells = [t for t in trades if t.get("isBuyer") is False and t["time"] >= opened_ts]
        if not sells:
            results.append({**pos, "exit_price": None, "pnl_usdt": None, "pnl_pct": None})
            continue
        # Use most recent sell close to closed_at
        sells_sorted = sorted(sells, key=lambda x: abs(x["time"] - closed_ts))
        best = sells_sorted[0]
        exit_price = float(best["price"])
        exit_qty = float(best["qty"])
        pnl_usdt = (exit_price - entry_price) * min(quantity, exit_qty)
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0
        results.append({
            **pos,
            "exit_price": exit_price,
            "pnl_usdt": pnl_usdt,
            "pnl_pct": pnl_pct,
        })
    return results


def create_trader():
    """Create BinanceTrader if API keys are available."""
    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
    private_key_path = os.getenv("PRIVATE_KEY_PATH", "test-prv-key.pem")
    if not api_key:
        return None
    try:
        return BinanceTrader(
            api_key=api_key,
            api_secret=api_secret if api_secret else None,
            private_key_path=Path(private_key_path) if Path(private_key_path).exists() else None,
            testnet=TESTNET,
        )
    except Exception:
        return None


def main():
    st.set_page_config(page_title="Trade Performance Dashboard", page_icon="📊", layout="wide")
    st.title("📊 Live Trading Performance Dashboard")

    trader = create_trader()
    if trader:
        st.success("Connected to Binance API (live data)")
    else:
        st.info("Running in DB-only mode. Set BINANCE_API_KEY in .env for live balance and PnL.")

    open_positions = get_open_positions()
    closed_positions = get_closed_positions()
    signals = get_signals_history(200)

    # When API available, count positions on exchange (any asset with balance) to match live_trading log
    positions_on_exchange = None
    if trader:
        try:
            ticker_list = trader.get_all_usdt_pairs()
            balances = trader.get_account_balance()
            if ticker_list and balances:
                positions_on_exchange = sum(
                    1 for s in ticker_list
                    if s.endswith("USDT") and (balances.get(s.replace("USDT", ""), {}).get("total", 0) or 0) > 0
                )
        except Exception:
            positions_on_exchange = None

    # Total realized PnL and trade details (when API available)
    total_pnl = None
    trades_with_pnl = []
    if trader and closed_positions:
        trades_with_pnl = compute_trades_with_pnl(trader, closed_positions)
        pnls = [t.get("pnl_usdt") for t in trades_with_pnl if t.get("pnl_usdt") is not None]
        total_pnl = sum(pnls) if pnls else None

    # Total unrealized PnL from open positions (when API available)
    total_unrealized_pnl = None
    if trader and open_positions:
        unrealized_sum = 0.0
        for pos in open_positions:
            current_price = trader.get_current_price(pos["ticker"])
            if current_price is not None:
                entry_value = pos["usdt_value"]
                current_value = pos["quantity"] * current_price
                unrealized_sum += current_value - entry_value
        total_unrealized_pnl = unrealized_sum

    # --- Account summary ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if trader:
            balances = trader.get_account_balance()
            usdt_balance = balances.get("USDT", {}).get("total", 0.0)
            st.metric("USDT Balance", f"${usdt_balance:,.2f}")
        else:
            st.metric("USDT Balance", "N/A (API required)")
    with col2:
        st.metric("Open Positions", len(open_positions))
    with col3:
        st.metric("Closed Trades", len(closed_positions))
    with col4:
        st.metric("Signals (recent)", len(signals))
    with col5:
        if total_pnl is not None:
            st.metric("Total PnL", f"${total_pnl:,.2f}")
        else:
            st.metric("Total PnL", "N/A (API + trades)")
    with col6:
        if total_unrealized_pnl is not None:
            st.metric("Unrealized PnL", f"${total_unrealized_pnl:,.2f}")
        else:
            st.metric("Unrealized PnL", "N/A (API + open positions)")

    # --- Open positions ---
    st.subheader("Open Positions")
    if open_positions:
        rows = []
        for pos in open_positions:
            current_price = trader.get_current_price(pos["ticker"]) if trader else None
            entry_value = pos["usdt_value"]
            current_value = (pos["quantity"] * current_price) if current_price else None
            pnl_usdt = (current_value - entry_value) if current_value else None
            pnl_pct = ((current_value - entry_value) / entry_value * 100) if current_value and entry_value else None
            rows.append({
                "Ticker": pos["ticker"],
                "Entry Price": pos["entry_price"],
                "Quantity": pos["quantity"],
                "Entry Value (USDT)": round(entry_value, 2),
                "Current Price": round(current_price, 4) if current_price else "-",
                "Current Value (USDT)": round(current_value, 2) if current_value else "-",
                "Unrealized PnL (USDT)": round(pnl_usdt, 2) if pnl_usdt is not None else "-",
                "Unrealized PnL (%)": f"{pnl_pct:.2f}%" if pnl_pct is not None else "-",
                "Duration": format_duration(pos["opened_at"]),
                "Opened At": pos["opened_at"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

    # --- Closed positions (with PnL if API available) ---
    st.subheader("Closed Trades")
    if closed_positions:
        if trader:
            rows = []
            total_pnl = 0
            wins = 0
            for t in trades_with_pnl:
                pnl = t.get("pnl_usdt")
                if pnl is not None:
                    total_pnl += pnl
                    if pnl > 0:
                        wins += 1
                rows.append({
                    "Ticker": t["ticker"],
                    "Entry": round(t["entry_price"], 4),
                    "Exit": round(t["exit_price"], 4) if t.get("exit_price") else "-",
                    "Quantity": t["quantity"],
                    "PnL (USDT)": round(t["pnl_usdt"], 2) if t.get("pnl_usdt") is not None else "-",
                    "PnL (%)": f"{t['pnl_pct']:.2f}%" if t.get("pnl_pct") is not None else "-",
                    "Exit Reason": t.get("exit_reason", ""),
                    "Closed At": t["closed_at"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if trades_with_pnl and any(t.get("pnl_usdt") is not None for t in trades_with_pnl):
                win_count = sum(1 for t in trades_with_pnl if t.get("pnl_usdt") and t["pnl_usdt"] > 0)
                total_count = sum(1 for t in trades_with_pnl if t.get("pnl_usdt") is not None)
                st.caption(f"Total PnL: ${total_pnl:,.2f} | Win rate: {win_count}/{total_count} ({100*win_count/total_count:.1f}%)" if total_count else "")
        else:
            df = pd.DataFrame(closed_positions)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades yet.")

    # --- Signals history ---
    st.subheader("Recent Signals")
    if signals:
        df_sig = pd.DataFrame(signals)
        st.dataframe(df_sig[["ticker", "signal_type", "source", "created_at"]], use_container_width=True, hide_index=True)
    else:
        st.info("No signals recorded.")

    # --- Simple equity curve (if we have closed trades with PnL) ---
    if trader and closed_positions:
        trades_with_pnl = compute_trades_with_pnl(trader, closed_positions)
        valid = [t for t in trades_with_pnl if t.get("pnl_usdt") is not None]
        if valid:
            st.subheader("Equity Curve (Cumulative PnL)")
            df_eq = pd.DataFrame([
                {"closed_at": t["closed_at"], "pnl_usdt": t["pnl_usdt"]} for t in sorted(valid, key=lambda x: x["closed_at"])
            ])
            df_eq["closed_at"] = pd.to_datetime(df_eq["closed_at"])
            df_eq["cumulative_pnl"] = df_eq["pnl_usdt"].cumsum()
            df_eq["balance"] = INITIAL_BALANCE + df_eq["cumulative_pnl"]
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_eq["closed_at"], df_eq["balance"], marker="o", markersize=4)
            ax.set_title("Account Balance (Equity Curve)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Balance (USDT)")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()

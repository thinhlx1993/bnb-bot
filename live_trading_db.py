"""
SQLite persistence for live trading: signals and position history only.
Trading logic uses the exchange (Binance) as source of truth for open positions.
insert_position/close_position record history; positions are not restored from DB for trading.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

Path("logs").mkdir(parents=True, exist_ok=True)
DB_PATH = Path("logs/live_trading.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                source TEXT NOT NULL,
                candle_time TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
            CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);

            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                usdt_value REAL NOT NULL,
                opened_at TEXT NOT NULL,  -- UTC, ISO 8601 with Z
                closed_at TEXT,
                exit_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
            CREATE INDEX IF NOT EXISTS idx_positions_closed ON positions(closed_at);
        """)
        conn.commit()
    finally:
        conn.close()


def insert_signal(ticker: str, signal_type: str, source: str, candle_time: Optional[str] = None) -> None:
    """Record a buy or sell signal. signal_type: 'buy'|'sell', source: 'strategy'|'rl'|'risk_management'."""
    now = datetime.utcnow().isoformat() + "Z"
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO signals (ticker, signal_type, source, candle_time, created_at) VALUES (?, ?, ?, ?, ?)",
            (ticker, signal_type, source, candle_time, now),
        )
        conn.commit()
    finally:
        conn.close()


def insert_position(ticker: str, entry_price: float, quantity: float, usdt_value: float) -> int:
    """Record an opened position. Returns position id."""
    now = datetime.utcnow().isoformat() + "Z"
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO positions (ticker, entry_price, quantity, usdt_value, opened_at) VALUES (?, ?, ?, ?, ?)",
            (ticker, entry_price, quantity, usdt_value, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def close_position(ticker: str, exit_reason: str) -> None:
    """Mark the latest open position for ticker as closed. exit_reason: 'strategy'|'rl'|'risk_management'."""
    now = datetime.utcnow().isoformat() + "Z"
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE positions SET closed_at = ?, exit_reason = ? WHERE ticker = ? AND closed_at IS NULL",
            (now, exit_reason, ticker),
        )
        conn.commit()
    finally:
        conn.close()


def get_open_positions() -> List[Dict[str, Any]]:
    """Return list of open positions (closed_at IS NULL) as dicts with entry_price, quantity, usdt_value, opened_at."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT ticker, entry_price, quantity, usdt_value, opened_at FROM positions WHERE closed_at IS NULL"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def reset_db() -> None:
    """Clear all positions and signals (local state only). Does not affect the exchange."""
    conn = get_conn()
    try:
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM signals")
        conn.commit()
    finally:
        conn.close()


def get_last_signal_per_ticker() -> Dict[str, Dict[str, Any]]:
    """For each ticker, return the latest signal row (signal_type, source, created_at, candle_time)."""
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT ticker, signal_type, source, created_at, candle_time,
                   id FROM signals s1
            WHERE id = (SELECT MAX(id) FROM signals s2 WHERE s2.ticker = s1.ticker)
        """).fetchall()
        return {r["ticker"]: dict(r) for r in rows}
    finally:
        conn.close()

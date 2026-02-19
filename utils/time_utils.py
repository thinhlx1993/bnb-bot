"""Time and interval utilities."""

from datetime import timedelta


def interval_to_timedelta(time_interval: str) -> timedelta:
    """Convert Binance interval string (e.g. '15m', '1h') to timedelta for one candle."""
    if "m" in time_interval:
        minutes = int(time_interval.replace("m", ""))
        return timedelta(minutes=minutes)
    if "h" in time_interval:
        hours = int(time_interval.replace("h", ""))
        return timedelta(hours=hours)
    if "d" in time_interval:
        days = int(time_interval.replace("d", ""))
        return timedelta(days=days)
    return timedelta(minutes=15)  # fallback

"""H2 — Opening Range Breakout (ORB) on intraday SPY hourly bars.

Hypothesis: The first hour establishes a price range. Breakouts ABOVE that range
(in the direction of trend) tend to continue intraday.

Data: SPY 1h bars, 2 years. 7 bars/day (09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30).

Method:
  - Use bar 1 (09:30-10:30) as the "opening range" — its High and Low.
  - At bar 2 (10:30) close, check: if Close > opening range High, go long for the day.
                                     if Close < opening range Low, go short.
  - Optional trend filter: only take longs when daily SMA slope > 0; shorts when < 0.
  - Exit: at end of session (15:30 close).
  - One trade per day max.

This is a different backtest pattern from daily — we run a stateful per-day loop.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.strategy import Strategy
from src.intraday_loader import load_intraday, add_session_columns
from src.metrics import summary, format_summary


class OpeningRangeBreakout(Strategy):
    name = "opening_range_breakout"

    def __init__(self, range_bars=1, signal_bar=2, trend_filter=False,
                 trend_lookback=20, direction="both"):
        """
        range_bars: number of bars to define opening range (1 → first hour only)
        signal_bar: bar index to evaluate breakout (2 → second hour close)
        trend_filter: if True, only take trades aligned with daily trend
        trend_lookback: days for daily SMA used as trend filter
        direction: 'both', 'long', or 'short'
        """
        super().__init__(
            range_bars=range_bars,
            signal_bar=signal_bar,
            trend_filter=trend_filter,
            trend_lookback=trend_lookback,
            direction=direction,
        )

    def generate_signals(self, data):
        """Not used — we override with a custom day loop in backtest_intraday_orb."""
        return data


def backtest_orb(strategy, intraday_data, daily_data=None,
                 initial_capital=10000, slippage=0.0005, commission=1.0):
    """Specialized intraday ORB backtest.

    intraday_data: hourly bars with session columns (use add_session_columns).
    daily_data: daily OHLCV for trend filter, optional.

    Per-day logic:
      1. At end of bar 1 (10:30), record range = [first bar High, first bar Low]
      2. At end of bar 2 (10:30 close) -- WAIT, the bar at 10:30 IS the second bar
         Actually in yfinance: bar at 09:30 covers 09:30-10:30, bar at 10:30 covers 10:30-11:30.
         So "first bar" = 09:30 bar. After it closes (at 10:30), we know its High/Low.
         "Second bar" starts at 10:30. Entry decision: if 09:30 bar's range was broken
         BY 10:30 bar's open OR within 10:30 bar.

      Cleaner version:
      - Bar 0 (09:30-10:30): the opening range. Wait for it to close.
      - At 10:30 (open of next bar = bar 1), evaluate the previous bar's range.
      - If 10:30 open > 09:30 bar's High: long.
        If 10:30 open < 09:30 bar's Low: short.
      - Hold until 15:30 close (session end).
    """
    p = strategy.params
    range_bars = p["range_bars"]
    direction = p["direction"]

    # Add daily trend
    if p["trend_filter"]:
        if daily_data is None:
            raise ValueError("trend_filter=True requires daily_data")
        daily_data = daily_data.copy()
        daily_data["sma"] = daily_data["Close"].rolling(p["trend_lookback"]).mean()
        daily_data["trend_up"] = daily_data["Close"] > daily_data["sma"]

    cash = initial_capital
    equity_records = []
    trade_records = []

    # Group by date
    df = intraday_data.copy()
    df["_date"] = pd.to_datetime(df["date"])

    for date, day_bars in df.groupby("_date"):
        # Need at least range_bars + 1 (range + signal entry bar) + close
        if len(day_bars) < range_bars + 2:
            continue

        # The opening range is the first `range_bars` bars
        range_data = day_bars.iloc[:range_bars]
        rng_high = range_data["High"].max()
        rng_low = range_data["Low"].min()

        # Signal bar: the bar AFTER the range. We use its OPEN as our entry decision.
        signal_bar_idx = range_bars  # 0-indexed
        if signal_bar_idx >= len(day_bars):
            continue
        signal_bar = day_bars.iloc[signal_bar_idx]
        entry_price_raw = signal_bar["Open"]

        # Determine signal
        sig = 0
        if entry_price_raw > rng_high and direction in ("both", "long"):
            sig = 1
        elif entry_price_raw < rng_low and direction in ("both", "short"):
            sig = -1

        if sig == 0:
            equity_records.append({"date": signal_bar.name, "equity": cash})
            continue

        # Trend filter
        if p["trend_filter"]:
            try:
                day_norm = pd.Timestamp(date).normalize()
                # Find the last daily close BEFORE this date for trend filter
                prior_daily = daily_data.loc[daily_data.index < day_norm]
                if len(prior_daily) == 0 or pd.isna(prior_daily["trend_up"].iloc[-1]):
                    equity_records.append({"date": signal_bar.name, "equity": cash})
                    continue
                trend_up = prior_daily["trend_up"].iloc[-1]
                if sig == 1 and not trend_up:
                    equity_records.append({"date": signal_bar.name, "equity": cash})
                    continue
                if sig == -1 and trend_up:
                    equity_records.append({"date": signal_bar.name, "equity": cash})
                    continue
            except Exception:
                continue

        # Execute: enter at signal bar open with slippage, exit at last bar close
        last_bar = day_bars.iloc[-1]
        exit_price_raw = last_bar["Close"]

        if sig == 1:
            entry = entry_price_raw * (1 + slippage)
            exit_p = exit_price_raw * (1 - slippage)
            shares = int(cash * 0.95 / entry)
            if shares > 0:
                pnl = (exit_p - entry) * shares - 2 * commission
                cash += pnl
                trade_records.append({
                    "date": date, "direction": "long",
                    "entry_time": signal_bar.name, "exit_time": last_bar.name,
                    "entry": entry, "exit": exit_p, "shares": shares,
                    "pnl": pnl, "return_pct": (exit_p - entry) / entry,
                    "rng_high": rng_high, "rng_low": rng_low,
                })
        elif sig == -1:
            entry = entry_price_raw * (1 - slippage)
            exit_p = exit_price_raw * (1 + slippage)
            shares = int(cash * 0.95 / entry)
            if shares > 0:
                pnl = (entry - exit_p) * shares - 2 * commission
                cash += pnl
                trade_records.append({
                    "date": date, "direction": "short",
                    "entry_time": signal_bar.name, "exit_time": last_bar.name,
                    "entry": entry, "exit": exit_p, "shares": shares,
                    "pnl": pnl, "return_pct": (entry - exit_p) / entry,
                    "rng_high": rng_high, "rng_low": rng_low,
                })

        equity_records.append({"date": last_bar.name, "equity": cash})

    equity = pd.DataFrame(equity_records).set_index("date")["equity"] if equity_records else pd.Series(dtype=float)
    trades = pd.DataFrame(trade_records)
    return equity, trades


def main():
    print("=" * 70)
    print("H2: OPENING RANGE BREAKOUT — SPY 1h bars, 2 years")
    print("=" * 70)
    intraday = load_intraday("SPY", interval="1h", period="2y")
    intraday = add_session_columns(intraday)
    print(f"Intraday: {len(intraday)} bars, {intraday['date'].nunique()} days")

    # Daily for trend filter
    from src.data_loader import load_daily
    daily = load_daily("SPY", start="2023-01-01", end="2026-12-31")
    print(f"Daily for trend filter: {len(daily)} bars")

    print("\n--- Variant 1: ORB no filter, both directions ---")
    s = OpeningRangeBreakout(range_bars=1, direction="both", trend_filter=False)
    eq, tr = backtest_orb(s, intraday, daily)
    if len(eq) > 0:
        print(format_summary(summary(eq, tr)))
    else:
        print("No trades")

    print("\n--- Variant 2: ORB long-only ---")
    s = OpeningRangeBreakout(range_bars=1, direction="long", trend_filter=False)
    eq, tr = backtest_orb(s, intraday, daily)
    if len(eq) > 0:
        print(format_summary(summary(eq, tr)))

    print("\n--- Variant 3: ORB short-only ---")
    s = OpeningRangeBreakout(range_bars=1, direction="short", trend_filter=False)
    eq, tr = backtest_orb(s, intraday, daily)
    if len(eq) > 0:
        print(format_summary(summary(eq, tr)))

    print("\n--- Variant 4: ORB with trend filter (long when up-trend, short when down) ---")
    s = OpeningRangeBreakout(range_bars=1, direction="both", trend_filter=True, trend_lookback=20)
    eq, tr = backtest_orb(s, intraday, daily)
    if len(eq) > 0:
        print(format_summary(summary(eq, tr)))

    print("\n--- Variant 5: ORB 2-bar range (first 2 hours), with trend filter ---")
    s = OpeningRangeBreakout(range_bars=2, direction="both", trend_filter=True, trend_lookback=20)
    eq, tr = backtest_orb(s, intraday, daily)
    if len(eq) > 0:
        print(format_summary(summary(eq, tr)))


if __name__ == "__main__":
    main()

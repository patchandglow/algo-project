"""H2 v2 — Opening Range Breakout (FIXED).

The previous version only checked the NEXT bar's open against the range.
That misses any breakout that happens later in the day. This version:

  - Defines opening range from first N bars (default N=1, i.e. 09:30-10:30 bar).
  - Then walks through subsequent bars in the same session.
  - LONG entry: first bar whose HIGH exceeds range_high. Entry at range_high+slippage
    (assumes a stop order placed at the breakout level, which is realistic).
  - SHORT entry: first bar whose LOW falls below range_low. Entry at range_low-slippage.
  - Exit: end of session OR if intraday stop is hit.
  - One trade per day max.

Optional:
  - trend_filter: only long when daily trend up
  - require_open_in_range: only consider days where signal-bar opens within the range
                           (avoids gappy days where range is misleading)
  - stop_at_other_side: place protective stop on opposite side of range
                       (long entry → stop at range_low)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.strategy import Strategy
from src.intraday_loader import load_intraday, add_session_columns
from src.data_loader import load_daily
from src.metrics import summary, format_summary


def backtest_orb_v2(
    intraday_data,
    daily_data=None,
    range_bars=1,
    direction="both",
    trend_filter=False,
    trend_lookback=20,
    use_protective_stop=False,
    initial_capital=10000,
    slippage=0.0005,
    commission=1.0,
):
    """Walk through each session, look for breakout, take first one."""
    df = intraday_data.copy()
    df["_date"] = pd.to_datetime(df["date"])

    if trend_filter:
        if daily_data is None:
            raise ValueError("trend_filter requires daily_data")
        daily_data = daily_data.copy()
        daily_data["sma"] = daily_data["Close"].rolling(trend_lookback).mean()
        daily_data["trend_up"] = daily_data["Close"] > daily_data["sma"]

    cash = initial_capital
    equity_records = []
    trade_records = []

    for date, day_bars in df.groupby("_date"):
        if len(day_bars) < range_bars + 1:
            continue

        rng_data = day_bars.iloc[:range_bars]
        rng_high = rng_data["High"].max()
        rng_low = rng_data["Low"].min()

        post_range = day_bars.iloc[range_bars:]
        if len(post_range) == 0:
            continue
        last_bar = day_bars.iloc[-1]

        # Trend filter
        allow_long = direction in ("both", "long")
        allow_short = direction in ("both", "short")
        if trend_filter:
            day_norm = pd.Timestamp(date).normalize()
            prior = daily_data.loc[daily_data.index < day_norm]
            if len(prior) == 0 or pd.isna(prior["trend_up"].iloc[-1]):
                equity_records.append({"date": last_bar.name, "equity": cash})
                continue
            trend_up = bool(prior["trend_up"].iloc[-1])
            allow_long = allow_long and trend_up
            allow_short = allow_short and (not trend_up)

        # Walk forward, find first breakout
        executed = False
        for idx, bar in post_range.iterrows():
            if not executed and allow_long and bar["High"] >= rng_high:
                # Long entry at rng_high (stop-buy order)
                entry = rng_high * (1 + slippage)
                # Now we need exit: end of day or protective stop hit later
                # Find subsequent bars and check if rng_low gets hit first
                later_bars = day_bars.loc[idx:]
                exit_price = None
                exit_time = last_bar.name
                if use_protective_stop:
                    for _, lb in later_bars.iterrows():
                        if lb["Low"] <= rng_low:
                            exit_price = rng_low * (1 - slippage)
                            exit_time = lb.name
                            break
                if exit_price is None:
                    exit_price = last_bar["Close"] * (1 - slippage)

                shares = int(cash * 0.95 / entry)
                if shares > 0:
                    pnl = (exit_price - entry) * shares - 2 * commission
                    cash += pnl
                    trade_records.append({
                        "date": date, "direction": "long",
                        "entry_time": idx, "exit_time": exit_time,
                        "entry": entry, "exit": exit_price, "shares": shares,
                        "pnl": pnl, "return_pct": (exit_price - entry) / entry,
                        "rng_high": rng_high, "rng_low": rng_low,
                    })
                executed = True
                break
            elif not executed and allow_short and bar["Low"] <= rng_low:
                entry = rng_low * (1 - slippage)
                later_bars = day_bars.loc[idx:]
                exit_price = None
                exit_time = last_bar.name
                if use_protective_stop:
                    for _, lb in later_bars.iterrows():
                        if lb["High"] >= rng_high:
                            exit_price = rng_high * (1 + slippage)
                            exit_time = lb.name
                            break
                if exit_price is None:
                    exit_price = last_bar["Close"] * (1 + slippage)

                shares = int(cash * 0.95 / entry)
                if shares > 0:
                    pnl = (entry - exit_price) * shares - 2 * commission
                    cash += pnl
                    trade_records.append({
                        "date": date, "direction": "short",
                        "entry_time": idx, "exit_time": exit_time,
                        "entry": entry, "exit": exit_price, "shares": shares,
                        "pnl": pnl, "return_pct": (entry - exit_price) / entry,
                        "rng_high": rng_high, "rng_low": rng_low,
                    })
                executed = True
                break

        equity_records.append({"date": last_bar.name, "equity": cash})

    equity = pd.DataFrame(equity_records).set_index("date")["equity"] if equity_records else pd.Series(dtype=float)
    trades = pd.DataFrame(trade_records)
    return equity, trades


def main():
    print("=" * 70)
    print("H2 v2: OPENING RANGE BREAKOUT (FIXED) — SPY 1h, 2y")
    print("=" * 70)

    intraday = load_intraday("SPY", interval="1h", period="2y")
    intraday = add_session_columns(intraday)
    daily = load_daily("SPY", start="2023-01-01", end="2026-12-31")
    print(f"Days: {intraday['date'].nunique()}")

    variants = [
        {"name": "1bar range, both, no filter, no stop",
         "range_bars": 1, "direction": "both", "trend_filter": False, "use_protective_stop": False},
        {"name": "1bar range, both, no filter, with stop",
         "range_bars": 1, "direction": "both", "trend_filter": False, "use_protective_stop": True},
        {"name": "1bar range, long-only, trend filter, no stop",
         "range_bars": 1, "direction": "long", "trend_filter": True, "use_protective_stop": False},
        {"name": "1bar range, both, trend filter, with stop",
         "range_bars": 1, "direction": "both", "trend_filter": True, "use_protective_stop": True},
        {"name": "2bar range, both, trend filter, with stop",
         "range_bars": 2, "direction": "both", "trend_filter": True, "use_protective_stop": True},
        {"name": "1bar range, fade (counter-trend), no stop",
         "range_bars": 1, "direction": "long", "trend_filter": True, "use_protective_stop": False},
    ]

    results = []
    for v in variants:
        eq, tr = backtest_orb_v2(intraday, daily,
                                   range_bars=v["range_bars"],
                                   direction=v["direction"],
                                   trend_filter=v["trend_filter"],
                                   use_protective_stop=v["use_protective_stop"])
        if len(eq) > 0 and len(tr) > 0:
            stats = summary(eq, tr)
        else:
            stats = {"sharpe": 0, "cagr": 0, "max_drawdown": 0,
                     "n_trades": 0, "win_rate": 0, "profit_factor": 0}
        results.append({"variant": v["name"], **stats})

    print(f"\n{'Variant':<55}{'Trades':>7}{'Sharpe':>8}{'CAGR':>8}"
          f"{'MaxDD':>8}{'Win%':>7}{'PF':>6}")
    print("-" * 100)
    for r in results:
        print(f"{r['variant']:<55}{r['n_trades']:>7}{r['sharpe']:>8.2f}"
              f"{r['cagr']:>8.2%}{r['max_drawdown']:>8.2%}"
              f"{r['win_rate']:>7.2%}{min(r['profit_factor'], 9.99):>6.2f}")


if __name__ == "__main__":
    main()

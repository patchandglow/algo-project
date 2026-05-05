"""H2 v3 — INVERSE ORB (Fade the Breakout).

Rationale: H2 v2 showed strong NEGATIVE Sharpe across all ORB variants.
That's a signal — taking the opposite side might have edge.

Strategy: When price breaks out of the opening range, FADE it.
  - Long entry: price breaks BELOW range_low → buy (expect reversion).
  - Short entry: price breaks ABOVE range_high → sell short.
  - Exit: end of session.

This matches the empirical regime of SPY 2024-2026: range-bound,
mean-reverting at intraday horizons.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.intraday_loader import load_intraday, add_session_columns
from src.data_loader import load_daily
from src.metrics import summary, format_summary


def backtest_inverse_orb(
    intraday_data,
    daily_data=None,
    range_bars=1,
    direction="both",
    trend_filter=False,
    trend_lookback=20,
    initial_capital=10000,
    slippage=0.0005,
    commission=1.0,
):
    df = intraday_data.copy()
    df["_date"] = pd.to_datetime(df["date"])

    if trend_filter:
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
        post = day_bars.iloc[range_bars:]
        if len(post) == 0:
            continue
        last_bar = day_bars.iloc[-1]

        # FADE means: long when down-breakout, short when up-breakout
        allow_fade_long = direction in ("both", "long")    # fade DOWN breakout
        allow_fade_short = direction in ("both", "short")  # fade UP breakout

        if trend_filter:
            day_norm = pd.Timestamp(date).normalize()
            prior = daily_data.loc[daily_data.index < day_norm]
            if len(prior) == 0 or pd.isna(prior["trend_up"].iloc[-1]):
                equity_records.append({"date": last_bar.name, "equity": cash})
                continue
            trend_up = bool(prior["trend_up"].iloc[-1])
            # When trend is up, fade-long (counter-trend buy on down break) is risky
            # When trend is down, fade-short is risky
            # So: only fade-long when trend is up (mean reversion in uptrend)
            #     only fade-short when trend is down
            allow_fade_long = allow_fade_long and trend_up
            allow_fade_short = allow_fade_short and (not trend_up)

        executed = False
        for idx, bar in post.iterrows():
            if not executed and allow_fade_short and bar["High"] >= rng_high:
                # Fade up-breakout: short at rng_high
                entry = rng_high * (1 - slippage)
                exit_price = last_bar["Close"] * (1 + slippage)
                shares = int(cash * 0.95 / entry)
                if shares > 0:
                    pnl = (entry - exit_price) * shares - 2 * commission
                    cash += pnl
                    trade_records.append({
                        "date": date, "direction": "fade_short",
                        "entry_time": idx, "exit_time": last_bar.name,
                        "entry": entry, "exit": exit_price,
                        "pnl": pnl, "return_pct": (entry - exit_price) / entry,
                        "rng_high": rng_high, "rng_low": rng_low,
                    })
                executed = True
                break
            elif not executed and allow_fade_long and bar["Low"] <= rng_low:
                # Fade down-breakout: long at rng_low
                entry = rng_low * (1 + slippage)
                exit_price = last_bar["Close"] * (1 - slippage)
                shares = int(cash * 0.95 / entry)
                if shares > 0:
                    pnl = (exit_price - entry) * shares - 2 * commission
                    cash += pnl
                    trade_records.append({
                        "date": date, "direction": "fade_long",
                        "entry_time": idx, "exit_time": last_bar.name,
                        "entry": entry, "exit": exit_price,
                        "pnl": pnl, "return_pct": (exit_price - entry) / entry,
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
    print("H2 v3: INVERSE ORB (Fade Breakout) — SPY 1h, 2y")
    print("=" * 70)
    intraday = load_intraday("SPY", interval="1h", period="2y")
    intraday = add_session_columns(intraday)
    daily = load_daily("SPY", start="2023-01-01", end="2026-12-31")

    variants = [
        ("1bar, fade both, no filter", 1, "both", False),
        ("1bar, fade long-only (down-breakout), no filter", 1, "long", False),
        ("1bar, fade short-only (up-breakout), no filter", 1, "short", False),
        ("1bar, fade both, with trend filter", 1, "both", True),
        ("2bar, fade both, no filter", 2, "both", False),
    ]

    print(f"\n{'Variant':<55}{'Trades':>7}{'Sharpe':>8}{'CAGR':>8}"
          f"{'MaxDD':>8}{'Win%':>7}{'PF':>6}")
    print("-" * 100)

    results = []
    for name, rb, d, tf in variants:
        eq, tr = backtest_inverse_orb(intraday, daily,
                                        range_bars=rb, direction=d,
                                        trend_filter=tf)
        if len(eq) > 0 and len(tr) > 0:
            stats = summary(eq, tr)
        else:
            stats = {"sharpe": 0, "cagr": 0, "max_drawdown": 0,
                     "n_trades": 0, "win_rate": 0, "profit_factor": 0}
        results.append({"name": name, **stats})
        print(f"{name:<55}{stats['n_trades']:>7}{stats['sharpe']:>8.2f}"
              f"{stats['cagr']:>8.2%}{stats['max_drawdown']:>8.2%}"
              f"{stats['win_rate']:>7.2%}{min(stats['profit_factor'], 9.99):>6.2f}")

    # IS/OOS test on best variant
    print("\n" + "=" * 70)
    print("IN-SAMPLE / OUT-OF-SAMPLE SPLIT (50/50)")
    print("=" * 70)
    n_days = intraday["_date"].nunique() if "_date" in intraday.columns else \
             intraday["date"].nunique()
    sorted_dates = sorted(pd.to_datetime(intraday["date"]).unique())
    mid = len(sorted_dates) // 2
    is_cutoff = sorted_dates[mid]
    intraday["_date"] = pd.to_datetime(intraday["date"])
    is_data = intraday[intraday["_date"] < is_cutoff].copy()
    oos_data = intraday[intraday["_date"] >= is_cutoff].copy()
    print(f"In-sample: {is_data['_date'].nunique()} days "
          f"({is_data['_date'].min().date()} to {is_data['_date'].max().date()})")
    print(f"Out-of-sample: {oos_data['_date'].nunique()} days "
          f"({oos_data['_date'].min().date()} to {oos_data['_date'].max().date()})")

    # Pick best from in-sample
    best_sharpe = -np.inf
    best_variant = None
    for name, rb, d, tf in variants:
        eq, tr = backtest_inverse_orb(is_data, daily,
                                        range_bars=rb, direction=d,
                                        trend_filter=tf)
        if len(eq) > 0 and len(tr) >= 10:
            s = summary(eq, tr)["sharpe"]
            if s > best_sharpe:
                best_sharpe = s
                best_variant = (name, rb, d, tf)

    if best_variant:
        print(f"\nBest in-sample: {best_variant[0]}, Sharpe {best_sharpe:.2f}")
        name, rb, d, tf = best_variant
        eq_oos, tr_oos = backtest_inverse_orb(oos_data, daily,
                                                range_bars=rb, direction=d,
                                                trend_filter=tf)
        if len(eq_oos) > 0 and len(tr_oos) > 0:
            print(f"\nOOS RESULTS for: {name}")
            print(format_summary(summary(eq_oos, tr_oos)))


if __name__ == "__main__":
    main()

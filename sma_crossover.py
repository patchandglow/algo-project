"""H1 — Overnight Gap Mean Reversion.

Hypothesis: After large overnight gaps, price partially reverts during the
regular session. Tested as a daily strategy: enter at open if overnight gap
exceeds threshold, exit at close.

Mechanism: Overnight markets are thin. Liquidity returns during regular session
and reverts overshoots. Effect documented for SPY (Lou-Polk-Skouras 2019,
Berkman et al. 2012).

This implementation uses daily OHLC. With proper futures intraday data we'd
get tighter execution and better signal/noise. Here we test if the basic
phenomenon exists at all.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.strategy import Strategy
from src.data_loader import load_daily
from src.metrics import summary, format_summary


class OvernightGapReversion(Strategy):
    """Fade overnight gaps that exceed a sigma threshold.

    Enter at today's open, exit at today's close.
    Long when overnight gap is sharply DOWN (-N sigma), short when gap is sharply UP.
    """
    name = "overnight_gap_reversion"

    def __init__(self, gap_sigma_threshold=1.5, vol_lookback=60,
                 vix_filter=None, direction="both"):
        super().__init__(
            gap_sigma_threshold=gap_sigma_threshold,
            vol_lookback=vol_lookback,
            vix_filter=vix_filter,
            direction=direction,
        )

    def generate_signals(self, data):
        df = data.copy()
        # Overnight return = (Open - prev Close) / prev Close
        df["prev_close"] = df["Close"].shift(1)
        df["overnight_ret"] = (df["Open"] - df["prev_close"]) / df["prev_close"]

        # Rolling stdev of overnight returns to normalize
        df["on_vol"] = df["overnight_ret"].rolling(
            self.params["vol_lookback"]
        ).std()
        df["gap_z"] = df["overnight_ret"] / df["on_vol"]

        threshold = self.params["gap_sigma_threshold"]
        direction = self.params["direction"]

        # Signal = position to ENTER at today's open
        df["entry_signal"] = 0
        if direction in ("both", "long"):
            df.loc[df["gap_z"] <= -threshold, "entry_signal"] = 1
        if direction in ("both", "short"):
            df.loc[df["gap_z"] >= threshold, "entry_signal"] = -1

        # We hold from open to close on the same day
        df["signal"] = df["entry_signal"]
        return df


def backtest_intraday_open_to_close(strategy, data, initial_capital=10000,
                                      commission=1.0, slippage=0.0005):
    """Specialized backtest for open-to-close strategies.

    Standard backtest engine assumes overnight holding. This one enters at open,
    exits at close on the same bar.
    """
    signals = strategy.generate_signals(data)

    cash = initial_capital
    equity_records = []
    trade_records = []

    for date, row in signals.iterrows():
        signal = row["signal"]
        open_price = row["Open"]
        close_price = row["Close"]

        if signal == 0 or pd.isna(signal):
            equity_records.append({"date": date, "equity": cash})
            continue

        # Long: enter at open*(1+slip), exit at close*(1-slip)
        # Short: enter at open*(1-slip) [sell short], exit at close*(1+slip) [cover]
        if signal == 1:
            entry = open_price * (1 + slippage)
            exit_p = close_price * (1 - slippage)
            shares = int(cash * 0.95 / entry)
            if shares > 0:
                pnl = (exit_p - entry) * shares - 2 * commission
                cash += pnl
                trade_records.append({
                    "entry_date": date, "exit_date": date,
                    "entry_price": entry, "exit_price": exit_p,
                    "shares": shares, "pnl": pnl,
                    "return_pct": (exit_p - entry) / entry,
                    "direction": "long",
                })
        elif signal == -1:
            entry = open_price * (1 - slippage)
            exit_p = close_price * (1 + slippage)
            shares = int(cash * 0.95 / entry)
            if shares > 0:
                pnl = (entry - exit_p) * shares - 2 * commission
                cash += pnl
                trade_records.append({
                    "entry_date": date, "exit_date": date,
                    "entry_price": entry, "exit_price": exit_p,
                    "shares": shares, "pnl": pnl,
                    "return_pct": (entry - exit_p) / entry,
                    "direction": "short",
                })

        equity_records.append({"date": date, "equity": cash})

    equity_df = pd.DataFrame(equity_records).set_index("date")["equity"]
    trades_df = pd.DataFrame(trade_records)
    return equity_df, trades_df


def main():
    print("=" * 60)
    print("H1: OVERNIGHT GAP MEAN REVERSION — Initial Investigation")
    print("=" * 60)

    print("\nLoading SPY 2010-2024...")
    data = load_daily("SPY", start="2010-01-01", end="2024-12-31")

    # Test variants: long-only, short-only, both directions
    for direction in ["both", "long", "short"]:
        print(f"\n--- Direction: {direction} ---")
        strategy = OvernightGapReversion(
            gap_sigma_threshold=1.5,
            vol_lookback=60,
            direction=direction,
        )
        equity, trades = backtest_intraday_open_to_close(
            strategy, data, initial_capital=10000
        )
        stats = summary(equity, trades)
        print(format_summary(stats))

    # Threshold sweep
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP (long+short, 60-day vol lookback)")
    print("=" * 60)
    print(f"{'Threshold':>10} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinRate':>8}")
    for thresh in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        strategy = OvernightGapReversion(
            gap_sigma_threshold=thresh, direction="both"
        )
        equity, trades = backtest_intraday_open_to_close(strategy, data)
        stats = summary(equity, trades)
        print(f"{thresh:>10.1f} {stats['cagr']:>8.2%} "
              f"{stats['sharpe']:>8.2f} {stats['max_drawdown']:>8.2%} "
              f"{stats['n_trades']:>8d} {stats['win_rate']:>8.2%}")


if __name__ == "__main__":
    main()

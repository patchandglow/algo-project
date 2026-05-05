"""H1B — Overnight Gap CONTINUATION (the inverse of the original H1).

Diagnostic showed gaps continue rather than reverse on SPY.
- Gaps ≥ +2σ: intraday mean +0.155%, 66% win rate, N=95
- Gaps ≥ +1σ: intraday mean +0.087%, 62% win rate, N=476

This strategy: BUY at open after a strong upward gap, SELL at close.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.strategy import Strategy
from src.data_loader import load_daily
from src.metrics import summary, format_summary
from strategies.overnight_gap import backtest_intraday_open_to_close


class GapContinuation(Strategy):
    """Long-only: enter at open if overnight gap up ≥ threshold sigma. Exit at close."""
    name = "gap_continuation"

    def __init__(self, gap_sigma_threshold=1.5, vol_lookback=60):
        super().__init__(
            gap_sigma_threshold=gap_sigma_threshold,
            vol_lookback=vol_lookback,
        )

    def generate_signals(self, data):
        df = data.copy()
        df["prev_close"] = df["Close"].shift(1)
        df["overnight_ret"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
        df["on_vol"] = df["overnight_ret"].rolling(
            self.params["vol_lookback"]
        ).std()
        df["gap_z"] = df["overnight_ret"] / df["on_vol"]

        threshold = self.params["gap_sigma_threshold"]
        df["signal"] = 0
        df.loc[df["gap_z"] >= threshold, "signal"] = 1
        return df


def main():
    print("=" * 60)
    print("H1B: GAP CONTINUATION (inverse of failed H1)")
    print("=" * 60)
    data = load_daily("SPY", start="2010-01-01", end="2024-12-31")

    print("\nThreshold sweep (long-only gap continuation):")
    print(f"{'Threshold':>10} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} "
          f"{'Trades':>8} {'WinRate':>8} {'PF':>6}")
    results = []
    for thresh in [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        strategy = GapContinuation(gap_sigma_threshold=thresh)
        equity, trades = backtest_intraday_open_to_close(strategy, data)
        stats = summary(equity, trades)
        results.append({"threshold": thresh, **stats})
        print(f"{thresh:>10.2f} {stats['cagr']:>8.2%} "
              f"{stats['sharpe']:>8.2f} {stats['max_drawdown']:>8.2%} "
              f"{stats['n_trades']:>8d} {stats['win_rate']:>8.2%} "
              f"{stats['profit_factor']:>6.2f}")

    # In/out-of-sample split: train on 2010-2019, test on 2020-2024
    print("\n" + "=" * 60)
    print("IN-SAMPLE (2010-2019) vs OUT-OF-SAMPLE (2020-2024) split")
    print("Best in-sample threshold deployed on test set")
    print("=" * 60)
    train = data.loc["2010-01-01":"2019-12-31"]
    test = data.loc["2020-01-01":"2024-12-31"]

    # Find best in-sample threshold
    best_sharpe = -np.inf
    best_thresh = None
    for thresh in [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        s = GapContinuation(gap_sigma_threshold=thresh)
        eq, tr = backtest_intraday_open_to_close(s, train)
        st = summary(eq, tr)
        if st["sharpe"] > best_sharpe and st["n_trades"] >= 20:
            best_sharpe = st["sharpe"]
            best_thresh = thresh

    print(f"\nBest in-sample threshold: {best_thresh} (Sharpe {best_sharpe:.2f})")

    # Apply to test
    strategy = GapContinuation(gap_sigma_threshold=best_thresh)
    eq_test, tr_test = backtest_intraday_open_to_close(strategy, test)
    st_test = summary(eq_test, tr_test)
    print(f"\nOOS RESULTS (test = 2020-2024):")
    print(format_summary(st_test))


if __name__ == "__main__":
    main()

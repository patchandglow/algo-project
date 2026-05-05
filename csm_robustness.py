"""Sprint 1 reference: 50/200 SMA crossover. Not expected to be profitable."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.strategy import Strategy
from src.data_loader import load_daily
from src.backtest import run_backtest
from src.metrics import summary, format_summary
from src.plots import plot_equity_curve


class SMACrossover(Strategy):
    name = "sma_crossover"

    def __init__(self, fast=50, slow=200):
        super().__init__(fast=fast, slow=slow)
        if fast >= slow:
            raise ValueError("fast must be less than slow")

    def generate_signals(self, data):
        df = data.copy()
        df["sma_fast"] = df["Close"].rolling(self.params["fast"]).mean()
        df["sma_slow"] = df["Close"].rolling(self.params["slow"]).mean()
        df["signal"] = 0
        df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
        return df


def main():
    print("Loading SPY...")
    data = load_daily("SPY", start="2010-01-01", end="2024-12-31")
    print(f"Loaded {len(data)} bars")

    strategy = SMACrossover(fast=50, slow=200)
    print(f"Running backtest: {strategy}")

    equity, trades = run_backtest(strategy, data, initial_capital=10000)
    stats = summary(equity, trades)
    print(format_summary(stats))

    reports = Path(__file__).parent.parent / "reports"
    reports.mkdir(exist_ok=True)

    trades.to_csv(reports / f"{strategy.name}_trades.csv", index=False)

    buy_hold_equity = data["Close"] / data["Close"].iloc[0] * 10000
    plot_equity_curve(
        equity,
        save_path=reports / f"{strategy.name}_equity.png",
        title=f"{strategy.name} on SPY",
        benchmark=buy_hold_equity,
    )

    bh_stats = summary(buy_hold_equity, pd.DataFrame({"pnl": [buy_hold_equity.iloc[-1] - 10000]}))
    print("\nBUY AND HOLD COMPARISON:")
    print(format_summary(bh_stats))

    return stats, bh_stats


if __name__ == "__main__":
    main()

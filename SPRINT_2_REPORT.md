"""Generate Sprint 2 final report with all visualizations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.data_loader import load_daily
from src.backtest import run_backtest
from src.metrics import summary
from src.walkforward import walk_forward
from src.montecarlo import monte_carlo_trade_resample
from src.sweep import parameter_sweep
from strategies.sma_crossover import SMACrossover
from strategies.gap_continuation import GapContinuation
from strategies.overnight_gap import (
    OvernightGapReversion, backtest_intraday_open_to_close
)


REPORTS = Path(__file__).parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)


def fig1_sma_vs_buyhold(data):
    strategy = SMACrossover(50, 200)
    equity, _ = run_backtest(strategy, data, initial_capital=10000)
    bh = data["Close"] / data["Close"].iloc[0] * 10000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity.index, equity.values, label="SMA(50,200)", linewidth=1.5)
    ax1.plot(bh.index, bh.values, label="Buy & Hold", linewidth=1.0, alpha=0.7)
    ax1.set_title("Figure 1: SMA Crossover vs Buy & Hold (SPY 2010-2024)")
    ax1.set_ylabel("Equity ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    eq_dd = (equity - equity.cummax()) / equity.cummax()
    bh_dd = (bh - bh.cummax()) / bh.cummax()
    ax2.fill_between(eq_dd.index, eq_dd.values, 0, color="blue", alpha=0.3, label="SMA DD")
    ax2.fill_between(bh_dd.index, bh_dd.values, 0, color="red", alpha=0.3, label="B&H DD")
    ax2.set_ylabel("Drawdown")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS / "fig1_sma_vs_buyhold.png", dpi=120)
    plt.close()
    print("Saved: fig1_sma_vs_buyhold.png")


def fig2_overnight_diagnostic(data):
    df = data.copy()
    df["prev_close"] = df["Close"].shift(1)
    df["overnight_ret"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
    df["intraday_ret"] = (df["Close"] - df["Open"]) / df["Open"]
    df["on_vol"] = df["overnight_ret"].rolling(60).std()
    df["gap_z"] = df["overnight_ret"] / df["on_vol"]
    df = df.dropna()

    bins = [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf]
    labels = ["<-2σ", "[-2,-1)", "[-1,-0.5)", "[-0.5,0)",
              "[0,0.5)", "[0.5,1)", "[1,2)", "≥2σ"]
    df["gap_bucket"] = pd.cut(df["gap_z"], bins=bins, labels=labels)
    grouped = df.groupby("gap_bucket", observed=True)["intraday_ret"]
    means = grouped.mean()
    counts = grouped.count()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["red" if m < 0 else "green" for m in means.values]
    ax1.bar(range(len(means)), means.values * 10000, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(means)))
    ax1.set_xticklabels(means.index.astype(str), rotation=45)
    ax1.set_ylabel("Mean Intraday Return (bps)")
    ax1.set_title("Figure 2a: Intraday Return by Overnight Gap Bucket\n"
                   "(Continuation pattern visible at >+1σ)")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(range(len(counts)), counts.values, color="steelblue", alpha=0.7)
    ax2.set_xticks(range(len(counts)))
    ax2.set_xticklabels(counts.index.astype(str), rotation=45)
    ax2.set_ylabel("N observations")
    ax2.set_title("Figure 2b: Sample Size by Bucket")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(REPORTS / "fig2_overnight_diagnostic.png", dpi=120)
    plt.close()
    print("Saved: fig2_overnight_diagnostic.png")


def fig3_walkforward_sharpe(data):
    def factory(train_data):
        return SMACrossover(fast=50, slow=200)

    folds, _ = walk_forward(factory, data, train_window=252*3,
                              test_window=252, step=252)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["green" if s > 0 else "red" for s in folds["sharpe"]]
    ax.bar(range(len(folds)), folds["sharpe"].values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([str(d.year) for d in folds["test_start"]], rotation=45)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(folds["sharpe"].mean(), color="blue", linestyle="--",
                label=f"Mean: {folds['sharpe'].mean():.2f}")
    ax.set_ylabel("OOS Sharpe Ratio")
    ax.set_xlabel("Test Year")
    ax.set_title("Figure 3: Walk-Forward OOS Sharpe by Fold (SMA 50/200)\n"
                  "High variance across regimes — typical for trend-following")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(REPORTS / "fig3_walkforward.png", dpi=120)
    plt.close()
    print("Saved: fig3_walkforward.png")


def fig4_monte_carlo(data):
    strategy = SMACrossover(50, 200)
    _, trades = run_backtest(strategy, data)
    mc = monte_carlo_trade_resample(trades, n_simulations=1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(mc["drawdowns"] * 100, bins=50, color="red", alpha=0.7, edgecolor="black")
    ax1.axvline(mc["drawdown_p5"] * 100, color="blue", linestyle="--",
                 label=f"P5: {mc['drawdown_p5']*100:.1f}%")
    ax1.axvline(mc["drawdown_p50"] * 100, color="black", linestyle="--",
                 label=f"P50: {mc['drawdown_p50']*100:.1f}%")
    ax1.axvline(mc["drawdown_p95"] * 100, color="darkred", linestyle="--",
                 label=f"P95: {mc['drawdown_p95']*100:.1f}%")
    ax1.set_xlabel("Max Drawdown (%)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Figure 4a: Monte Carlo Drawdown Distribution\n(1000 trade-order permutations)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Median curve
    ax2.plot(mc["median_curve"], label="Median path", color="black")
    ax2.fill_between(range(len(mc["median_curve"])),
                       mc["p5_curve"], mc["p95_curve"],
                       alpha=0.3, color="steelblue", label="P5-P95 band")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Equity ($)")
    ax2.set_title("Figure 4b: Equity Path Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS / "fig4_monte_carlo.png", dpi=120)
    plt.close()
    print("Saved: fig4_monte_carlo.png")


def fig5_threshold_sweep(data):
    """Show that gap continuation has positive PF only at extreme thresholds
    AND that does not survive OOS."""
    thresholds = [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    sharpes = []
    pfs = []
    n_trades = []
    for t in thresholds:
        s = GapContinuation(gap_sigma_threshold=t)
        eq, tr = backtest_intraday_open_to_close(s, data)
        st = summary(eq, tr)
        sharpes.append(st["sharpe"])
        pfs.append(min(st["profit_factor"], 2.0))  # cap for plot
        n_trades.append(st["n_trades"])

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, sharpes, "o-", color="steelblue", label="Sharpe")
    ax1.set_xlabel("Gap σ Threshold")
    ax1.set_ylabel("Sharpe", color="steelblue")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.bar(thresholds, n_trades, alpha=0.3, color="gray", width=0.15, label="N trades")
    ax2.set_ylabel("N trades", color="gray")

    ax1.set_title("Figure 5: Gap Continuation — Threshold Sweep on SPY\n"
                   "Marginal positive Sharpe at high thresholds, but trade count too low — KILLED")
    plt.tight_layout()
    plt.savefig(REPORTS / "fig5_gap_continuation_sweep.png", dpi=120)
    plt.close()
    print("Saved: fig5_gap_continuation_sweep.png")


def main():
    print("Generating Sprint 2 visual reports...")
    data = load_daily("SPY", start="2010-01-01", end="2024-12-31")

    fig1_sma_vs_buyhold(data)
    fig2_overnight_diagnostic(data)
    fig3_walkforward_sharpe(data)
    fig4_monte_carlo(data)
    fig5_threshold_sweep(data)

    print(f"\nAll reports in: {REPORTS}/")


if __name__ == "__main__":
    main()

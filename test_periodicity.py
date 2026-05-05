"""Sprint 3 robustness gauntlet for the survivor: Cross-Sectional Momentum.

Tests CSM(top-K, lookback) under:
  1. Parameter sensitivity (full grid, plateau check)
  2. Walk-forward analysis (rolling train/test windows)
  3. Monte Carlo on monthly returns (block bootstrap)
  4. Regime breakdown (year by year)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.metrics import summary, format_summary, sharpe_ratio
from strategies.csm import cross_sectional_momentum, load_universe


def parameter_sensitivity(prices):
    print("=" * 70)
    print("1. PARAMETER SENSITIVITY")
    print("=" * 70)
    results = []
    for k in range(2, 8):
        for lb in [3, 6, 9, 12, 15, 18]:
            eq, tr, _ = cross_sectional_momentum(prices, top_k=k, lookback_months=lb)
            s = summary(eq, tr)
            results.append({"k": k, "lookback": lb, "sharpe": s["sharpe"],
                            "cagr": s["cagr"], "maxdd": s["max_drawdown"]})
    df = pd.DataFrame(results)

    # Pivot for heatmap
    sharpe_pivot = df.pivot(index="k", columns="lookback", values="sharpe")
    print("\nSharpe heatmap (rows=top_k, cols=lookback months):")
    print(sharpe_pivot.round(2).to_string())

    # Plateau check: if best params are isolated peaks, that's overfitting
    best = df.loc[df["sharpe"].idxmax()]
    print(f"\nBest: top-{int(best['k'])}, {int(best['lookback'])}m, Sharpe {best['sharpe']:.2f}")

    # Range of Sharpe across all params
    print(f"Sharpe range across grid: {df['sharpe'].min():.2f} to {df['sharpe'].max():.2f}")
    print(f"Sharpe std across grid:   {df['sharpe'].std():.2f}")
    if df["sharpe"].min() > 0:
        print(">>> All grid points positive — strong plateau, robust signal")
    elif df["sharpe"].min() > -0.2:
        print(">>> Most points positive — moderate plateau")
    else:
        print(">>> Wide range — some overfitting likely if best is much higher than median")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(sharpe_pivot.values, cmap="RdYlGn", aspect="auto",
                    vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(len(sharpe_pivot.columns)))
    ax.set_xticklabels(sharpe_pivot.columns)
    ax.set_yticks(range(len(sharpe_pivot.index)))
    ax.set_yticklabels(sharpe_pivot.index)
    ax.set_xlabel("Lookback (months)")
    ax.set_ylabel("Top-K")
    ax.set_title("Figure 8: CSM Parameter Sensitivity (Sharpe)")
    for i in range(len(sharpe_pivot.index)):
        for j in range(len(sharpe_pivot.columns)):
            ax.text(j, i, f"{sharpe_pivot.values[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / "reports" / "fig8_csm_sensitivity.png", dpi=120)
    plt.close()
    print("\nSaved: fig8_csm_sensitivity.png")
    return df


def walk_forward_csm(prices, top_k=2, lookback_months=12, train_years=3, test_years=1):
    """Walk-forward CSM over multiple folds."""
    print("\n" + "=" * 70)
    print("2. WALK-FORWARD ANALYSIS")
    print("=" * 70)
    print(f"Strategy: top-{top_k}, {lookback_months}m lookback")
    print(f"Train: {train_years}y, Test: {test_years}y, Step: {test_years}y")
    print()

    start_year = prices.index[0].year + train_years
    end_year = prices.index[-1].year

    folds = []
    for test_start_year in range(start_year, end_year):
        test_start = pd.Timestamp(f"{test_start_year}-01-01")
        test_end = pd.Timestamp(f"{test_start_year + test_years}-01-01")
        # Need lookback warmup before test
        warmup_start = test_start - pd.DateOffset(months=lookback_months + 1)
        test_slice = prices.loc[warmup_start:test_end]
        if len(test_slice) < lookback_months * 22:
            continue
        eq, tr, _ = cross_sectional_momentum(test_slice, top_k=top_k,
                                               lookback_months=lookback_months)
        eq = eq.loc[test_start:]
        if len(eq) < 2:
            continue
        eq = eq / eq.iloc[0] * 10000
        oos_trades = tr[(tr["date"] >= test_start) & (tr["date"] < test_end)] if "date" in tr.columns else tr
        s = summary(eq, oos_trades)
        folds.append({"year": test_start_year, **s})

    folds_df = pd.DataFrame(folds)
    print(f"{'Year':>6}{'CAGR':>10}{'Sharpe':>10}{'MaxDD':>10}{'Trades':>8}")
    for _, row in folds_df.iterrows():
        print(f"{row['year']:>6}{row['cagr']:>10.2%}{row['sharpe']:>10.2f}"
              f"{row['max_drawdown']:>10.2%}{int(row['n_trades']):>8d}")
    print(f"\nMean OOS Sharpe: {folds_df['sharpe'].mean():.2f}")
    print(f"Std OOS Sharpe:  {folds_df['sharpe'].std():.2f}")
    print(f"% positive years: {(folds_df['sharpe'] > 0).mean():.1%}")

    return folds_df


def monthly_block_bootstrap(prices, top_k=2, lookback_months=12, n_sims=1000):
    """Bootstrap monthly returns to estimate distribution of outcomes."""
    print("\n" + "=" * 70)
    print("3. MONTHLY BLOCK BOOTSTRAP")
    print("=" * 70)
    eq, tr, _ = cross_sectional_momentum(prices, top_k=top_k,
                                            lookback_months=lookback_months)
    monthly_rets = eq.pct_change().dropna().values

    if len(monthly_rets) < 12:
        print("Too few months for bootstrap")
        return

    rng = np.random.default_rng(42)
    block_size = 6
    horizon = 60  # 5 years
    finals = []
    sharpes = []
    drawdowns = []

    for _ in range(n_sims):
        n_blocks = (horizon + block_size - 1) // block_size
        starts = rng.integers(0, len(monthly_rets) - block_size + 1, size=n_blocks)
        sample = np.concatenate([monthly_rets[s:s+block_size] for s in starts])[:horizon]
        cum = np.cumprod(1 + sample)
        finals.append(cum[-1])
        sharpes.append(np.mean(sample) / np.std(sample) * np.sqrt(12) if np.std(sample) > 0 else 0)
        running_max = np.maximum.accumulate(cum)
        running_max = np.maximum(running_max, 1.0)
        dd = (cum - running_max) / running_max
        drawdowns.append(dd.min())

    finals = np.array(finals)
    sharpes = np.array(sharpes)
    drawdowns = np.array(drawdowns)

    print(f"5-year final wealth (start = $1):")
    print(f"  P5:  ${np.percentile(finals, 5):.2f}")
    print(f"  P50: ${np.percentile(finals, 50):.2f}")
    print(f"  P95: ${np.percentile(finals, 95):.2f}")
    print(f"\n5-year drawdown distribution:")
    print(f"  P5  (best):   {np.percentile(drawdowns, 5):.2%}")
    print(f"  P50:          {np.percentile(drawdowns, 50):.2%}")
    print(f"  P95 (worst):  {np.percentile(drawdowns, 95):.2%}")
    print(f"\nSharpe distribution:")
    print(f"  P5:  {np.percentile(sharpes, 5):.2f}")
    print(f"  P50: {np.percentile(sharpes, 50):.2f}")
    print(f"  P95: {np.percentile(sharpes, 95):.2f}")


def yearly_breakdown(prices, top_k=2, lookback_months=12):
    """Show year-by-year returns vs SPY."""
    print("\n" + "=" * 70)
    print("4. YEAR-BY-YEAR BREAKDOWN")
    print("=" * 70)
    eq, _, _ = cross_sectional_momentum(prices, top_k=top_k,
                                            lookback_months=lookback_months)
    spy = prices.get("SPY")
    if spy is None:
        from src.data_loader import load_daily
        spy_df = load_daily("SPY", start=str(prices.index[0].date()),
                             end=str(prices.index[-1].date()))
        spy = spy_df["Close"]

    csm_yearly = eq.resample("YE").last().pct_change().dropna()
    spy_aligned = spy.reindex(eq.index, method="ffill")
    spy_yearly = spy_aligned.resample("YE").last().pct_change().dropna()

    print(f"{'Year':>6}{'CSM':>10}{'SPY':>10}{'Diff':>10}")
    for year in csm_yearly.index.year:
        csm_r = csm_yearly[csm_yearly.index.year == year].iloc[0]
        spy_r = spy_yearly[spy_yearly.index.year == year].iloc[0] if any(spy_yearly.index.year == year) else 0
        print(f"{year:>6}{csm_r:>10.2%}{spy_r:>10.2%}{csm_r - spy_r:>+10.2%}")


def main():
    prices = load_universe(start="2015-01-01", end="2024-12-31")
    print(f"\nUniverse loaded: {prices.shape}")

    # 1. Parameter sensitivity
    parameter_sensitivity(prices)

    # 2. Walk-forward (use plateau-stable params, not absolute best)
    walk_forward_csm(prices, top_k=3, lookback_months=12, train_years=3, test_years=1)

    # 3. Bootstrap
    monthly_block_bootstrap(prices, top_k=3, lookback_months=12)

    # 4. Year-by-year
    yearly_breakdown(prices, top_k=3, lookback_months=12)


if __name__ == "__main__":
    main()

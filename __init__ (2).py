"""Sprint 2 integration: walk-forward + Monte Carlo + parameter sweep on SMA.

Validates that the analytical modules work end-to-end on a known strategy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data_loader import load_daily
from src.backtest import run_backtest
from src.metrics import summary, format_summary
from src.walkforward import walk_forward
from src.montecarlo import monte_carlo_trade_resample, monte_carlo_block_bootstrap
from src.sweep import parameter_sweep, parameter_plateau_score
from src.lockbox import OutOfSampleLockbox
from strategies.sma_crossover import SMACrossover


def run():
    data = load_daily("SPY", start="2010-01-01", end="2024-12-31")
    print(f"Loaded {len(data)} bars")

    # ---- 1. Lockbox setup ----
    print("\n" + "=" * 70)
    print("1. OUT-OF-SAMPLE LOCKBOX")
    print("=" * 70)
    lockbox = OutOfSampleLockbox(data, train_frac=0.70)
    print(lockbox)
    train = lockbox.train

    # ---- 2. Parameter sweep on TRAIN only ----
    print("\n" + "=" * 70)
    print("2. PARAMETER SWEEP — SMA Crossover on TRAIN data only")
    print("=" * 70)
    grid = {"fast": [20, 30, 50, 70, 100], "slow": [150, 200, 250]}
    sweep = parameter_sweep(SMACrossover, grid, train)
    sweep_scored = parameter_plateau_score(sweep, metric="sharpe")

    cols_to_show = ["fast", "slow", "cagr", "sharpe", "max_drawdown",
                    "n_trades", "plateau_score"]
    print(sweep_scored[cols_to_show].head(10).to_string(index=False))

    best = sweep_scored.iloc[0]
    print(f"\nBest in-sample params: fast={best['fast']}, slow={best['slow']}, "
          f"Sharpe={best['sharpe']:.2f}, plateau={best['plateau_score']:.2f}")

    # ---- 3. Walk-forward analysis ----
    print("\n" + "=" * 70)
    print("3. WALK-FORWARD ANALYSIS — SMA on full data, 3y train / 1y test")
    print("=" * 70)

    def factory(train_data):
        # In a more sophisticated version, we'd optimize fast/slow on train_data.
        # For SMA crossover, use fixed canonical params.
        return SMACrossover(fast=50, slow=200)

    folds, full_equity = walk_forward(
        factory, data,
        train_window=252 * 3,
        test_window=252,
        step=252,
    )
    print(f"\n{len(folds)} folds completed")
    print(folds[["fold", "test_start", "test_end", "cagr", "sharpe",
                 "max_drawdown", "n_trades"]].to_string(index=False))
    print(f"\nMean OOS Sharpe across folds: {folds['sharpe'].mean():.2f}")
    print(f"Std  OOS Sharpe across folds: {folds['sharpe'].std():.2f}")

    # ---- 4. Monte Carlo on actual trades ----
    print("\n" + "=" * 70)
    print("4. MONTE CARLO — SMA, 1000 simulations of trade order")
    print("=" * 70)
    strategy = SMACrossover(fast=50, slow=200)
    equity, trades = run_backtest(strategy, data, initial_capital=10000)

    if len(trades) > 0:
        mc = monte_carlo_trade_resample(trades, n_simulations=1000)
        print(f"\nDrawdown distribution:")
        print(f"  P5  (best):    {mc['drawdown_p5']:>8.2%}")
        print(f"  P50 (median):  {mc['drawdown_p50']:>8.2%}")
        print(f"  P95 (worst):   {mc['drawdown_p95']:>8.2%}")
        print(f"\nFinal equity distribution (start = $10,000):")
        print(f"  P5  (worst):   ${mc['final_p5']:>10,.0f}")
        print(f"  P50 (median):  ${mc['final_p50']:>10,.0f}")
        print(f"  P95 (best):    ${mc['final_p95']:>10,.0f}")

    # ---- 5. Block bootstrap on returns ----
    print("\n" + "=" * 70)
    print("5. BLOCK BOOTSTRAP — buy & hold returns, 5-year horizon")
    print("=" * 70)
    daily_rets = data["Close"].pct_change().dropna()
    bb = monte_carlo_block_bootstrap(
        daily_rets, n_simulations=1000, block_size=20,
        horizon=252 * 5, initial_capital=10000,
    )
    print(f"\n5-year drawdown distribution (block-bootstrapped):")
    print(f"  P5  (best):    {bb['drawdown_p5']:>8.2%}")
    print(f"  P50 (median):  {bb['drawdown_p50']:>8.2%}")
    print(f"  P95 (worst):   {bb['drawdown_p95']:>8.2%}")
    print(f"\n5-year final value distribution (start = $10,000):")
    print(f"  P5:    ${bb['final_p5']:>10,.0f}")
    print(f"  P50:   ${bb['final_p50']:>10,.0f}")
    print(f"  P95:   ${bb['final_p95']:>10,.0f}")

    # ---- 6. OOS validation gate ----
    print("\n" + "=" * 70)
    print("6. OUT-OF-SAMPLE GATE")
    print("=" * 70)
    print("(Lockbox would be unlocked here only if a strategy passed all prior gates)")
    print("SMA crossover — verifying with lockbox unlock for demo:")
    test = lockbox.unlock_test(
        strategy_name="sma_crossover_50_200",
        justification="Sprint 2 framework validation, not a real deployment",
    )
    eq_oos, tr_oos = run_backtest(SMACrossover(50, 200), test)
    st_oos = summary(eq_oos, tr_oos)
    print(format_summary(st_oos))


if __name__ == "__main__":
    run()

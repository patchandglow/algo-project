"""Walk-forward analysis. Rolling window train+test to simulate live deployment."""

import pandas as pd
import numpy as np
from src.backtest import run_backtest
from src.metrics import summary


def walk_forward(
    strategy_factory,
    data,
    train_window=252 * 3,
    test_window=252,
    step=252,
    initial_capital=10000,
):
    """Roll a window through history.

    strategy_factory: callable returning a fresh Strategy instance
                      (signature: factory(train_data) -> Strategy)
                      Use this to optimize params on train data.
                      For non-optimizing strategies, ignore train_data.
    data: full DataFrame
    train_window: bars used for fitting/optimization
    test_window: bars used for OOS evaluation per fold
    step: bars to advance between folds

    Returns DataFrame of fold-by-fold metrics + concatenated equity curve.
    """
    if len(data) < train_window + test_window:
        raise ValueError(
            f"Need at least {train_window + test_window} bars, got {len(data)}"
        )

    folds = []
    all_equity = []
    capital = initial_capital
    fold_id = 0

    start = 0
    while start + train_window + test_window <= len(data):
        train = data.iloc[start:start + train_window]
        test = data.iloc[start + train_window:start + train_window + test_window]

        strategy = strategy_factory(train)
        # Need lookback context for indicators on test slice
        # so we feed (train_tail + test) and trim equity to test period
        context_bars = max(252, train_window // 4)
        feed = pd.concat([train.iloc[-context_bars:], test])
        equity, trades = run_backtest(strategy, feed, initial_capital=capital)

        # Trim to OOS portion only
        equity_oos = equity.loc[test.index[0]:]
        if len(equity_oos) < 2:
            start += step
            continue

        # Re-anchor equity so each fold starts fresh from prior end
        equity_oos = equity_oos / equity_oos.iloc[0] * capital
        capital = equity_oos.iloc[-1]

        # Trades that occurred in OOS window
        if len(trades) > 0:
            oos_trades = trades[trades["entry_date"] >= test.index[0]].copy()
        else:
            oos_trades = trades

        stats = summary(equity_oos, oos_trades)
        stats["fold"] = fold_id
        stats["train_start"] = train.index[0]
        stats["train_end"] = train.index[-1]
        stats["test_start"] = test.index[0]
        stats["test_end"] = test.index[-1]
        folds.append(stats)
        all_equity.append(equity_oos)

        fold_id += 1
        start += step

    folds_df = pd.DataFrame(folds)
    if all_equity:
        full_equity = pd.concat(all_equity)
    else:
        full_equity = pd.Series(dtype=float)
    return folds_df, full_equity

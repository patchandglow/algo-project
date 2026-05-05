"""Monte Carlo simulation. Randomize trade order to estimate drawdown distribution."""

import numpy as np
import pandas as pd


def monte_carlo_trade_resample(
    trades: pd.DataFrame,
    initial_capital: float = 10000,
    n_simulations: int = 1000,
    seed: int = 42,
):
    """Randomize trade order, generate equity curve distribution.

    Returns dict with:
        - drawdowns: array of max drawdown per simulation
        - finals: array of final equity per simulation
        - p5, p50, p95: percentiles of equity curves
    """
    if len(trades) == 0:
        return {"drawdowns": np.array([]), "finals": np.array([])}

    rng = np.random.default_rng(seed)
    pnls = trades["pnl"].values
    n_trades = len(pnls)

    drawdowns = np.empty(n_simulations)
    finals = np.empty(n_simulations)
    all_curves = np.empty((n_simulations, n_trades + 1))
    all_curves[:, 0] = initial_capital

    for i in range(n_simulations):
        shuffled = rng.permutation(pnls)
        equity = initial_capital + np.cumsum(shuffled)
        all_curves[i, 1:] = equity

        running_max = np.maximum.accumulate(equity)
        # Initial capital is the starting peak
        running_max = np.maximum(running_max, initial_capital)
        dd = (equity - running_max) / running_max
        drawdowns[i] = dd.min()
        finals[i] = equity[-1]

    return {
        "drawdowns": drawdowns,
        "finals": finals,
        "median_curve": np.median(all_curves, axis=0),
        "p5_curve": np.percentile(all_curves, 5, axis=0),
        "p95_curve": np.percentile(all_curves, 95, axis=0),
        "drawdown_p5": np.percentile(drawdowns, 5),
        "drawdown_p50": np.percentile(drawdowns, 50),
        "drawdown_p95": np.percentile(drawdowns, 95),
        "final_p5": np.percentile(finals, 5),
        "final_p50": np.percentile(finals, 50),
        "final_p95": np.percentile(finals, 95),
    }


def monte_carlo_block_bootstrap(
    returns: pd.Series,
    n_simulations: int = 1000,
    block_size: int = 20,
    horizon: int = 252,
    initial_capital: float = 10000,
    seed: int = 42,
):
    """Block bootstrap from a return series. Preserves short-term autocorrelation.

    Better than trade-shuffling when returns have serial dependence.
    """
    rng = np.random.default_rng(seed)
    rets = returns.values
    n = len(rets)
    if n < block_size:
        block_size = max(1, n // 4)

    n_blocks = (horizon + block_size - 1) // block_size

    finals = np.empty(n_simulations)
    drawdowns = np.empty(n_simulations)

    for i in range(n_simulations):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([rets[s:s + block_size] for s in starts])[:horizon]
        equity = initial_capital * np.cumprod(1 + sample)
        finals[i] = equity[-1]
        running_max = np.maximum.accumulate(equity)
        running_max = np.maximum(running_max, initial_capital)
        dd = (equity - running_max) / running_max
        drawdowns[i] = dd.min()

    return {
        "drawdowns": drawdowns,
        "finals": finals,
        "drawdown_p5": np.percentile(drawdowns, 5),
        "drawdown_p50": np.percentile(drawdowns, 50),
        "drawdown_p95": np.percentile(drawdowns, 95),
        "final_p5": np.percentile(finals, 5),
        "final_p50": np.percentile(finals, 50),
        "final_p95": np.percentile(finals, 95),
    }

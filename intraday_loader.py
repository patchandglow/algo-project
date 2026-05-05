"""Parameter sensitivity. Sweep params, plot heatmap. Real edges have plateaus."""

import itertools
import pandas as pd
import numpy as np
from src.backtest import run_backtest
from src.metrics import summary


def parameter_sweep(strategy_class, param_grid, data, initial_capital=10000):
    """
    strategy_class: callable(**params) -> Strategy
    param_grid: dict of param_name -> list of values
    data: OHLCV DataFrame

    Returns DataFrame with one row per param combination.
    """
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    rows = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        try:
            strategy = strategy_class(**params)
            equity, trades = run_backtest(strategy, data, initial_capital=initial_capital)
            stats = summary(equity, trades)
            row = {**params, **stats}
            rows.append(row)
        except Exception as e:
            row = {**params, "error": str(e)}
            rows.append(row)

    return pd.DataFrame(rows)


def parameter_plateau_score(sweep_df, metric="sharpe", window_pct=0.20):
    """
    Score each parameter combination by how stable it is in its neighborhood.
    Real edges live on plateaus — if perturbing params ±20% degrades the metric a lot,
    the params were just lucky.

    Returns DataFrame with columns: params + metric + plateau_score
    """
    df = sweep_df.copy().dropna(subset=[metric])

    # Identify parameter columns
    param_cols = [c for c in df.columns if c not in {
        "cagr", "annualized_vol", "sharpe", "max_drawdown", "calmar",
        "n_trades", "win_rate", "profit_factor", "avg_win_loss_ratio",
        "final_equity", "total_return", "error", "plateau_score"
    }]

    # For each row, find neighbors within window_pct on every param dim
    scores = []
    for _, row in df.iterrows():
        mask = pd.Series(True, index=df.index)
        for p in param_cols:
            if pd.api.types.is_numeric_dtype(df[p]):
                lo = row[p] * (1 - window_pct)
                hi = row[p] * (1 + window_pct)
                mask &= df[p].between(lo, hi)
        neighbors = df.loc[mask, metric]
        if len(neighbors) > 1:
            # Plateau score: 1 - (range of metric in neighborhood / abs mean)
            spread = neighbors.std()
            mean = abs(neighbors.mean())
            if mean > 1e-9:
                scores.append(1 - min(1.0, spread / mean))
            else:
                scores.append(0.0)
        else:
            scores.append(np.nan)

    df["plateau_score"] = scores
    return df.sort_values(by=metric, ascending=False)

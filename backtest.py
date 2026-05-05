"""Performance metrics. All numbers strategies report go through here.

CRITICAL: All annualization defaults assume DAILY data (TRADING_DAYS=252).
For non-daily data (weekly, monthly), pass `periods_per_year` explicitly.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _periods_per_year(equity_or_returns, default=TRADING_DAYS):
    """Try to infer annualization factor from a DatetimeIndex. Falls back to default."""
    idx = equity_or_returns.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return default
    span_days = (idx[-1] - idx[0]).days
    if span_days <= 0:
        return default
    obs_per_year = (len(idx) - 1) * 365.25 / span_days
    # Snap to known frequencies
    if obs_per_year > 200:
        return TRADING_DAYS  # daily
    if 40 < obs_per_year <= 200:
        return 52              # weekly
    if 20 < obs_per_year <= 40:
        return 26              # bi-weekly
    if 8 < obs_per_year <= 20:
        return 12              # monthly
    if 3 < obs_per_year <= 8:
        return 4               # quarterly
    return max(1, int(obs_per_year))


def cagr(equity, periods_per_year=None):
    if len(equity) < 2:
        return 0.0
    if periods_per_year is None:
        periods_per_year = _periods_per_year(equity)
    n_years = len(equity) / periods_per_year
    if n_years <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    if total <= 0:
        return -1.0
    return total ** (1 / n_years) - 1


def annualized_volatility(returns, periods_per_year=None):
    if len(returns) < 2:
        return 0.0
    if periods_per_year is None:
        periods_per_year = _periods_per_year(returns)
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=None):
    if len(returns) < 2:
        return 0.0
    if periods_per_year is None:
        periods_per_year = _periods_per_year(returns)
    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    sigma = excess.std()
    if sigma < 1e-12:
        return 0.0
    return float((excess.mean() / sigma) * np.sqrt(periods_per_year))


def max_drawdown(equity):
    if len(equity) < 2:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return dd.min()


def calmar_ratio(equity):
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return cagr(equity) / abs(mdd)


def win_rate(trades):
    if len(trades) == 0:
        return 0.0
    return (trades["pnl"] > 0).mean()


def profit_factor(trades):
    if len(trades) == 0:
        return 0.0
    wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def avg_win_loss_ratio(trades):
    if len(trades) == 0:
        return 0.0
    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] < 0, "pnl"]
    if len(losses) == 0:
        return float("inf") if len(wins) > 0 else 0.0
    if len(wins) == 0:
        return 0.0
    return wins.mean() / abs(losses.mean())


def summary(equity, trades):
    returns = equity.pct_change().dropna()
    return {
        "cagr": cagr(equity),
        "annualized_vol": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity),
        "n_trades": len(trades),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "avg_win_loss_ratio": avg_win_loss_ratio(trades),
        "final_equity": equity.iloc[-1] if len(equity) > 0 else 0,
        "total_return": (equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0,
    }


def format_summary(s):
    return "\n".join([
        "=" * 50,
        "BACKTEST SUMMARY",
        "=" * 50,
        f"  CAGR:               {s['cagr']:>10.2%}",
        f"  Annualized vol:     {s['annualized_vol']:>10.2%}",
        f"  Sharpe ratio:       {s['sharpe']:>10.2f}",
        f"  Max drawdown:       {s['max_drawdown']:>10.2%}",
        f"  Calmar ratio:       {s['calmar']:>10.2f}",
        f"  Total return:       {s['total_return']:>10.2%}",
        f"  Final equity:       {s['final_equity']:>10,.0f}",
        f"  Number of trades:   {s['n_trades']:>10d}",
        f"  Win rate:           {s['win_rate']:>10.2%}",
        f"  Profit factor:      {s['profit_factor']:>10.2f}",
        f"  Avg win/loss ratio: {s['avg_win_loss_ratio']:>10.2f}",
        "=" * 50,
    ])

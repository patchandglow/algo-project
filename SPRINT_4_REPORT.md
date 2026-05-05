"""Tests for periodicity handling in metrics. Sprint 3 bug fix."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest

from src.metrics import cagr, sharpe_ratio, annualized_volatility


def test_cagr_explicit_monthly():
    """5-year monthly equity that 2x: CAGR ~ 14.87%."""
    n_months = 60
    equity = pd.Series(np.linspace(10000, 20000, n_months))
    result = cagr(equity, periods_per_year=12)
    assert result == pytest.approx(0.1487, abs=0.01)


def test_cagr_inferred_monthly():
    """Same series with monthly DatetimeIndex — should infer 12 ppy."""
    n_months = 60
    idx = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    equity = pd.Series(np.linspace(10000, 20000, n_months), index=idx)
    result = cagr(equity)
    assert result == pytest.approx(0.1487, abs=0.02)


def test_cagr_inferred_daily():
    """Daily equity: CAGR over 1 year doubling = 100%."""
    idx = pd.date_range("2020-01-01", periods=253, freq="B")
    equity = pd.Series(np.linspace(10000, 20000, 253), index=idx)
    result = cagr(equity)
    assert result == pytest.approx(1.0, abs=0.05)


def test_sharpe_explicit_monthly():
    """Monthly returns with given mean and std produces correct annualized Sharpe."""
    monthly_returns = pd.Series([0.01] * 60)  # constant 1%/mo → std=0
    # Constant returns yields 0 (we hard-zero std=0 case)
    assert sharpe_ratio(monthly_returns, periods_per_year=12) == 0.0


def test_sharpe_explicit_monthly_with_vol():
    """Sharpe of (mean=1%/mo, std=2%/mo) annualized = (0.01/0.02) * sqrt(12) = 1.732."""
    np.random.seed(42)
    monthly_rets = pd.Series(np.random.normal(0.01, 0.02, 1000))
    result = sharpe_ratio(monthly_rets, periods_per_year=12)
    assert result == pytest.approx(1.732, abs=0.2)


def test_volatility_explicit_periodicity():
    """Std=0.02 monthly returns → annualized vol = 0.02*sqrt(12) ≈ 6.93%."""
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0, 0.02, 1000))
    result = annualized_volatility(rets, periods_per_year=12)
    assert result == pytest.approx(0.0693, abs=0.01)


def test_no_explosion_on_monthly():
    """The original bug: monthly equity going 10k→167k over 5y produced CAGR=178k% with default periodicity."""
    n_months = 60
    idx = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    equity = pd.Series([10000 * 1.05 ** i for i in range(n_months)], index=idx)
    # 60 months at 5%/mo. Annualized return ~ (1.05)^12 - 1 = 79.6%
    result = cagr(equity)
    assert result < 1.0  # less than 100% — the broken version returned 1781x
    assert result == pytest.approx(0.796, abs=0.02)

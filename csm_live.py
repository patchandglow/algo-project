"""CSM v2 — parameterized rebalancing frequency for retail deployment.

Sprint 3 used monthly rebalancing. For real deployment in a TFSA, costs eat
returns. Quarterly cuts costs by 3x with minor signal degradation.

This module:
  1. Validates CSM works on Canadian-listed universe
  2. Compares monthly vs quarterly rebalancing
  3. Picks the deployable parameter set
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data_loader import load_daily
from src.metrics import summary, format_summary
from src.ca_universe import CA_UNIVERSE, CA_DESCRIPTIONS


def load_universe(symbols, start, end):
    """Load close prices for a list of symbols. Drop any symbol missing >5% of data."""
    print(f"Loading {len(symbols)} ETFs...")
    closes = {}
    for sym in symbols:
        try:
            df = load_daily(sym, start=start, end=end)
            closes[sym] = df["Close"]
        except Exception as e:
            print(f"  {sym}: skipped ({str(e)[:60]})")
    df = pd.DataFrame(closes)
    # Trim to longest common period
    df = df.dropna(how="all").dropna(thresh=int(len(df) * 0.95), axis=1)
    df = df.dropna()
    return df


def cross_sectional_momentum(prices, top_k=3, lookback_months=12, skip_months=1,
                              rebal_freq="ME", initial_capital=10000,
                              slippage=0.0005):
    """Run CSM with configurable rebalance frequency.

    rebal_freq:
        'ME' = month-end (monthly)
        'QE' = quarter-end (quarterly)
        '2ME' = bi-monthly
    """
    # Resample to rebalance dates
    rebal_prices = prices.resample(rebal_freq).last()

    # Trailing N-1 momentum (in months, regardless of rebal freq)
    # We want momentum measured in months, so resample to monthly first for signal
    monthly = prices.resample("ME").last()
    mom_lookback = lookback_months - skip_months
    mom_monthly = monthly.pct_change(periods=mom_lookback).shift(skip_months)

    # For each rebalance date, find the most recent monthly momentum signal
    capital = initial_capital
    equity_records = []
    holdings_log = []

    rebal_dates = rebal_prices.index
    if len(rebal_dates) < 4:
        return pd.Series(dtype=float), pd.DataFrame(), []

    # Skip warmup period
    warmup_end = pd.Timestamp(rebal_dates[0]) + pd.DateOffset(months=lookback_months)
    valid_rebal = [d for d in rebal_dates if d >= warmup_end]

    if len(valid_rebal) < 2:
        return pd.Series(dtype=float), pd.DataFrame(), []

    equity_records.append({"date": valid_rebal[0], "equity": capital})

    for i in range(len(valid_rebal) - 1):
        rebal_date = valid_rebal[i]
        next_date = valid_rebal[i + 1]

        # Find most recent monthly momentum row at-or-before rebal_date
        mom_row = mom_monthly.loc[mom_monthly.index <= rebal_date]
        if len(mom_row) == 0:
            continue
        scores = mom_row.iloc[-1].dropna()
        if len(scores) < top_k:
            continue
        winners = scores.nlargest(top_k).index.tolist()

        # Compute period return (equal-weight)
        period_prices = prices.loc[rebal_date:next_date, winners]
        if len(period_prices) < 2:
            continue
        period_returns = (period_prices.iloc[-1] / period_prices.iloc[0]) - 1
        # Cost: 2x rotation per rebalance (sell old, buy new)
        net_return = period_returns.mean() - 2 * slippage
        capital = capital * (1 + net_return)

        holdings_log.append({
            "rebal_date": rebal_date, "next_rebal": next_date,
            "winners": winners, "ret": net_return, "capital_after": capital,
        })
        equity_records.append({"date": next_date, "equity": capital})

    equity = pd.DataFrame(equity_records).set_index("date")["equity"]
    trades = pd.DataFrame([
        {"date": h["next_rebal"], "pnl": h["ret"] * 10000, "winners": h["winners"]}
        for h in holdings_log
    ])
    return equity, trades, holdings_log


def main():
    print("=" * 70)
    print("CSM ON CANADIAN UNIVERSE — Sprint 4 deployment validation")
    print("=" * 70)

    prices = load_universe(CA_UNIVERSE, start="2014-01-01", end="2025-12-31")
    print(f"Universe loaded: {prices.shape[1]} ETFs, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Symbols: {list(prices.columns)}")

    print(f"\n{'=' * 70}")
    print("PARAMETER SWEEP — top_k × lookback × rebal_freq")
    print(f"{'=' * 70}")
    print(f"\n{'top_k':>6}{'lookback':>10}{'freq':>8}{'CAGR':>9}{'Sharpe':>8}"
          f"{'MaxDD':>9}{'#trades':>9}")
    print("-" * 60)

    results = []
    for k in [2, 3, 4]:
        for lb in [6, 9, 12]:
            for freq in ["ME", "QE"]:
                eq, tr, _ = cross_sectional_momentum(
                    prices, top_k=k, lookback_months=lb, rebal_freq=freq
                )
                if len(eq) < 2:
                    continue
                s = summary(eq, tr)
                results.append({"k": k, "lookback": lb, "freq": freq, **s})
                print(f"{k:>6}{lb:>10}{freq:>8}{s['cagr']:>9.2%}{s['sharpe']:>8.2f}"
                      f"{s['max_drawdown']:>9.2%}{s['n_trades']:>9}")

    df = pd.DataFrame(results)

    # Compare benchmarks
    print(f"\n{'=' * 70}")
    print("BENCHMARKS")
    print(f"{'=' * 70}")
    benchmarks = ["XIC.TO", "VFV.TO", "XSP.TO"]
    for b in benchmarks:
        try:
            bd = load_daily(b, start=str(prices.index[0].date()),
                              end=str(prices.index[-1].date()))
            bd_eq = bd["Close"] / bd["Close"].iloc[0] * 10000
            bd_rets = bd["Close"].pct_change().dropna()
            n_yrs = len(bd_eq) / 252
            cagr_b = (bd_eq.iloc[-1] / bd_eq.iloc[0]) ** (1/n_yrs) - 1
            sharpe_b = bd_rets.mean() / bd_rets.std() * np.sqrt(252)
            mdd_b = ((bd_eq - bd_eq.cummax()) / bd_eq.cummax()).min()
            print(f"  {b:8s} ({CA_DESCRIPTIONS.get(b, b)}): "
                  f"CAGR {cagr_b:.2%}, Sharpe {sharpe_b:.2f}, MaxDD {mdd_b:.2%}")
        except Exception as e:
            print(f"  {b}: failed ({e})")

    # Recommendation
    print(f"\n{'=' * 70}")
    print("DEPLOYMENT RECOMMENDATION")
    print(f"{'=' * 70}")
    quart = df[df["freq"] == "QE"].copy()
    quart_pos = quart[quart["sharpe"] > 0]
    print(f"\nQuarterly variants positive: {len(quart_pos)}/{len(quart)}")
    print(f"Quarterly Sharpe range: {quart['sharpe'].min():.2f} to {quart['sharpe'].max():.2f}")
    if len(quart) > 0:
        print(f"Quarterly Sharpe std: {quart['sharpe'].std():.2f}")

    # IS/OOS
    print(f"\n{'=' * 70}")
    print("IS/OOS VALIDATION (Quarterly, train 2014-2020 / test 2021-2025)")
    print(f"{'=' * 70}")
    train = prices.loc["2014-01-01":"2020-12-31"]
    test = prices.loc["2019-12-01":"2025-12-31"]   # 1y warmup overlap

    best_is = -np.inf
    best_p = None
    for k in [2, 3, 4]:
        for lb in [6, 9, 12]:
            eq, tr, _ = cross_sectional_momentum(train, top_k=k, lookback_months=lb,
                                                    rebal_freq="QE")
            if len(eq) < 4:
                continue
            s = summary(eq, tr)
            if s["sharpe"] > best_is:
                best_is = s["sharpe"]
                best_p = (k, lb)
    if best_p:
        print(f"\nBest IS (quarterly): top-{best_p[0]}, {best_p[1]}m lookback, "
              f"Sharpe {best_is:.2f}")
        eq_oos, tr_oos, _ = cross_sectional_momentum(
            test, top_k=best_p[0], lookback_months=best_p[1], rebal_freq="QE"
        )
        eq_oos = eq_oos.loc["2021-01-01":]
        if len(eq_oos) > 1:
            eq_oos = eq_oos / eq_oos.iloc[0] * 10000
            tr_oos_filtered = tr_oos[tr_oos["date"] >= "2021-01-01"] if len(tr_oos) else tr_oos
            s_oos = summary(eq_oos, tr_oos_filtered)
            print(f"\nOOS RESULTS:")
            print(format_summary(s_oos))


if __name__ == "__main__":
    main()

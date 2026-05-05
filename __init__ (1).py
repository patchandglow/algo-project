"""H4 — Cross-Sectional Momentum on Sector ETFs.

Hypothesis: Across a basket of sector/asset ETFs, the strongest performers over
the prior N months continue to outperform over the next month. Documented since
Jegadeesh-Titman 1993.

Universe: 11 SPDR sector ETFs + bond ETF + gold ETF + emerging markets ETF.
This is INSURANCE: works at retail scale, doesn't need intraday data, doesn't
fit prop firms (longer horizon) — but proves the framework handles multi-asset
strategies and gives us a working alternative deployment if intraday hypotheses
all die (which is now confirmed for the ones we've tested).

Method:
  - Monthly rebalancing (last trading day of month)
  - For each rebalance: rank universe by trailing 12-1 month return
    (12-month return excluding most recent month, classic JT specification)
  - Hold equal-weight top K performers for next month
  - Compare to equal-weight benchmark and SPY
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data_loader import load_daily
from src.metrics import summary, format_summary
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Diversified universe
UNIVERSE = [
    "XLK",   # Tech
    "XLF",   # Financials
    "XLV",   # Healthcare
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLY",   # Cons. Discretionary
    "XLP",   # Cons. Staples
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
    "TLT",   # Long-term Treasuries
    "GLD",   # Gold
    "EEM",   # Emerging Markets
]


def load_universe(start, end):
    print(f"Loading {len(UNIVERSE)} ETFs...")
    closes = {}
    for sym in UNIVERSE:
        try:
            df = load_daily(sym, start=start, end=end)
            closes[sym] = df["Close"]
        except Exception as e:
            print(f"  {sym}: skipped ({e})")
    return pd.DataFrame(closes).dropna(how="all")


def cross_sectional_momentum(prices, top_k=3, lookback_months=12, skip_months=1,
                              initial_capital=10000, slippage=0.0005):
    """Monthly rebalanced cross-sectional momentum."""
    # Resample to month-end prices
    monthly = prices.resample("ME").last()

    # Trailing 12-1 momentum: return from t-12 to t-1 (skip most recent month)
    mom = monthly.pct_change(periods=lookback_months - skip_months).shift(skip_months)

    capital = initial_capital
    equity_records = [{"date": monthly.index[lookback_months], "equity": capital}]
    holdings_log = []
    monthly_returns = []

    for i in range(lookback_months, len(monthly) - 1):
        rebal_date = monthly.index[i]
        next_date = monthly.index[i + 1]

        # Rank universe by momentum at rebal_date
        scores = mom.iloc[i].dropna()
        if len(scores) < top_k:
            continue
        winners = scores.nlargest(top_k).index.tolist()

        # Compute next month's return (equal weight on winners)
        # Use daily prices for proper returns
        period_prices = prices.loc[rebal_date:next_date, winners]
        if len(period_prices) < 2:
            continue
        period_returns = (period_prices.iloc[-1] / period_prices.iloc[0]) - 1
        # Cost: 2x rotation per rebalance (sell old, buy new) at slippage
        net_return = period_returns.mean() - 2 * slippage
        capital = capital * (1 + net_return)

        monthly_returns.append({
            "date": next_date, "winners": winners,
            "ret": net_return, "capital": capital,
        })
        equity_records.append({"date": next_date, "equity": capital})

    equity = pd.DataFrame(equity_records).set_index("date")["equity"]
    trades = pd.DataFrame([
        {"date": r["date"], "pnl": r["ret"] * r["capital"]}
        for r in monthly_returns
    ])
    return equity, trades, monthly_returns


def main():
    print("=" * 70)
    print("H4: CROSS-SECTIONAL MOMENTUM ON SECTOR/ASSET ETFs")
    print("=" * 70)

    prices = load_universe(start="2015-01-01", end="2024-12-31")
    print(f"Loaded universe: {prices.shape[1]} ETFs, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")

    # Variant sweep
    print(f"\n{'Variant':<35}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'Calmar':>8}")
    print("-" * 75)
    results = []
    for k in [2, 3, 5]:
        for lookback in [3, 6, 12]:
            eq, tr, _ = cross_sectional_momentum(prices, top_k=k,
                                                   lookback_months=lookback)
            s = summary(eq, tr)
            label = f"top-{k}, {lookback}m lookback"
            print(f"{label:<35}{s['cagr']:>8.2%}{s['sharpe']:>8.2f}"
                  f"{s['max_drawdown']:>8.2%}{s['calmar']:>8.2f}")
            results.append({"k": k, "lookback": lookback, **s})

    # Compare to SPY benchmark
    spy = load_daily("SPY", start="2015-01-01", end="2024-12-31")
    spy_eq = spy["Close"] / spy["Close"].iloc[0] * 10000
    spy_returns = spy["Close"].pct_change().dropna()
    spy_stats = {
        "cagr": (spy_eq.iloc[-1] / spy_eq.iloc[0]) ** (252 / len(spy_eq)) - 1,
        "sharpe": (spy_returns.mean() / spy_returns.std()) * np.sqrt(252),
        "max_drawdown": ((spy_eq - spy_eq.cummax()) / spy_eq.cummax()).min(),
    }
    print(f"{'SPY Buy & Hold (benchmark)':<35}{spy_stats['cagr']:>8.2%}"
          f"{spy_stats['sharpe']:>8.2f}{spy_stats['max_drawdown']:>8.2%}{'':>8}")

    # Equal-weight benchmark
    ew = (prices / prices.iloc[0]).mean(axis=1) * 10000
    ew_rets = (prices.pct_change().mean(axis=1)).dropna()
    ew_stats = {
        "cagr": (ew.iloc[-1] / ew.iloc[0]) ** (252 / len(ew)) - 1,
        "sharpe": (ew_rets.mean() / ew_rets.std()) * np.sqrt(252),
        "max_drawdown": ((ew - ew.cummax()) / ew.cummax()).min(),
    }
    print(f"{'EW universe (benchmark)':<35}{ew_stats['cagr']:>8.2%}"
          f"{ew_stats['sharpe']:>8.2f}{ew_stats['max_drawdown']:>8.2%}{'':>8}")

    # IS/OOS validation on best
    best = max(results, key=lambda r: r["sharpe"])
    print(f"\nBest in-grid: top-{best['k']}, {best['lookback']}m lookback, "
          f"Sharpe {best['sharpe']:.2f}")

    print("\n" + "=" * 70)
    print("IS/OOS SPLIT: train 2015-2019, test 2020-2024")
    print("=" * 70)

    train_prices = prices.loc["2015-01-01":"2019-12-31"]
    test_prices = prices.loc["2019-01-01":"2024-12-31"]  # need lookback overlap

    # Re-search best on train only
    best_sharpe = -np.inf
    best_params = None
    for k in [2, 3, 5]:
        for lb in [3, 6, 12]:
            eq, tr, _ = cross_sectional_momentum(train_prices, top_k=k,
                                                   lookback_months=lb)
            s = summary(eq, tr)
            if s["sharpe"] > best_sharpe:
                best_sharpe = s["sharpe"]
                best_params = (k, lb)
    print(f"In-sample best: top-{best_params[0]}, {best_params[1]}m, "
          f"Sharpe {best_sharpe:.2f}")

    # Apply to OOS
    eq_oos, tr_oos, _ = cross_sectional_momentum(test_prices,
                                                    top_k=best_params[0],
                                                    lookback_months=best_params[1])
    # Trim equity to OOS period only (after lookback warmup)
    eq_oos = eq_oos.loc["2020-01-01":]
    if len(eq_oos) > 1:
        eq_oos = eq_oos / eq_oos.iloc[0] * 10000
        s_oos = summary(eq_oos, tr_oos[tr_oos["date"] >= "2020-01-01"])
        print(f"\nOOS RESULTS:")
        print(format_summary(s_oos))

    # Plot best variant equity vs SPY
    eq_full, _, _ = cross_sectional_momentum(prices,
                                                top_k=best_params[0],
                                                lookback_months=best_params[1])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(eq_full.index, eq_full.values, label=f"CSM top-{best_params[0]}, {best_params[1]}m", linewidth=1.5)
    spy_aligned = spy_eq.loc[eq_full.index[0]:].copy()
    spy_aligned = spy_aligned / spy_aligned.iloc[0] * eq_full.iloc[0]
    ax.plot(spy_aligned.index, spy_aligned.values, label="SPY B&H", alpha=0.7)
    ax.set_title("H4: Cross-Sectional Momentum vs SPY")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = Path(__file__).parent.parent / "reports" / "fig7_csm.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nSaved: {out.name}")


if __name__ == "__main__":
    main()

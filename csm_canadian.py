"""Diagnostic: understand overnight vs intraday return structure of SPY.

Before we can design an overnight gap strategy, we need to know:
- Is the average overnight return positive or negative?
- Do gaps continue or reverse?
- Does this differ by gap magnitude?
- Does this differ by VIX regime?

This is research, not a strategy. Output goes to a report.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data_loader import load_daily


def analyze(symbol="SPY", start="2010-01-01", end="2024-12-31"):
    print(f"\n{'=' * 70}")
    print(f"OVERNIGHT vs INTRADAY RETURN STRUCTURE: {symbol} {start} to {end}")
    print(f"{'=' * 70}")

    data = load_daily(symbol, start=start, end=end)
    df = data.copy()

    df["prev_close"] = df["Close"].shift(1)
    df["overnight_ret"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
    df["intraday_ret"] = (df["Close"] - df["Open"]) / df["Open"]
    df["full_day_ret"] = (df["Close"] - df["prev_close"]) / df["prev_close"]
    df = df.dropna()

    n = len(df)
    print(f"\nN bars: {n}")
    print(f"\nMEAN DAILY RETURNS:")
    print(f"  Overnight:    {df['overnight_ret'].mean():>10.6f}  "
          f"(annualized: {df['overnight_ret'].mean() * 252:>7.2%})")
    print(f"  Intraday:     {df['intraday_ret'].mean():>10.6f}  "
          f"(annualized: {df['intraday_ret'].mean() * 252:>7.2%})")
    print(f"  Full day:     {df['full_day_ret'].mean():>10.6f}  "
          f"(annualized: {df['full_day_ret'].mean() * 252:>7.2%})")

    # If overnight > intraday on average, that's the documented pattern (Lou-Polk)
    if df["overnight_ret"].mean() > df["intraday_ret"].mean():
        print("\n>>> Overnight return EXCEEDS intraday on average")
        print("    Documented pattern (Lou-Polk-Skouras 2019)")

    # Q1: Does intraday return REVERSE or CONTINUE the overnight gap?
    print(f"\n{'=' * 70}")
    print("Q1: Do intraday returns continue (momentum) or reverse (mean-rev)?")
    print(f"{'=' * 70}")

    # Bin overnight returns by sigma bucket
    on_vol = df["overnight_ret"].rolling(60).std()
    df["gap_z"] = df["overnight_ret"] / on_vol
    df = df.dropna()

    bins = [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf]
    labels = ["< -2σ", "[-2,-1)", "[-1,-0.5)", "[-0.5,0)",
              "[0,0.5)", "[0.5,1)", "[1,2)", "≥ 2σ"]
    df["gap_bucket"] = pd.cut(df["gap_z"], bins=bins, labels=labels)

    print(f"\n{'Gap bucket':>15}{'N':>8}{'Mean intraday':>18}{'Median':>12}"
          f"{'Win%':>10}{'Stdev':>10}")
    print("-" * 73)
    for bucket in labels:
        sub = df[df["gap_bucket"] == bucket]
        if len(sub) > 0:
            mean = sub["intraday_ret"].mean()
            med = sub["intraday_ret"].median()
            wins = (sub["intraday_ret"] > 0).mean()
            std = sub["intraday_ret"].std()
            print(f"{bucket:>15}{len(sub):>8d}{mean:>18.5f}{med:>12.5f}"
                  f"{wins:>10.2%}{std:>10.4f}")

    # Q2: Does behavior differ by VIX regime?
    print(f"\n{'=' * 70}")
    print("Q2: VIX regime breakdown")
    print(f"{'=' * 70}")

    vix = load_daily("^VIX", start=start, end=end)
    vix_df = vix[["Close"]].rename(columns={"Close": "vix"})
    df_vix = df.join(vix_df, how="inner")
    df_vix["vix_regime"] = pd.qcut(df_vix["vix"], q=3, labels=["Low VIX", "Mid VIX", "High VIX"])

    for regime in ["Low VIX", "Mid VIX", "High VIX"]:
        sub = df_vix[df_vix["vix_regime"] == regime]
        print(f"\n  {regime} (N={len(sub)}, VIX range: "
              f"{sub['vix'].min():.1f}-{sub['vix'].max():.1f}):")
        # Test: do extreme gaps revert in this regime?
        extreme_up = sub[sub["gap_z"] >= 1.5]
        extreme_dn = sub[sub["gap_z"] <= -1.5]
        if len(extreme_up) > 5:
            print(f"    Gap up ≥ 1.5σ: N={len(extreme_up)}, "
                  f"intraday mean: {extreme_up['intraday_ret'].mean():.5f} "
                  f"(t-stat: {_tstat(extreme_up['intraday_ret']):.2f})")
        if len(extreme_dn) > 5:
            print(f"    Gap dn ≤ -1.5σ: N={len(extreme_dn)}, "
                  f"intraday mean: {extreme_dn['intraday_ret'].mean():.5f} "
                  f"(t-stat: {_tstat(extreme_dn['intraday_ret']):.2f})")

    # Q3: Year-by-year stability of any effect
    print(f"\n{'=' * 70}")
    print("Q3: Year-by-year — is any effect stable across time?")
    print(f"{'=' * 70}")
    print(f"\nFocus: gap up ≥ 1.5σ → intraday return")
    df["year"] = df.index.year
    print(f"\n{'Year':>6}{'N':>6}{'Mean intraday':>18}{'t-stat':>10}")
    for year, group in df[df["gap_z"] >= 1.5].groupby("year"):
        if len(group) >= 3:
            t = _tstat(group["intraday_ret"])
            print(f"{year:>6}{len(group):>6}{group['intraday_ret'].mean():>18.5f}{t:>10.2f}")


def _tstat(series):
    if len(series) < 2:
        return 0.0
    return float(series.mean() / (series.std() / np.sqrt(len(series))))


if __name__ == "__main__":
    analyze()

"""H3 — Time-of-day return diagnostic on SPY hourly bars.

Question: Are there systematic intraday return patterns we can exploit?

Method: For each hourly bar, compute its return (Close-Open)/Open.
Group by time-of-day. Look at:
  - Mean return per hour
  - Vol per hour
  - Hit rate (% positive)
  - t-statistic to assess significance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.intraday_loader import load_intraday, add_session_columns


def analyze():
    print("=" * 70)
    print("H3: TIME-OF-DAY RETURN DIAGNOSTIC — SPY 1h bars, 2 years")
    print("=" * 70)

    df = load_intraday("SPY", interval="1h", period="2y")
    df = add_session_columns(df)
    df["bar_return"] = (df["Close"] - df["Open"]) / df["Open"]
    df = df.dropna(subset=["bar_return"])

    print(f"\nTotal bars: {len(df)}, days: {df['date'].nunique()}")

    # Group by hour_min
    print(f"\n{'Bar (ET)':>10} {'N':>6} {'Mean (bps)':>12} {'Std (bps)':>10} "
          f"{'Win %':>8} {'t-stat':>8}")
    print("-" * 60)

    hour_stats = []
    for hour_min, group in df.groupby("hour_min"):
        if hour_min == -1:
            continue
        if len(group) < 20:
            continue
        rets = group["bar_return"]
        n = len(rets)
        mean_bps = rets.mean() * 10000
        std_bps = rets.std() * 10000
        win = (rets > 0).mean() * 100
        t = rets.mean() / (rets.std() / np.sqrt(n)) if rets.std() > 0 else 0
        hr_str = f"{hour_min // 100:02d}:{hour_min % 100:02d}"
        print(f"{hr_str:>10} {n:>6} {mean_bps:>12.3f} {std_bps:>10.3f} "
              f"{win:>8.2f} {t:>8.2f}")
        hour_stats.append({
            "hour_min": hour_min, "hr_str": hr_str, "n": n,
            "mean_bps": mean_bps, "std_bps": std_bps,
            "win_pct": win, "t_stat": t,
        })

    stats_df = pd.DataFrame(hour_stats)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["red" if r < 0 else "green" for r in stats_df["mean_bps"]]
    ax1.bar(stats_df["hr_str"], stats_df["mean_bps"], color=colors, alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Mean return per bar (bps)")
    ax1.set_xlabel("Bar start time (ET)")
    ax1.set_title("Figure 6a: Mean Hourly Return by Bar (SPY 1h, 2yr)")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(stats_df["hr_str"], stats_df["t_stat"], color="steelblue", alpha=0.7)
    ax2.axhline(2, color="red", linestyle="--", label="t=±2")
    ax2.axhline(-2, color="red", linestyle="--")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("t-statistic")
    ax2.set_xlabel("Bar start time (ET)")
    ax2.set_title("Figure 6b: t-statistic by Bar")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = Path(__file__).parent.parent / "reports" / "fig6_intraday_tod.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nSaved: {out.name}")

    # Highlight any significant findings
    print("\n" + "=" * 70)
    print("SIGNIFICANT BARS (|t| > 1.5)")
    print("=" * 70)
    sig = stats_df[stats_df["t_stat"].abs() > 1.5]
    if len(sig) == 0:
        print("None. No bar has |t| > 1.5.")
        print("Conclusion: no robust time-of-day return signal in SPY 1h data over 2y.")
    else:
        print(sig.to_string(index=False))

    return stats_df


if __name__ == "__main__":
    analyze()

"""Plotting utilities."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_equity_curve(equity, save_path=None, title="Equity Curve", benchmark=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(equity.index, equity.values, label="Strategy", linewidth=1.5)
    if benchmark is not None:
        bench_norm = benchmark / benchmark.iloc[0] * equity.iloc[0]
        ax1.plot(bench_norm.index, bench_norm.values,
                 label="Buy & Hold", linewidth=1.0, alpha=0.7)
        ax1.legend()

    ax1.set_title(title)
    ax1.set_ylabel("Equity ($)")
    ax1.grid(True, alpha=0.3)

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.close()

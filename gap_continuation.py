"""Forward paper-test runner.

Sprint 4 deliverable. Two functions:
  1. Replay historical quarterly rebalances using the deployed parameters
     to confirm the deployed system reproduces the backtest performance.
  2. Generate the schedule of upcoming rebalance dates so the user knows
     when to run --rebalance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.data_loader import load_daily
from src.metrics import summary, format_summary
from src.ca_universe import CA_UNIVERSE, CA_DESCRIPTIONS
from strategies.csm_canadian import cross_sectional_momentum, load_universe


def replay_history():
    """Replay 2014-2025 quarterly rebalances using DEPLOYED params.

    Confirms the live system's parameters reproduce the validated backtest.
    """
    print("=" * 72)
    print("PHASE 1: Historical replay using deployed parameters")
    print("=" * 72)
    print("Params: top-3, 9m lookback, quarterly, CA universe")
    print()

    prices = load_universe(CA_UNIVERSE, start="2014-01-01", end="2025-12-31")
    print(f"Universe: {prices.shape[1]} ETFs, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")

    eq, tr, holdings = cross_sectional_momentum(
        prices, top_k=3, lookback_months=9, skip_months=1, rebal_freq="QE"
    )

    if len(eq) < 4:
        print("Insufficient data")
        return

    s = summary(eq, tr)
    print(f"\nBACKTEST RESULTS (full period):")
    print(format_summary(s))

    print(f"\nALL HISTORICAL REBALANCES:")
    print(f"{'Rebal date':<12}{'Held next quarter':<35}{'Period ret':>12}{'Cum equity':>14}")
    print("-" * 75)
    for h in holdings:
        winners = ", ".join(h["winners"])
        print(f"{h['rebal_date'].strftime('%Y-%m-%d'):<12}{winners:<35}"
              f"{h['ret']:>12.2%}{h['capital_after']:>14,.0f}")

    return eq, tr, holdings


def project_upcoming_rebalances(n_quarters=4):
    """Show the next N rebalance dates so user can mark calendar."""
    print("\n" + "=" * 72)
    print(f"PHASE 2: Upcoming rebalance schedule (next {n_quarters} quarters)")
    print("=" * 72)
    today = pd.Timestamp.now().normalize()

    # Quarterly rebalances are last business day of Mar/Jun/Sep/Dec
    quarter_ends = pd.date_range(
        start=today,
        end=today + pd.DateOffset(months=n_quarters * 3 + 3),
        freq="QE",   # quarter end (calendar)
    )
    # Adjust to last business day
    bdays = pd.bdate_range(start=today,
                            end=today + pd.DateOffset(months=n_quarters * 3 + 3))
    rebal_dates = []
    for q in quarter_ends:
        # Find last business day of that quarter
        q_bdays = bdays[(bdays.year == q.year) & (bdays.month == q.month)]
        if len(q_bdays) > 0:
            rebal_dates.append(q_bdays[-1])

    upcoming = [d for d in rebal_dates if d >= today][:n_quarters]
    print(f"\n{'Rebalance date':<18}{'Days from now':>15}")
    print("-" * 35)
    for d in upcoming:
        days = (d - today).days
        print(f"{d.strftime('%Y-%m-%d (%a)'):<18}{days:>15d}")
    print()
    print("On each date:")
    print("  1. Run: python -m strategies.csm_live --capital <YOUR_CAPITAL> --rebalance")
    print("  2. Review the trade ticket")
    print("  3. Execute trades manually at your broker")
    print("  4. After confirmation: python -m strategies.csm_live --record")
    return upcoming


def paper_trade_log():
    """Initialize / report on a paper-trading log.

    For Sprint 4, you simulate following the strategy on paper for 3-6 months
    before deploying real capital. This logs hypothetical fills and tracks
    whether live performance matches backtest expectations.
    """
    print("\n" + "=" * 72)
    print("PHASE 3: Paper-trading log status")
    print("=" * 72)

    log_path = Path(__file__).parent.parent / "data" / "positions" / "paper_log.csv"

    if log_path.exists():
        log = pd.read_csv(log_path, parse_dates=["date"])
        print(f"\nExisting paper log: {len(log)} rebalances")
        print(log.to_string(index=False))
    else:
        print("\nNo paper log yet. To start:")
        print("  - Run --check on a quarterly date")
        print("  - Manually record what the trade ticket says (in a spreadsheet or here)")
        print("  - At each subsequent rebalance, log the actual returns")
        print("  - Compare cumulative paper returns to backtest expectation after 3-6 months")
        print()
        print("Validation gate (before deploying real capital):")
        print(f"  - Backtest expectation: ~{0.137:.1%} CAGR (2014-2025), Sharpe ~1.0")
        print(f"  - Paper-trade gate: 6 months of paper performance within ±50% of backtest expectation")
        print(f"  - Specifically: 6m paper return between -3% and +12% would be acceptable")


def main():
    eq, tr, holdings = replay_history()
    project_upcoming_rebalances(n_quarters=4)
    paper_trade_log()


if __name__ == "__main__":
    main()

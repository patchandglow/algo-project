"""Live deployment module for CSM CA strategy.

USAGE (after Sprint 4 setup):
    python -m strategies.csm_live --capital 5000 --check
        Shows current holdings recommendation. No action.

    python -m strategies.csm_live --capital 5000 --rebalance
        Generates a trade ticket if a rebalance is due.

DEPLOYMENT PARAMETERS (FROZEN — do not optimize after this point):
    Universe:    13 Canadian-listed ETFs (see src/ca_universe.py)
    top_k:       3       (concentrated but diversified across asset classes)
    lookback:    9       (9-month trailing momentum, skip last 1)
    skip:        1
    freq:        QE      (quarterly rebalance — last trading day of mar/jun/sep/dec)

WHY THESE PARAMS:
    - top-3 plateau-stable in CA universe (Sharpe 0.92-1.21 across all freqs/lookbacks)
    - 9m lookback: middle of robust range (6/9/12 all work)
    - Quarterly: 3x lower transaction cost vs monthly with marginal signal cost
    - Locked AFTER OOS confirmation; ANY parameter change requires new OOS test

REBALANCE WORKFLOW (manual execution):
    1. Run --check on the morning of the rebalance date
    2. Review the trade ticket
    3. Execute trades manually at your broker (Wealthsimple Trade / Questrade)
    4. Confirm fills, then run --record to log the actual fills
    5. Position state persists in data/positions/state.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime, date
import pandas as pd
import numpy as np
from src.data_loader import load_daily
from src.ca_universe import CA_UNIVERSE, CA_DESCRIPTIONS

# FROZEN PARAMETERS — changes require new OOS validation
PARAMS = {
    "top_k": 3,
    "lookback_months": 9,
    "skip_months": 1,
    "rebal_freq": "QE",   # quarter-end
    "universe": CA_UNIVERSE,
    "min_trade_amount_cad": 100,    # don't trade if delta < $100 (broker minimums)
    "cash_buffer_pct": 0.02,         # leave 2% cash for slippage/fees
}

STATE_DIR = Path(__file__).parent.parent / "data" / "positions"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"


def get_current_signal(as_of: pd.Timestamp = None):
    """Compute the current top-K winners for the most recent signal date."""
    if as_of is None:
        as_of = pd.Timestamp.now().normalize()

    # Use full historical cache (don't use refresh=True which truncates).
    # Load max history; the strategy only uses the most recent N months.
    start = "2014-01-01"
    end = as_of.strftime("%Y-%m-%d")

    closes = {}
    for sym in PARAMS["universe"]:
        try:
            df = load_daily(sym, start=start, end=end)
            closes[sym] = df["Close"]
        except Exception as e:
            print(f"WARNING: failed to load {sym}: {e}")

    prices = pd.DataFrame(closes).dropna(thresh=int(0.95 * len(closes)), axis=1)
    prices = prices.dropna()
    if len(prices) < 200:
        raise ValueError(f"Insufficient data: only {len(prices)} bars")

    # Trailing momentum: 9m return, skip last 1m
    monthly = prices.resample("ME").last()
    lookback = PARAMS["lookback_months"]
    skip = PARAMS["skip_months"]
    mom = monthly.pct_change(periods=lookback - skip).shift(skip)

    if len(mom.dropna()) == 0:
        raise ValueError("No valid momentum signals computed")

    last_mom = mom.dropna().iloc[-1]
    last_signal_date = mom.dropna().index[-1]

    winners = last_mom.nlargest(PARAMS["top_k"])

    # Get current prices (most recent close)
    current_prices = prices.iloc[-1]

    return {
        "signal_date": last_signal_date,
        "as_of_price_date": prices.index[-1],
        "winners": winners,
        "all_scores": last_mom.sort_values(ascending=False),
        "current_prices": current_prices,
    }


def is_rebalance_due(last_rebal_date: pd.Timestamp = None,
                      as_of: pd.Timestamp = None,
                      freq: str = "QE") -> bool:
    """Determine if quarterly rebalance is due.

    Quarterly rebalance dates: last trading day of Mar/Jun/Sep/Dec.
    Approximation: any check after the last business day of those months
    where we haven't rebalanced this quarter.
    """
    if as_of is None:
        as_of = pd.Timestamp.now().normalize()
    if last_rebal_date is None:
        return True
    last_rebal = pd.Timestamp(last_rebal_date)

    # Quarterly: trigger if we've crossed into a new quarter since last rebal
    last_q = (last_rebal.year, (last_rebal.month - 1) // 3)
    cur_q = (as_of.year, (as_of.month - 1) // 3)
    return cur_q > last_q


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "current_holdings": {},     # sym -> shares
        "cash_cad": 0,
        "last_rebalance_date": None,
        "rebalance_history": [],
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def generate_trade_ticket(target_capital_cad: float, current_holdings: dict,
                            cash_cad: float, signal: dict):
    """Build a trade ticket: list of (action, symbol, shares, est_dollar) to execute."""
    winners = signal["winners"]
    prices = signal["current_prices"]

    target_per_position = target_capital_cad * (1 - PARAMS["cash_buffer_pct"]) / PARAMS["top_k"]

    target_holdings = {}
    for sym in winners.index:
        if sym not in prices.index or pd.isna(prices[sym]):
            print(f"WARNING: no price for {sym}, skipping")
            continue
        target_shares = int(target_per_position / prices[sym])
        target_holdings[sym] = target_shares

    # Compute deltas
    all_syms = set(current_holdings.keys()) | set(target_holdings.keys())
    actions = []
    for sym in sorted(all_syms):
        cur = current_holdings.get(sym, 0)
        tgt = target_holdings.get(sym, 0)
        delta = tgt - cur
        if delta == 0:
            continue
        price = prices.get(sym)
        if pd.isna(price):
            actions.append({"sym": sym, "action": "MANUAL_REVIEW",
                             "shares": delta, "price": None,
                             "est_amount": None,
                             "note": "Price unavailable — verify manually"})
            continue
        est_amount = abs(delta) * price
        if est_amount < PARAMS["min_trade_amount_cad"]:
            continue  # skip dust
        action_type = "BUY" if delta > 0 else "SELL"
        actions.append({
            "sym": sym, "action": action_type, "shares": abs(delta),
            "price": float(price), "est_amount": float(est_amount),
            "current": cur, "target": tgt,
        })
    return actions, target_holdings


def format_trade_ticket(actions, signal, target_capital):
    """Pretty-print the trade ticket for manual execution."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"  CSM REBALANCE TRADE TICKET")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Target capital: ${target_capital:,.0f} CAD")
    lines.append(f"  Signal date:    {signal['signal_date'].strftime('%Y-%m-%d')}")
    lines.append(f"  Prices as of:   {signal['as_of_price_date'].strftime('%Y-%m-%d')}")
    lines.append("=" * 72)

    lines.append(f"\nTOP-{PARAMS['top_k']} WINNERS (by 9m-1m momentum):")
    for sym, score in signal["winners"].items():
        desc = CA_DESCRIPTIONS.get(sym, "")
        lines.append(f"  {sym:8s}  {score:+7.2%}   {desc}")

    lines.append(f"\nFULL UNIVERSE RANKING:")
    for sym, score in signal["all_scores"].items():
        marker = " ★" if sym in signal["winners"].index else "  "
        desc = CA_DESCRIPTIONS.get(sym, "")
        lines.append(f"  {marker} {sym:8s}  {score:+7.2%}   {desc}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("  TRADES TO EXECUTE (manual at broker):")
    lines.append("=" * 72)
    if not actions:
        lines.append("  None — current holdings already match target.")
    else:
        # Sells first (frees cash), then buys
        sells = [a for a in actions if a["action"] == "SELL"]
        buys = [a for a in actions if a["action"] == "BUY"]
        manual = [a for a in actions if a["action"] == "MANUAL_REVIEW"]
        for grp_name, grp in [("SELLS (execute first)", sells), ("BUYS", buys),
                                ("MANUAL REVIEW", manual)]:
            if not grp:
                continue
            lines.append(f"\n  {grp_name}:")
            for a in grp:
                if a.get("price") is not None:
                    lines.append(
                        f"    {a['action']:5s} {a['shares']:>5d} {a['sym']:8s}  "
                        f"@ ~${a['price']:.2f}  ≈ ${a['est_amount']:,.0f}  "
                        f"({a.get('current', 0)}→{a.get('target', 0)})"
                    )
                else:
                    lines.append(f"    {a['action']:5s} {a['sym']:8s}  {a.get('note', '')}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("  EXECUTION NOTES:")
    lines.append("    1. Use LIMIT orders, not market. Set limit at ask+0.05 for buys,")
    lines.append("       bid-0.05 for sells. Cancel and adjust if not filled in 30 min.")
    lines.append("    2. Execute SELLs first to free cash for BUYs.")
    lines.append("    3. After all fills confirmed, run: --record to update state.")
    lines.append("    4. Spreadsheet your fills: date, symbol, action, shares, price.")
    lines.append("=" * 72)
    return "\n".join(lines)


def cmd_check(args):
    """Show what the strategy would recommend right now. No state changes."""
    state = load_state()
    print(f"Current state:")
    print(f"  Last rebalance: {state.get('last_rebalance_date', 'NEVER')}")
    print(f"  Holdings: {state.get('current_holdings', {})}")
    print(f"  Cash: ${state.get('cash_cad', 0):,.2f}")
    print()

    last = state.get("last_rebalance_date")
    last_ts = pd.Timestamp(last) if last else None
    due = is_rebalance_due(last_ts)
    print(f"Rebalance due: {due}\n")

    print("Computing current signal...")
    signal = get_current_signal()

    target_capital = float(args.capital)
    actions, target = generate_trade_ticket(
        target_capital, state.get("current_holdings", {}),
        state.get("cash_cad", 0), signal
    )
    print(format_trade_ticket(actions, signal, target_capital))


def cmd_rebalance(args):
    """Generate ticket and write it to a dated file for the user to execute."""
    state = load_state()
    last = state.get("last_rebalance_date")
    last_ts = pd.Timestamp(last) if last else None
    if not is_rebalance_due(last_ts) and not args.force:
        print(f"Rebalance NOT due. Last rebalance: {last_ts}.")
        print("Use --force to override.")
        return

    signal = get_current_signal()
    target_capital = float(args.capital)
    actions, target = generate_trade_ticket(
        target_capital, state.get("current_holdings", {}),
        state.get("cash_cad", 0), signal
    )
    ticket = format_trade_ticket(actions, signal, target_capital)
    print(ticket)

    # Save to dated file
    out_dir = Path(__file__).parent.parent / "reports" / "tickets"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"ticket_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(fname, "w") as f:
        f.write(ticket)
    print(f"\n[Ticket saved to: {fname}]")


def cmd_record(args):
    """After executing the trades, record the new state."""
    print("Manual fill recording — enter your actual fills.")
    print("(In production, this would parse a broker confirmation export.)")
    print("\nFor Sprint 4 demo: pass --auto to apply target holdings as if filled at signal prices.")
    if args.auto:
        signal = get_current_signal()
        state = load_state()
        target_capital = float(args.capital)
        _, target = generate_trade_ticket(target_capital,
                                            state.get("current_holdings", {}),
                                            state.get("cash_cad", 0), signal)
        # Compute remaining cash
        deployed = sum(target[s] * float(signal["current_prices"][s]) for s in target)
        cash = target_capital - deployed
        state["current_holdings"] = target
        state["cash_cad"] = cash
        state["last_rebalance_date"] = signal["as_of_price_date"].strftime("%Y-%m-%d")
        state["rebalance_history"].append({
            "date": signal["as_of_price_date"].strftime("%Y-%m-%d"),
            "winners": list(target.keys()),
            "holdings": target,
            "cash_after": cash,
            "method": "auto_recorded_at_signal_price",
        })
        save_state(state)
        print("State updated:")
        print(json.dumps(state, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="CSM Live Deployment Tool")
    parser.add_argument("--capital", type=float, default=5000,
                        help="Target capital in CAD")
    parser.add_argument("--check", action="store_true",
                        help="Show current signal & target holdings (no state change)")
    parser.add_argument("--rebalance", action="store_true",
                        help="Generate trade ticket if rebalance due")
    parser.add_argument("--record", action="store_true",
                        help="Record fills after manual execution")
    parser.add_argument("--auto", action="store_true",
                        help="(--record only) auto-fill at signal price")
    parser.add_argument("--force", action="store_true",
                        help="Force rebalance even if not due")
    args = parser.parse_args()

    if args.check:
        cmd_check(args)
    elif args.rebalance:
        cmd_rebalance(args)
    elif args.record:
        cmd_record(args)
    else:
        cmd_check(args)


if __name__ == "__main__":
    main()

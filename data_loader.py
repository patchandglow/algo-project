"""Event-loop backtest engine. Long-only equity in Sprint 1."""

import pandas as pd
from pathlib import Path
import yaml

from src.strategy import Strategy
from src.risk import RiskManager


def _load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_backtest(strategy, data, initial_capital=None):
    config = _load_config()
    bt = config["backtest"]
    initial_capital = initial_capital or bt["initial_capital"]
    commission = bt["commission_per_trade"]
    slippage = bt["slippage_pct"]

    signals = strategy.generate_signals(data)
    if "signal" not in signals.columns:
        raise ValueError("Strategy must produce a 'signal' column")

    signals["position"] = signals["signal"].shift(1).fillna(0)

    cash = initial_capital
    shares_held = 0
    entry_price = 0.0
    entry_date = None

    equity_records = []
    trade_records = []

    for date, row in signals.iterrows():
        position = row["position"]
        open_price = row["Open"]
        close_price = row["Close"]

        if position == 1 and shares_held == 0:
            fill_price = open_price * (1 + slippage)
            shares_held = int(cash * 0.95 / fill_price)
            if shares_held > 0:
                cash -= shares_held * fill_price + commission
                entry_price = fill_price
                entry_date = date

        elif position == 0 and shares_held > 0:
            fill_price = open_price * (1 - slippage)
            cash += shares_held * fill_price - commission
            pnl = (fill_price - entry_price) * shares_held - 2 * commission
            ret_pct = (fill_price - entry_price) / entry_price
            trade_records.append({
                "entry_date": entry_date,
                "exit_date": date,
                "entry_price": entry_price,
                "exit_price": fill_price,
                "shares": shares_held,
                "pnl": pnl,
                "return_pct": ret_pct,
            })
            shares_held = 0
            entry_price = 0.0
            entry_date = None

        equity = cash + shares_held * close_price
        equity_records.append({"date": date, "equity": equity})

    if shares_held > 0:
        last_row = signals.iloc[-1]
        fill_price = last_row["Close"] * (1 - slippage)
        cash += shares_held * fill_price - commission
        pnl = (fill_price - entry_price) * shares_held - 2 * commission
        ret_pct = (fill_price - entry_price) / entry_price
        trade_records.append({
            "entry_date": entry_date,
            "exit_date": signals.index[-1],
            "entry_price": entry_price,
            "exit_price": fill_price,
            "shares": shares_held,
            "pnl": pnl,
            "return_pct": ret_pct,
        })

    equity_df = pd.DataFrame(equity_records).set_index("date")["equity"]
    trades_df = pd.DataFrame(trade_records)
    return equity_df, trades_df

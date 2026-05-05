data:
  cache_dir: data/processed
  raw_dir: data/raw

backtest:
  initial_capital: 10000
  commission_per_trade: 1.00
  slippage_pct: 0.0005

risk:
  max_risk_per_trade: 0.004
  daily_loss_soft_stop: -0.015
  daily_loss_hard_stop: -0.025
  max_drawdown_kill: 0.10

reporting:
  output_dir: reports

# Systematic Trading Research Framework

A from-scratch Python framework for backtesting, validating, and deploying
quantitative trading strategies. Built across 4 sprints with rigorous
hypothesis testing, walk-forward analysis, out-of-sample validation, and
a production deployment system.

**Most important outcome: the framework correctly killed 5 hypotheses
and identified 1 that survives the gauntlet.** That ratio (1 in 6) is
the actual measure of disciplined research — not finding a strategy that
"works in backtest."

---

## Highlights

| Metric | Value |
|---|---|
| Lines of code | ~3,900 |
| Unit tests passing | 39 / 39 |
| Hypotheses tested | 6 |
| Hypotheses killed | 5 |
| Hypotheses validated and deployed | 1 |
| Critical bugs caught | 1 (50,000× CAGR inflation in monthly metrics) |

The surviving strategy: **Cross-Sectional Momentum on a 13-ETF Canadian
universe**, quarterly rebalancing.

| Metric | In-sample (2014-2020) | Out-of-sample (2021-2025) |
|---|---|---|
| Sharpe | 0.95 | 1.25 |
| CAGR | 10.4% | 15.8% |
| Max drawdown | -16.0% | -13.1% |
| Win rate | 73% | 80% |

For comparison: SPY benchmark over the same period had a Sharpe of 0.68
with a 34% drawdown.

---

## What this framework does

```
hypothesis → backtest → walk-forward → out-of-sample → deploy or kill
   ↓             ↓            ↓              ↓
   |       cost-realistic    rolling     locked test data
   |       slippage,         train/test  audited on access
   |       commission        windows
   |
   plateau-tested parameters, not curve-fit
```

Concrete components:

- **`src/data_loader.py`** — Cached, validated yfinance daily/intraday loaders
- **`src/metrics.py`** — Sharpe, CAGR, drawdown, etc. with periodicity-aware annualization
- **`src/risk.py`** — Position sizing, daily loss kills, drawdown-based circuit breakers
- **`src/backtest.py`** — Event-loop backtester with realistic costs
- **`src/walkforward.py`** — Rolling train/test fold analysis
- **`src/montecarlo.py`** — Trade-shuffle and block-bootstrap simulators
- **`src/sweep.py`** — Parameter grid search with plateau scoring
- **`src/lockbox.py`** — Out-of-sample data lockbox with access audit log
- **`src/intraday_loader.py`** — Hourly bar loader with session-aware columns
- **`strategies/csm_live.py`** — Production trade ticket generator

---

## The hypothesis kill record

Each strategy was tested with the same gauntlet: parameter sensitivity,
walk-forward validation, in-sample / out-of-sample split, realistic costs.

| Hypothesis | Verdict | Reason |
|---|---|---|
| H0: SMA(50,200) crossover | reference benchmark | Underperforms B&H in bull markets (correctly) |
| H1: Overnight gap mean reversion | KILLED | All thresholds lose money on SPY |
| H1B: Overnight gap continuation | KILLED | Real signal in raw data, eaten by costs, fails OOS |
| H2: Opening range breakout | KILLED | Zero-cost test confirmed no signal in either direction |
| H3: Time-of-day return patterns | KILLED | No hourly bar has │t│ > 1.5 over 2 years |
| H4: Cross-sectional ETF momentum | **VALIDATED** | All 36 grid points positive Sharpe; OOS confirms |

The kills are documented as carefully as the success. Most retail
research never publishes the kills, which is exactly why retail keeps
deploying overfit strategies.

---

## The bug that mattered

During Sprint 3, the cross-sectional momentum strategy initially reported
**408% CAGR and Sharpe 3.28**. That looked too good — in finance, "too
good" should always trigger a code audit, not celebration.

Audit findings: the `cagr()` function assumed daily data
(252 periods/year). When fed monthly equity series (12 periods/year), it
exponentiated returns by 21× more periods than reality. Result: a
50,000× inflation in reported CAGR for any non-daily strategy.

Fix: `metrics.py` now infers periodicity from the DatetimeIndex
automatically and accepts an explicit `periods_per_year` parameter for
non-standard frequencies. 7 regression tests in `tests/test_periodicity.py`
prevent recurrence.

The lesson generalizes: numbers that look like alpha are usually bugs.
Audit before celebrating.

---

## Quickstart

```bash
git clone https://github.com/<your-username>/algo-project.git
cd algo-project
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run the reference SMA crossover backtest
python strategies/sma_crossover.py

# Run the validated CSM strategy historical replay
python strategies/csm_forward.py

# Generate a live trade ticket for the deployed strategy
python -m strategies.csm_live --capital 5000 --check
```

---

## Architecture decisions

A few opinionated choices worth flagging for code reviewers:

1. **All strategies inherit from a `Strategy` base class** with a single
   `generate_signals()` method. Forces consistent interface across
   research and production.

2. **Out-of-sample data is enforced via a Lockbox class.** Accessing test
   data writes a permanent audit log entry. Re-running OOS with different
   parameters defeats the purpose; the audit makes that visible.

3. **Realistic transaction costs are baked in by default.** Every
   backtest applies commission and slippage. A strategy that needs zero
   costs to look profitable is not a strategy.

4. **Risk management is centralized** in a single `RiskManager` class.
   Position sizing logic lives in one place, not scattered across
   strategies. Same goes for daily loss limits and drawdown kills.

5. **Manual execution for live deployment**, not broker API automation.
   For retail-scale capital, the operational risk of an automated trader
   exceeds the benefit. The system generates a trade ticket; the human
   executes at their broker.

---

## Project documentation

- **[Sprint 1 → 4 reports](.)** — full research log, every decision documented
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** — operational manual for
  the deployed strategy

---

## Background

I'm Léonard Barile, a CEGEP student in Sept-Îles, Quebec, building this
as a self-directed research project. Started from "I want to day trade
through a prop firm." Ended at "I have one verified strategy in my TFSA
and several killed hypotheses I trust the kill on."

The journey is documented sprint-by-sprint:

- **Sprint 1**: Built the framework. Validated SMA crossover end-to-end.
- **Sprint 2**: Added walk-forward, Monte Carlo, lockbox. Killed H1.
- **Sprint 3**: Added intraday support. Killed H2 and H3. Validated H4.
  Caught and fixed the metrics bug.
- **Sprint 4**: Adapted H4 to Canadian-listed universe for TFSA tax
  efficiency. Built live deployment system.

If you're looking at this for hiring purposes: I'm open to junior quant
developer or quant analyst roles in Montreal/Quebec. Email: [your email].

---

## License

MIT. Use it, fork it, learn from it.

If you find a bug or improvement, PRs welcome. If you find I overfit
something I claimed wasn't overfit, definitely tell me.

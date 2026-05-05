# Algo Project

Systematic trading research framework. Building toward a profitable
retail futures algo, evaluating prop firm deployment as one possible path.

**Current status:** Sprint 4 complete. **CSM is deployable.** Canadian-listed
universe, quarterly rebalancing, OOS Sharpe 1.25 / CAGR 15.80% / MaxDD -13%
(2021-2025). Trade ticket generator, deployment guide, and paper-trading
workflow all in place.

See [SPRINT_4_REPORT.md](SPRINT_4_REPORT.md) and
[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## Setup

```bash
# Create venv
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Run tests (must all pass before any backtest)
pytest tests/ -v

# Run reference SMA strategy
python strategies/sma_crossover.py

# Run overnight gap diagnostic
python strategies/overnight_diagnostic.py

# Run gap continuation hypothesis test
python strategies/gap_continuation.py

# Run sprint 2 integration (walk-forward + Monte Carlo + lockbox)
python strategies/sprint2_integration.py

# Generate all visualizations
python strategies/generate_reports.py
```

## Project structure

```
algo_project/
├── README.md
├── SPRINT_2_REPORT.md
├── requirements.txt
├── config/config.yaml
├── data/
│   ├── raw/
│   └── processed/   # cached parquet
├── src/             # framework modules
├── strategies/      # concrete strategies + research scripts
├── tests/           # unit tests (24 passing)
└── reports/         # generated outputs (plots, CSVs)
```

## Hard rules (do not violate)

1. **No look-ahead bias.** Ever.
2. **Out-of-sample data is locked** until final test. The lockbox audits.
3. **Risk per trade ≤ 0.4%** of equity.
4. **All strategies inherit from Strategy base class.**
5. **All metrics go through metrics.py** (tested).
6. **Killed strategies stay killed.** No re-litigating after OOS failure.

## Roadmap

- [x] Sprint 1: framework + SMA crossover end-to-end
- [x] Sprint 2: walk-forward + Monte Carlo + parameter sweep + lockbox + first hypotheses
- [x] Sprint 3: intraday data + ORB + time-of-day + cross-sectional momentum (SURVIVOR)
- [x] Sprint 4: Canadian universe + quarterly rebalance + LIVE DEPLOYMENT SYSTEM
- [ ] Sprint 5+: paper-trade 3-6 months, then real capital (or pivot to quant career)

## Live deployment commands

```bash
# Show current signal & trade ticket (no state change)
python -m strategies.csm_live --capital 5000 --check

# Generate ticket if rebalance due
python -m strategies.csm_live --capital 5000 --rebalance

# Record fills after manual execution at broker
python -m strategies.csm_live --record --capital 5000 --auto

# Historical replay + upcoming schedule
python strategies/csm_forward.py
```

## Decisions log

- **2026-05-05 [S4]:** Pivoted to Canadian-listed universe (XEG.TO etc).
  TFSA tax efficiency, no US dividend withholding.
- **2026-05-05 [S4]:** Quarterly rebalancing. 3x lower cost than monthly,
  marginal signal cost.
- **2026-05-05 [S4]:** **DEPLOYMENT PARAMS LOCKED:** top-3, 9m lookback,
  quarterly, 13 CA ETFs. OOS Sharpe 1.25, CAGR 15.80%, MaxDD -13.08%.
  Changes require new OOS validation.
- **2026-05-05 [S4]:** Manual execution workflow (no algo trading).
- **2026-05-05 [S3]:** Killed H2 (ORB), H3 (time-of-day). Confirmed no
  intraday edge in SPY at retail data resolution.
- **2026-05-05 [S3]:** **H4 (CSM) SURVIVED** in initial US universe.
- **2026-05-05 [S3]:** Fixed CAGR/Sharpe periodicity bug. 7 regression tests.
- **2026-05-05 [S2]:** Killed H1 (overnight gap reversion) and H1B (continuation).
- **2026-05-05 [S2]:** Reframed objective from "build prop firm algo" to
  "build systematic trading capability."

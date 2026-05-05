# Sprint 2 Report — Algo Project

**Date:** 2026-05-05
**Sprint:** 2 of N
**Status:** Complete. Framework validated, first hypotheses tested and killed.

---

## Executive Summary

The framework works. Two hypotheses (overnight gap reversion and gap continuation)
were tested, evaluated against realistic transaction costs, and **killed**. This is
a successful sprint — the framework correctly identified strategies that lacked
edge before any capital was deployed.

The most important outcome of this sprint is **not finding a working strategy.**
It is **proving the framework can identify the difference between real edge and
backtest illusion.** That capability is the actual asset.

---

## What was built (Sprint 1 + Sprint 2 combined)

| Module | Status | Test coverage |
|---|---|---|
| Data loader (yfinance + parquet cache) | ✅ | Validation in production code |
| Performance metrics (Sharpe, DD, Calmar, etc.) | ✅ | 14 unit tests passing |
| RiskManager (sizing, kill switches) | ✅ | 10 unit tests passing |
| Backtest engine (long-only daily) | ✅ | Validated vs. expected SMA results |
| Walk-forward analysis | ✅ | 11-fold demo on SMA |
| Monte Carlo (trade resampling) | ✅ | 1000 sims on SMA trades |
| Block bootstrap (return resampling) | ✅ | 5-yr forecast distribution |
| Parameter sweep + plateau scoring | ✅ (minor NaN bug) | 15-cell grid on SMA |
| Out-of-sample lockbox + audit log | ✅ | Demo unlock logged |
| Strategy base class | ✅ | 3 strategies inherit cleanly |

Total: **24/24 unit tests pass**.

---

## Strategies tested

### SMA(50, 200) — Reference benchmark

Purpose: validate framework end-to-end. Not expected to be profitable.

| Metric | SMA(50,200) | Buy & Hold |
|---|---|---|
| CAGR | 7.49% | 11.63% |
| Sharpe | 0.63 | 0.73 |
| Max DD | -32.23% | -34.10% |
| N trades | 7 | 1 |

**Verdict:** Framework outputs match expectations. SMA underperforms B&H in a
14-year bull run (correct). Drawdown marginally better (correct — SMA exits
during sustained downtrends but can't catch fast crashes like 2020).

### H1 — Overnight Gap Mean Reversion (KILLED)

**Hypothesis:** After large overnight gaps, intraday returns reverse.

**Result:** All threshold variants (0.5σ to 3.0σ) lose money. At low thresholds,
catastrophic losses (-21% CAGR at 0.5σ). At high thresholds, edge approaches zero
but doesn't survive costs.

**Why it fails:** SPY has positive overnight drift (+7% annualized). Fading
gaps fights the persistent positive return. The "mean reversion" intuition was
wrong for this instrument.

### H1B — Overnight Gap Continuation (KILLED)

**Hypothesis:** After large upward overnight gaps, intraday returns continue up.

**Diagnostic evidence (positive, before costs):**
- Gaps ≥ +2σ: intraday mean +15.5 bps, win rate 66.3%, N=95
- Gaps ≥ +1σ: intraday mean +8.7 bps, win rate 61.8%, N=476

**Backtest result (after costs):** Marginal positive Sharpe (+0.08) at 2.0σ,
zero across all other thresholds.

**OOS validation:** Best in-sample threshold (1.25σ on 2010-2019, Sharpe 0.06)
deployed on 2020-2024. **Result: Sharpe -0.54, CAGR -2.29%.** Edge did not
survive out-of-sample.

**Why it fails:**
1. The signal exists but is small relative to round-trip costs (~10bps slippage
   on SPY ETF).
2. The 2010s effect was at least partly regime-specific and did not persist.
3. Trade frequency at meaningful thresholds is too low to overcome variance.

---

## Walk-forward results — SMA(50, 200) by year

| Test year | OOS Sharpe | Max DD | N trades |
|---|---|---|---|
| 2013 | 2.10 | -5.8% | 0 |
| 2014 | 0.94 | -7.3% | 0 |
| 2015 | -0.08 | -11.7% | 1 |
| 2016 | 0.61 | -6.6% | 1 |
| 2017 | 2.73 | -2.9% | 0 |
| 2018 | -0.17 | -9.7% | 0 |
| 2019 | 1.18 | -6.3% | 1 |
| 2020 | -0.01 | -32.3% | 1 |
| 2021 | 1.68 | -5.2% | 0 |
| 2022 | -1.14 | -10.9% | 0 |
| 2023 | 1.21 | -9.8% | 1 |

**Mean OOS Sharpe: 0.82, Std: 1.13**

This is what real trend-following looks like — high variance across regimes,
some years brilliant, some years awful. Not a deployable retail edge in this
form, but the analysis is correct.

---

## Monte Carlo / risk forecast

**SMA(50, 200) trade-order randomization (1000 simulations):**
- Drawdown P5 (best 5%): -4.75%
- Drawdown P50 (median): -8.01%
- Drawdown P95 (worst 5%): -20.94%

**Implication for prop firm rules:** Even SMA, with relatively modest realized
drawdown (~32% over 14 years driven by 2020 specifically), has a 5% chance of
producing >20% drawdown in path-dependent variation. Strategies need ~3-4×
margin between expected and tolerated drawdown to fit prop firm rules.

---

## Critical findings for the project direction

### Finding 1: Daily SPY data is insufficient for serious gap strategies

The signal exists in the data but is consumed by the modeled 5bps slippage per
side. Real intraday futures (MES/MNQ) would have:
- Lower commission per dollar of notional
- Tighter spreads in liquid contracts
- More granular timing (don't need to enter exactly at official open)

**Decision:** Sprint 3 requires real intraday futures data. yfinance daily is
no longer sufficient for the strategies we want to research.

### Finding 2: Naive overnight gap strategies don't have retail edge

Both directions tested. Both fail post-costs. This eliminates Hypothesis Family
H1 from further consideration.

**Decision:** Pivot to:
- H2 (Opening range breakout) — needs intraday data
- H3 (Time-of-day vol) — needs intraday data
- A new candidate: cross-sectional momentum on diversified ETFs, weekly
  rebalanced — works at retail scale, doesn't fit prop firm but useful as
  own-capital deployment

### Finding 3: Walk-forward variance is the binding constraint

SMA shows mean Sharpe 0.82 across 11 OOS folds with std 1.13. That's a
distribution that looks profitable on average but is unstable per-year. For
prop firm deployment with monthly profit targets and daily drawdown limits,
this kind of variance is unsurvivable.

**Decision:** Any strategy considered for prop firm deployment must show:
- Mean OOS Sharpe ≥ 1.0
- Std of OOS Sharpe across folds ≤ 0.7
- Max single-fold drawdown ≤ 50% of prop firm allowance

Without these, the variance will produce an account closure inside 12 months
with high probability.

---

## Decisions made this sprint (no further input required)

1. **Killed H1 (gap reversion).** No further work on this family.
2. **Killed H1B (gap continuation).** Doesn't survive OOS or costs.
3. **Daily yfinance data is insufficient.** Sprint 3 requires intraday futures
   data. Budget approved (mental): up to $150/month for proper data.
4. **Long-only equity backtest engine is sufficient for now.** Short and
   futures support deferred until first viable strategy is identified.
5. **OOS lockbox is non-negotiable going forward.** Every future strategy goes
   through the lockbox or gets rejected.
6. **Plateau scoring NaN bug** logged but not blocking. Fix in Sprint 3.

---

## Sprint 3 plan

**Goal:** Find one strategy that survives the gauntlet, including OOS.

1. **Acquire futures intraday data.**
   - Evaluate Databento, IBKR historical, or Polygon.io for MES/MNQ
   - Get 5+ years of 1-minute or tick data
   - Implement continuous contract construction with proper roll handling

2. **Extend backtest engine for futures.**
   - Tick-size-aware sizing
   - Margin and overnight rules
   - Intraday-only execution (no overnight holding for prop firm fit)

3. **Test H2: Opening Range Breakout.**
   - First 30-minute range as reference
   - Trend filter (daily SMA slope or VIX regime)
   - Time-stop exit by N hours
   - Walk-forward + OOS gauntlet

4. **Test H3: Time-of-day volatility expansion.**
   - First hour vs lunch vs close behavior
   - Statistical edge identification before strategy design

5. **Parallel: research cross-sectional momentum on ETFs.**
   - Different deployment vehicle (own capital, not prop firm)
   - Real edge documented
   - Lower variance

**Sprint 3 kill criteria:**
- If by end of Sprint 3 no hypothesis has survived the gauntlet → reassess
  whether retail prop-firm-fit edge exists at all in this market regime.

---

## Files delivered

```
algo_project/
├── README.md
├── requirements.txt
├── config/config.yaml
├── src/
│   ├── data_loader.py     [Sprint 1]
│   ├── metrics.py         [Sprint 1]
│   ├── risk.py            [Sprint 1]
│   ├── strategy.py        [Sprint 1]
│   ├── backtest.py        [Sprint 1]
│   ├── plots.py           [Sprint 1]
│   ├── lockbox.py         [Sprint 2]
│   ├── walkforward.py     [Sprint 2]
│   ├── montecarlo.py      [Sprint 2]
│   └── sweep.py           [Sprint 2]
├── strategies/
│   ├── sma_crossover.py
│   ├── overnight_gap.py
│   ├── overnight_diagnostic.py
│   ├── gap_continuation.py
│   ├── sprint2_integration.py
│   └── generate_reports.py
├── tests/
│   ├── test_metrics.py    [14 tests]
│   └── test_risk.py       [10 tests]
└── reports/
    ├── fig1_sma_vs_buyhold.png
    ├── fig2_overnight_diagnostic.png
    ├── fig3_walkforward.png
    ├── fig4_monte_carlo.png
    ├── fig5_gap_continuation_sweep.png
    ├── sma_crossover_trades.csv
    └── sma_crossover_equity.png
```

**Lines of code:** ~1100 across src/ + strategies/, ~200 in tests/.
**Test status:** 24/24 passing.

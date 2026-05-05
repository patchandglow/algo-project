"""Canadian-listed ETF universe for CSM in TFSA.

Why Canadian-listed:
  - TFSA holdings of US-listed ETFs trigger 15% US dividend withholding
    (no relief in TFSA, unlike RRSP).
  - All Canadian-listed ETFs settle in CAD — no FX spread on rebalance.
  - All trade on TSX during Canadian market hours.

Universe selection criteria:
  1. Sufficient history (≥ 12 years for 12-month lookback + warmup + test)
  2. Liquid (>100k shares ADV)
  3. Diversified across sectors AND asset classes (key for momentum to work)
  4. Tradable on Wealthsimple Trade and Questrade (the two major retail brokers)

Final universe = 13 ETFs:
  - 7 Canadian sectors (XEG, XFN, XIT, XMA, XRE, XUT, XST)
  - 2 US-equity exposure CAD-hedged (XSP S&P, ZQQ Nasdaq)
  - 1 International developed (XEF)
  - 1 Emerging markets (XEC)
  - 1 Long bonds (XLB)
  - 1 Gold (CGL)

Inception date constraint: Earliest start = max(individual inception dates).
With XCD/XEC at 2013-04, our earliest viable backtest = 2014-04 (after
12-month lookback). That's still ~11 years of usable data through 2025.
"""

# Core universe — Canadian-listed, TFSA-friendly, diversified
CA_UNIVERSE = [
    # Canadian sectors
    "XEG.TO",    # Energy
    "XFN.TO",    # Financials
    "XIT.TO",    # Tech (Canadian — narrow but exists)
    "XMA.TO",    # Materials
    "XRE.TO",    # REIT
    "XUT.TO",    # Utilities
    "XST.TO",    # Consumer Staples
    # US equity exposure (CAD-hedged, no withholding when held in CAD wrappers)
    "XSP.TO",    # S&P 500 CAD-hedged
    "ZQQ.TO",    # Nasdaq 100 CAD-hedged
    # International / EM
    "XEF.TO",    # International developed (ex North America)
    "XEC.TO",    # Emerging markets
    # Defensive / non-equity
    "XLB.TO",    # Long-term Canadian bonds
    "CGL.TO",    # Gold (CAD-hedged)
]

CA_DESCRIPTIONS = {
    "XEG.TO":  "iShares S&P/TSX Capped Energy",
    "XFN.TO":  "iShares S&P/TSX Capped Financials",
    "XIT.TO":  "iShares S&P/TSX Capped Information Tech",
    "XMA.TO":  "iShares S&P/TSX Capped Materials",
    "XRE.TO":  "iShares S&P/TSX Capped REIT",
    "XUT.TO":  "iShares S&P/TSX Capped Utilities",
    "XST.TO":  "iShares S&P/TSX Capped Consumer Staples",
    "XSP.TO":  "iShares Core S&P 500 (CAD-Hedged)",
    "ZQQ.TO":  "BMO NASDAQ 100 (CAD-Hedged)",
    "XEF.TO":  "iShares Core MSCI EAFE IMI",
    "XEC.TO":  "iShares Core MSCI Emerging Markets IMI",
    "XLB.TO":  "iShares Core Canadian Long-Term Bond",
    "CGL.TO":  "iShares Gold Bullion ETF (CAD-Hedged)",
}

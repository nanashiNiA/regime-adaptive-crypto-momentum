# Regime-Adaptive Cross-Sectional Crypto Momentum (RACM)

Multi-asset long/short momentum strategy for cryptocurrency markets with regime detection, crypto-native quality gates, and Kelly-based position sizing.

## Architecture

```
[1H BTC Price + Derivatives]        [8H 6-Asset Prices]
         |                                |
   Crypto Signal Generation         LS Momentum Ranking
   (Funding z, OI z, Liq z)        (60d+90d lookback)
         |                                |
         v                                v
   Quality Gate ─────────────> Position Sizing
   (crash block,                 (Kelly f=0.15,
    grind bear detect)            VT scaling)
         |                                |
         +────────> Regime Detection <────+
                    (4-stage: MA110d,
                     skew<-0.5, DD<-0.12)
                          |
                          v
                   Equity DD Control
                   (-15% → 0.7x, -22% → 0.4x, -30% → 0.1x)
                          |
                          v
                    Final Position (cap ≤ 3.0x)
```

## Key Results

### In-Sample (2021-2024, 6 assets, realistic slippage)

| Metric | Value |
|--------|------:|
| WFS (monthly mean × 12) | 367% |
| Win rate | 39/45 folds (87%) |
| Bear 2022 return | +121% |
| Worst fold MaxDD | -15.7% |
| Shuffle test (1000×3) | p=0.000 (all PASS) |
| Leak check | 21/21 PASS |

### Out-of-Sample (2025, no parameter changes)

| Metric | Value |
|--------|------:|
| Annual return | +93% |
| BTC Buy & Hold | -7% |
| Outperformance | +100pt |
| MaxDD | -13.6% |
| Win months | 8/12 (67%) |

### Statistical Robustness

| Test | Result |
|------|--------|
| t-test: mean > 0% | p < 0.0001 (certain) |
| t-test: mean > 25% (WFS>300%) | p = 0.206 (NOT proven) |
| Win rate > 50% (binomial) | p < 0.0001 (certain) |
| Bootstrap 95% CI (block) | [200%, 575%] |
| Fold Sharpe (annualized) | 2.35 |

**Honest note**: WFS 367% depends on 3 explosive months. Removing the top fold (+255.7%) alone drops WFS to 305%. The t-test cannot prove WFS > 300% at p < 0.05. Win rate 87% and OOS +93% are the statistically strongest claims.

## Components

| Component | Method | Source |
|-----------|--------|--------|
| LS Momentum | Cross-sectional, vol-adjusted | Moskowitz et al. (2012) |
| Regime Detection | 4-stage (MA/skew/DD composite) | Multi-indicator |
| Crypto Quality Gates | Funding z, OI z, Liq cascade | best.py project |
| Position Sizing | Fractional Kelly (f=0.15) | Thorp (1969) |
| DD Control | 3-level equity-based | Dynamic risk management |
| Execution | Lighter.xyz DEX (zero fees) | 6 liquid assets |
| Slippage | Per-asset 3-8 bps | BTC/ETH: 3bps, SOL/XRP/DOGE: 5bps, LINK: 8bps |

## Assets

BTC, ETH, SOL, XRP, DOGE, LINK (6 high-liquidity assets on Lighter.xyz)

## Walk-Forward Structure

- Train: 3 months (~2,160 hourly bars)
- Test: 1 month (~720 hourly bars)
- Rolling: monthly, 45 folds (2021-04 to 2024-12)
- Optimization: Kelly fraction only (KF ∈ {0.15, 0.20, 0.25, 0.30})
- Result: KF=0.15 selected in 43/45 folds (stable)

## Data Requirements

| Data | Source | Resolution |
|------|--------|-----------|
| BTC OHLC | Binance API | 1H |
| ETH/SOL/XRP/DOGE/LINK | Binance API | 8H |
| Funding rate | Binance Futures | 1H |
| Open Interest | Binance Futures | 1H |
| Liquidation count | Tardis / Custom collector | 1H |

## Files

### Scripts

| File | Description |
|------|-------------|
| `scripts/v24_6asset_realistic.py` | **Main model**: 6-asset proper WF with realistic slippage |
| `scripts/v24_6asset_shuffle.py` | Shuffle test (1000 iterations × 3 types) |
| `scripts/v24_2025_oos.py` | 2025 out-of-sample test |
| `scripts/v24_honest_audit.py` | Leak impact quantification |
| `scripts/v24_final.py` | Position cap sweep + initial shuffle test |
| `scripts/v24_proper_wf.py` | Proper WF (Train 3M → Test 1M) |
| `scripts/v24_wf_table.py` | 45-fold detailed table + leak checklist |
| `scripts/v24_sanity_check.py` | Leverage & PnL distribution verification |
| `scripts/v24_hybrid_1h.py` | Initial 1H hybrid (pre-position cap) |
| `scripts/v24_hybrid.py` | Initial 8H hybrid |
| `scripts/v23_bestpy_integration.py` | best.py technique integration test |

### Docs

| File | Description |
|------|-------------|
| `docs/model_results.md` | Comprehensive model results |
| `docs/presentation.md` | Presentation material with all data |
| `docs/presentation_memo.md` | Presentation speaking notes |

### Results

| File | Description |
|------|-------------|
| `results/shuffle_1000_results.txt` | 1000-iteration shuffle test output |

## Known Limitations

1. **IS→OOS degradation**: WFS 367% (IS) → 81% (OOS), -78% decay
2. **Outlier dependence**: Top 3 folds account for disproportionate share of WFS
3. **cap=3.0 is result-driven**: cap=2.5 gives WFS 290% (below 300%)
4. **Capacity**: $50K-200K max (DEX liquidity constraint)
5. **Binance signal → Lighter execution**: Price/funding mismatch risk
6. **Liquidation data**: Requires 24/7 custom WebSocket collector

## Honest Performance Tiers

| Tier | WFS | Condition |
|------|----:|-----------|
| Fully blind (zero data contact) | ~160% | All theory defaults |
| Theory defaults + 1H | 224% | cap=2.0, LW=0.50 |
| Inherited params | 290% | cap=2.5, LW=0.80 |
| Reported (result-optimized) | 367% | cap=3.0, all features |
| OOS reality (2025) | 81% | Forward performance |

"""
RACM Core Engine - Single source of truth for all signal/position logic.
Used by: backtest, OOS test, production bot.
No data fetching here - pure computation only.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class RACMParams:
    """All model parameters in one place"""
    # Assets
    assets: list = field(default_factory=lambda: ['BTC','ETH','SOL','XRP','DOGE','LINK'])
    slippage_bps: dict = field(default_factory=lambda: {'BTC':3,'ETH':3,'SOL':5,'XRP':5,'DOGE':5,'LINK':8})

    # LS Momentum
    ls_lookbacks: list = field(default_factory=lambda: [60, 90])  # days
    ls_weight: float = 0.80

    # Regime
    regime_ma_hours: int = 2640        # 110 days
    regime_ma_short_hours: int = 480   # 20 days
    regime_skew_threshold: float = -0.5
    regime_dd_threshold: float = -0.12
    regime_skew_window: int = 720      # 30 days in 1H
    regime_peak_window: int = 1080     # 45 days in 1H
    regime_crash_ret: float = -0.006
    regime_crash_cooldown: int = 4     # hours

    # Crypto gates
    funding_z_bull: float = -1.5
    funding_z_bear: float = 2.0
    oi_z_threshold: float = -1.5
    liq_z_threshold: float = 2.0
    grind_atr: float = 0.006
    crash_signals: int = 3
    vol_ratio_threshold: float = 1.5
    vol_ratio_gate: float = 0.85
    crypto_alpha_weight: float = 0.10

    # Position sizing
    kelly_fraction: float = 0.15
    position_cap: float = 3.0
    vol_target_base: float = 1.5
    kelly_lookback_hours: int = 2160   # 90 days

    # DD control
    dd_levels: list = field(default_factory=lambda: [(-0.15, 0.7), (-0.22, 0.4), (-0.30, 0.1)])

    # Warmup
    warmup_hours: int = 2760  # 115 days


class RACMFeatures:
    """Compute all features from 1H price/derivatives data. All lag-1+."""

    @staticmethod
    def funding_zscore(funding_rate: np.ndarray) -> np.ndarray:
        """Funding rate z-score. lag-2 (raw lag-1 + normalization lag-1)"""
        fr = np.roll(funding_rate, 1)
        fr_mean = pd.Series(fr).rolling(168, min_periods=48).mean().values
        fr_std = pd.Series(fr).rolling(168, min_periods=48).std().values
        return np.roll(np.where(fr_std > 1e-10, (fr - fr_mean) / fr_std, 0), 1)

    @staticmethod
    def oi_zscore(oi: np.ndarray) -> np.ndarray:
        """OI z-score. lag-2 (raw oi[i-1] + np.roll)"""
        n = len(oi)
        oi_raw = np.zeros(n)
        for i in range(25, n):
            if not np.isnan(oi[i-1]) and not np.isnan(oi[i-25]):
                oi_raw[i] = oi[i-1] - oi[i-25]
        oi_m = pd.Series(oi_raw).rolling(168, min_periods=48).mean().values
        oi_s = pd.Series(oi_raw).rolling(168, min_periods=48).std().values
        return np.roll(np.where(oi_s > 1e-10, (oi_raw - oi_m) / oi_s, 0), 1)

    @staticmethod
    def oi_change_24h(oi: np.ndarray) -> np.ndarray:
        """OI 24h percentage change. lag-1"""
        n = len(oi)
        oc = np.zeros(n)
        for i in range(25, n):
            if oi[i-1] > 0 and not np.isnan(oi[i-1]) and not np.isnan(oi[i-25]):
                oc[i] = (oi[i-1] - oi[i-25]) / (oi[i-25] + 1e-10)
        return oc

    @staticmethod
    def liq_zscore(liq_count: np.ndarray) -> np.ndarray:
        """Liquidation cascade z-score. lag-1"""
        liq_24h = np.roll(pd.Series(liq_count).rolling(24, min_periods=6).sum().values, 1)
        liq_m = pd.Series(liq_24h).rolling(168, min_periods=48).mean().values
        liq_s = pd.Series(liq_24h).rolling(168, min_periods=48).std().values
        return np.where(liq_s > 1e-10, (liq_24h - liq_m) / liq_s, 0)

    @staticmethod
    def vol_ratio(ret: np.ndarray) -> np.ndarray:
        """Short/long vol ratio. lag-1"""
        vs = np.roll(pd.Series(np.abs(ret)).rolling(9, min_periods=3).mean().values, 1)
        vl = np.roll(pd.Series(np.abs(ret)).rolling(720, min_periods=240).mean().values, 1)
        return vs / (vl + 1e-10)

    @staticmethod
    def rv_pctrank(ret: np.ndarray) -> np.ndarray:
        """Realized vol percentile rank. lag-1"""
        return np.roll(
            pd.Series(np.abs(ret)).rolling(24, min_periods=6).sum()
            .rolling(720, min_periods=168).rank(pct=True).values, 1)

    @staticmethod
    def portfolio_vol(ret: np.ndarray) -> np.ndarray:
        """Annualized portfolio vol. lag-1"""
        return np.roll(pd.Series(ret).rolling(720, min_periods=240).std().values * np.sqrt(365*24), 1)

    @staticmethod
    def ret_nd(log_price: np.ndarray, lag_bars: int) -> np.ndarray:
        """N-bar log return. lag-1"""
        n = len(log_price)
        r = np.zeros(n)
        for i in range(lag_bars + 1, n):
            r[i] = log_price[i-1] - log_price[i-lag_bars-1]
        return r

    @staticmethod
    def atr_pct(ret: np.ndarray, price: np.ndarray) -> np.ndarray:
        """ATR as % of price. lag-1"""
        return np.roll(pd.Series(np.abs(ret)).rolling(24, min_periods=6).mean().values / (price + 1e-12), 1)

    @staticmethod
    def ma_cross_slow(price: np.ndarray) -> np.ndarray:
        """MA72/MA168 cross. lag-1"""
        ma72 = pd.Series(price).ewm(span=72, min_periods=24).mean().values
        ma168 = pd.Series(price).ewm(span=168, min_periods=48).mean().values
        return np.roll((ma72 - ma168) / (price + 1e-12), 1)


class RACMRegime:
    """4-stage regime detection. Identical logic for backtest and bot."""

    @staticmethod
    def compute(price: np.ndarray, ret: np.ndarray, params: RACMParams) -> np.ndarray:
        """Returns bp_h array (position multiplier per bar)"""
        n = len(price)
        S = params.warmup_hours
        ma_long = pd.Series(price).rolling(params.regime_ma_hours, min_periods=params.regime_ma_hours//2).mean().values
        ma_short = pd.Series(price).rolling(params.regime_ma_short_hours, min_periods=params.regime_ma_short_hours//2).mean().values
        skew = pd.Series(ret).rolling(params.regime_skew_window, min_periods=params.regime_skew_window//2).skew().values
        pk = pd.Series(price).rolling(params.regime_peak_window, min_periods=params.regime_peak_window//2).max().values
        dd = (price - pk) / pk
        rv = pd.Series(ret).rolling(params.regime_skew_window, min_periods=params.regime_skew_window//2).std().values * np.sqrt(365*24)
        c_sum = pd.Series(ret).rolling(params.regime_skew_window, min_periods=params.regime_skew_window//2).sum().values

        ma_slope = np.zeros(n)
        for i in range(S, n):
            if (not np.isnan(ma_long[i-1]) and not np.isnan(ma_long[i-121])
                    and ma_long[i-121] > 0):
                ma_slope[i] = (ma_long[i-1] - ma_long[i-121]) / ma_long[i-121]

        bp = np.ones(n)
        m1 = 0; ch = 0
        for i in range(S, n):
            dl = 0; ds = 0
            if not np.isnan(ma_long[i-1]) and price[i-1] < ma_long[i-1]: dl += 1
            if not np.isnan(ma_short[i-1]) and price[i-1] < ma_short[i-1]: ds += 1
            if not np.isnan(skew[i-1]) and skew[i-1] < params.regime_skew_threshold: dl += 1; ds += 1
            if dd[i-1] < params.regime_dd_threshold: dl += 1; ds += 1
            dc = dl * 0.3 + ds * 0.7

            if i >= 2 and ret[i-2] < params.regime_crash_ret:
                m1 = params.regime_crash_cooldown
            if m1 > 0: m1 -= 1

            if dc >= 1.5:
                if ma_slope[i] < -0.001 and ret[i-1] < 0: bp[i] = -0.7
                elif ma_slope[i] > 0.0005: bp[i] = 0.5
                else: bp[i] = 0.2
                continue
            if dc >= 0.8: bp[i] = 0.5; ch = 120
            elif dc >= 0.5: bp[i] = 0.7; ch = 120
            else:
                if ch > 0:
                    ch -= 1
                    bp[i] = 0.7 if not (not np.isnan(c_sum[i-1]) and c_sum[i-1] > 0.05) else 1.0
                elif (not np.isnan(rv[i-1]) and rv[i-1] < 0.50
                      and i >= 240 and np.sum(ret[i-240:i-1]) > 0):
                    bp[i] = 1.5
                else:
                    bp[i] = 1.0
            if m1 > 0: bp[i] = min(bp[i], 0.7)
        return bp


class RACMCryptoGates:
    """Crypto quality gates + alpha signal"""

    @staticmethod
    def compute(funding_z: float, oi_z: float, oi_chg: float,
                liq_z: float, rv_pctrank: float, bp: float,
                atr_pct: float, ret_30d: float, ma_cross_slow: float,
                vr: float, ret_1d: float, params: RACMParams) -> tuple:
        """Returns (crypto_alpha, gate_multiplier)"""
        # Crypto alpha
        cs = 0.0
        if funding_z < -2.0: cs += 0.5
        elif funding_z < params.funding_z_bull: cs += 0.3
        if oi_z < params.oi_z_threshold and ret_1d < -0.02: cs += 0.3
        if oi_chg > 0.015 and ret_1d < -0.015: cs += 0.2
        if funding_z > params.funding_z_bear: cs -= 0.3
        if liq_z > params.liq_z_threshold: cs -= 0.3
        cs = float(np.clip(cs, -1, 1))

        # Gate
        n_danger = int(rv_pctrank > 0.95) + int(liq_z > params.liq_z_threshold) + int(bp < 0.3)
        grind = (atr_pct < params.grind_atr and ret_30d < -0.05
                 and ma_cross_slow < 0 and rv_pctrank < 0.50)
        gate = 1.0
        if n_danger >= params.crash_signals: gate = 0.1
        elif grind: gate = 0.5
        elif vr > params.vol_ratio_threshold: gate = params.vol_ratio_gate
        return cs, gate


class RACMKelly:
    """Kelly criterion + vol targeting"""

    @staticmethod
    def compute(past_returns: np.ndarray, params: RACMParams) -> tuple:
        """Returns (leverage, vol_target_scale)"""
        if len(past_returns) < 240:
            return 1.5, params.vol_target_base
        mu = np.mean(past_returns)
        var = np.var(past_returns) + 1e-10
        lev = float(np.clip(mu / var * params.kelly_fraction, 1.0, params.position_cap))
        vol_p = np.std(past_returns) * np.sqrt(365 * 24)
        if vol_p > 0:
            vt = float(np.clip(params.vol_target_base / vol_p, 0.5, params.position_cap / lev))
        else:
            vt = 1.0
        return lev, vt


class RACMDDControl:
    """Equity drawdown control"""

    @staticmethod
    def compute(equity: float, peak_equity: float, params: RACMParams) -> float:
        """Returns DD multiplier (0.1 to 1.0)"""
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        for level, mult in sorted(params.dd_levels):
            if dd < level:
                return mult
        return 1.0


class RACMLS:
    """Cross-sectional LS momentum"""

    @staticmethod
    def compute_ranking(asset_returns: dict, asset_names: list,
                        lookbacks: list, bar_multiplier: int = 3) -> tuple:
        """Returns (long_asset, short_asset) based on vol-adjusted momentum.
        bar_multiplier: bars per day for this frequency (3 for 8H, 24 for 1H)
        """
        na = len(asset_names)
        if na < 3:
            return '', ''

        # Vol for each asset
        a_vol = {}
        for name in asset_names:
            r = asset_returns[name]
            vol = np.roll(pd.Series(np.abs(r)).rolling(30, min_periods=10).mean().values, 1)
            a_vol[name] = vol

        # Score = average momentum across lookbacks
        scores = {name: 0.0 for name in asset_names}
        valid = False
        for lb_days in lookbacks:
            lb = lb_days * bar_multiplier
            for name in asset_names:
                r = asset_returns[name]
                if len(r) < lb + 10:
                    continue
                mom = np.roll(pd.Series(r).rolling(lb, min_periods=lb//3).sum().values, 1)
                if len(mom) > 0 and not np.isnan(mom[-1]) and a_vol[name][-1] > 1e-10:
                    scores[name] += mom[-1] / a_vol[name][-1]
                    valid = True

        if not valid:
            return '', ''

        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_assets[0][0], sorted_assets[-1][0]

    @staticmethod
    def compute_pnl_8h(asset_returns_8h: dict, lookbacks_days: list,
                       n_bars: int) -> tuple:
        """Compute LS PnL array at 8H resolution.
        Returns (lp_8h, ls_long_names, ls_short_names, ls_slip_costs)
        """
        na = len(asset_returns_8h)
        a_vol = {n: np.roll(pd.Series(np.abs(r)).rolling(30, min_periods=10).mean().values, 1)
                 for n, r in asset_returns_8h.items()}

        lp = np.zeros(n_bars)
        long_names = [''] * n_bars
        short_names = [''] * n_bars

        for lb_days in lookbacks_days:
            lb = lb_days * 3  # 3 bars per day at 8H
            w = 1.0 / len(lookbacks_days)
            am = {n: np.roll(pd.Series(r).rolling(lb, min_periods=lb//3).sum().values, 1)
                  for n, r in asset_returns_8h.items()}
            start = lb + 10
            for i in range(start, n_bars):
                moms = [(am[n][i] / (a_vol[n][i] + 1e-10), n, asset_returns_8h[n][i])
                        for n in asset_returns_8h if not np.isnan(am[n][i])]
                if len(moms) < 3:
                    continue
                moms.sort(key=lambda x: x[0], reverse=True)
                lp[i] += (moms[0][2] / na - moms[-1][2] / na) * w
                if lb_days == lookbacks_days[0]:
                    long_names[i] = moms[0][1]
                    short_names[i] = moms[-1][1]

        return lp, long_names, short_names

    @staticmethod
    def map_8h_to_1h(lp_8h: np.ndarray, index_8h, index_1h, nn: int) -> np.ndarray:
        """Distribute 8H LS PnL to 1H bars"""
        lp_1h = np.zeros(nn)
        for j in range(len(index_8h)):
            ts = index_8h[j]
            ts_end = index_8h[j+1] if j+1 < len(index_8h) else ts + pd.Timedelta(hours=8)
            idxs = np.where((index_1h >= ts) & (index_1h < ts_end))[0]
            if len(idxs) > 0:
                for ii in idxs:
                    lp_1h[ii] = lp_8h[j] / len(idxs)
        return lp_1h

    @staticmethod
    def count_slippage(long_names: list, short_names: list,
                       slippage_bps: dict, na: int, start: int = 180) -> np.ndarray:
        """Compute per-bar slippage cost from LS ranking changes"""
        n = len(long_names)
        slip = np.zeros(n)
        for i in range(start + 1, n):
            if long_names[i] != long_names[i-1] and long_names[i-1]:
                old_s = slippage_bps.get(long_names[i-1], 10)
                new_s = slippage_bps.get(long_names[i], 10)
                slip[i] += (old_s + new_s) / 10000 / na
            if short_names[i] != short_names[i-1] and short_names[i-1]:
                old_s = slippage_bps.get(short_names[i-1], 10)
                new_s = slippage_bps.get(short_names[i], 10)
                slip[i] += (old_s + new_s) / 10000 / na
        return slip


def safe_val(arr, idx, default=0.0):
    """Safely get value from array, returning default if NaN"""
    v = arr[idx] if idx < len(arr) else default
    return v if not np.isnan(v) else default

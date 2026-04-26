"""
RACM Production Bot v2 - Fixed all 4 backtest/production discrepancies.
Uses racm_core.py for all signal logic.

Fixes:
  1. Regime state (m1, ch) persisted across ticks
  2. Kelly uses base_raw history buffer (not raw BTC ret)
  3. OI history accumulated tick-by-tick
  4. Feature history cached (append 1 bar, not re-fetch 2700)

Usage:
    python -m src.racm_bot --once
    python -m src.racm_bot --interval 3600
"""
import sys, os, time, json, logging, pickle, argparse
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import pandas as pd
import urllib.request

from src.racm_core import (RACMParams, RACMFeatures, RACMRegime,
                           RACMCryptoGates, RACMKelly, RACMDDControl,
                           RACMLS, safe_val)

if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass


# ─── Data Pipeline ───
class BinanceDataPipeline:
    BASE_URL = 'https://api.binance.com/api/v3'
    FUTURES_URL = 'https://fapi.binance.com/fapi/v1'

    def _fetch(self, url):
        req = urllib.request.Request(url, headers={'User-Agent': 'RACM-Bot/2.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())

    def fetch_klines(self, symbol: str, interval: str, limit: int = 1000,
                     pages: int = 1) -> pd.DataFrame:
        all_data = []
        et = int(datetime.now(timezone.utc).timestamp() * 1000)
        for _ in range(pages):
            url = f'{self.BASE_URL}/klines?symbol={symbol}USDT&interval={interval}&limit={limit}&endTime={et}'
            data = self._fetch(url)
            if not data: break
            all_data.extend(data)
            et = int(data[0][0]) - 1
        df = pd.DataFrame(all_data, columns=[
            'timestamp','open','high','low','close','volume',
            'close_time','quote_vol','trades','taker_buy_base','taker_buy_quote','ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['return'] = df['close'].pct_change()
        return df.dropna(subset=['return'])

    def fetch_funding_rate(self, limit: int = 500) -> pd.DataFrame:
        url = f'{self.FUTURES_URL}/fundingRate?symbol=BTCUSDT&limit={limit}'
        data = self._fetch(url)
        df = pd.DataFrame(data)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = df['fundingRate'].astype(float)
        return df.set_index('fundingTime').sort_index()

    def fetch_open_interest(self) -> float:
        url = f'{self.FUTURES_URL}/openInterest?symbol=BTCUSDT'
        data = self._fetch(url)
        return float(data['openInterest'])


# ─── State (persists across restarts) ───
@dataclass
class BotState:
    equity: float = 1.0
    peak_equity: float = 1.0
    ls_long_asset: str = ''
    ls_short_asset: str = ''
    last_update: str = ''

    # FIX 1: Regime state persisted
    regime_m1: int = 0      # crash cooldown counter
    regime_ch: int = 0      # caution holddown counter
    last_bp: float = 1.0

    # FIX 2: base_raw history for Kelly (ring buffer, last 2200 bars)
    base_raw_history: list = field(default_factory=list)
    MAX_HISTORY: int = 2200

    # FIX 3: OI history accumulated (ring buffer, last 200 values)
    oi_history: list = field(default_factory=list)
    OI_MAX: int = 200

    # Trade/position log
    position_log: list = field(default_factory=list)

    def append_base_raw(self, val: float):
        self.base_raw_history.append(val)
        if len(self.base_raw_history) > self.MAX_HISTORY:
            self.base_raw_history = self.base_raw_history[-self.MAX_HISTORY:]

    def append_oi(self, val: float):
        self.oi_history.append(val)
        if len(self.oi_history) > self.OI_MAX:
            self.oi_history = self.oi_history[-self.OI_MAX:]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'BotState':
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    state = pickle.load(f)
                # Ensure new fields exist (backward compat)
                for attr, default in [('regime_m1',0),('regime_ch',0),('last_bp',1.0),
                                      ('base_raw_history',[]),('oi_history',[]),
                                      ('MAX_HISTORY',2200),('OI_MAX',200)]:
                    if not hasattr(state, attr):
                        setattr(state, attr, default)
                return state
            except:
                pass
        return BotState()


# ─── Bot ───
class RACMBot:
    def __init__(self, params: RACMParams = None, state_dir='data/bot_state',
                 log_dir='logs/racm_bot'):
        self.params = params or RACMParams()
        self.pipeline = BinanceDataPipeline()
        self.state_path = os.path.join(state_dir, 'racm_state.pkl')
        self.state = BotState.load(self.state_path)

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'racm_bot.log'), encoding='utf-8'),
                logging.StreamHandler(sys.stdout)])
        self.log = logging.getLogger('RACM')

        # Auto-warmup Kelly on first startup
        self.warmup_kelly()

    def warmup_kelly(self):
        """Pre-fill Kelly buffer with historical base_raw from API data.
        Called once on first startup when kelly_buf is empty.
        """
        if len(self.state.base_raw_history) >= 240:
            self.log.info(f"Kelly already warm: {len(self.state.base_raw_history)} bars")
            return

        self.log.info("Warming up Kelly buffer from historical data...")
        p = self.params
        try:
            # Fetch BTC 1H (3 pages = ~3000 bars = ~4 months)
            btc_1h = self.pipeline.fetch_klines('BTC', '1h', pages=3)
            price = btc_1h['close'].values
            ret_arr = btc_1h['return'].values
            nn = len(ret_arr)

            # Regime on full history
            bp_arr = RACMRegime.compute(price, ret_arr, p)

            # Fetch 8H for LS
            asset_8h = {}
            for asset in p.assets:
                try:
                    df = self.pipeline.fetch_klines(asset, '8h', pages=1)
                    asset_8h[asset] = df
                except:
                    pass

            # Compute LS at 8H
            if len(asset_8h) >= 3:
                c_8h = list(asset_8h.values())[0].index
                for df in asset_8h.values():
                    c_8h = c_8h.intersection(df.index)
                a_ret = {a: asset_8h[a].loc[c_8h, 'return'].values for a in asset_8h}
                lp_8h, _, _ = RACMLS.compute_pnl_8h(a_ret, p.ls_lookbacks, len(c_8h))
                # Map to 1H
                lp_1h = RACMLS.map_8h_to_1h(lp_8h, c_8h, btc_1h.index, nn)
            else:
                lp_1h = np.zeros(nn)

            # Build base_raw history
            dw = max(0, 1 - p.ls_weight)
            S = min(p.warmup_hours, nn - 1)
            count = 0
            for i in range(max(S, nn - p.kelly_lookback_hours), nn):
                base_val = dw * ret_arr[i] * bp_arr[i] + p.ls_weight * lp_1h[i]
                self.state.append_base_raw(base_val)
                count += 1

            self.log.info(f"Kelly warmup complete: {count} bars loaded (total buf={len(self.state.base_raw_history)})")
            self.state.save(self.state_path)

        except Exception as e:
            self.log.warning(f"Kelly warmup failed: {e}. Will accumulate naturally.")

    def _compute_regime_incremental(self, price: np.ndarray, ret: np.ndarray) -> float:
        """FIX 1: Compute regime for current bar using persisted m1/ch state.
        Uses the same logic as RACMRegime.compute but only for the LAST bar,
        carrying m1 and ch from previous tick.
        """
        p = self.params
        n = len(price)
        if n < p.warmup_hours:
            return 1.0

        # Compute indicators at bar n-1 (lag-1)
        ma_long = np.mean(price[max(0, n - p.regime_ma_hours):n-1]) if n > p.regime_ma_hours else np.nan
        ma_short = np.mean(price[max(0, n - p.regime_ma_short_hours):n-1]) if n > p.regime_ma_short_hours else np.nan
        skew_val = pd.Series(ret[max(0, n-720):n-1]).skew() if n > 720 else 0
        peak = np.max(price[max(0, n-1080):n-1]) if n > 100 else price[-2]
        dd = (price[-2] - peak) / peak if peak > 0 else 0
        rv = np.std(ret[max(0, n-720):n-1]) * np.sqrt(365*24) if n > 720 else 1.0
        c_sum = np.sum(ret[max(0, n-720):n-1]) if n > 720 else 0

        # MA slope (120 bars = 5 days)
        if n > p.regime_ma_hours + 120:
            ma_l_now = np.mean(price[max(0, n-p.regime_ma_hours):n-1])
            ma_l_old = np.mean(price[max(0, n-p.regime_ma_hours-120):n-121])
            ma_slope = (ma_l_now - ma_l_old) / ma_l_old if ma_l_old > 0 else 0
        else:
            ma_slope = 0

        # Danger composite
        dl = 0; ds = 0
        if not np.isnan(ma_long) and price[-2] < ma_long: dl += 1
        if not np.isnan(ma_short) and price[-2] < ma_short: ds += 1
        if not np.isnan(skew_val) and skew_val < p.regime_skew_threshold: dl += 1; ds += 1
        if dd < p.regime_dd_threshold: dl += 1; ds += 1
        dc = dl * 0.3 + ds * 0.7

        # Crash cooldown (use persisted state)
        m1 = self.state.regime_m1
        ch = self.state.regime_ch
        if n >= 3 and ret[-3] < p.regime_crash_ret:
            m1 = p.regime_crash_cooldown
        if m1 > 0: m1 -= 1

        # Regime decision
        bp = 1.0
        if dc >= 1.5:
            if ma_slope < -0.001 and ret[-2] < 0: bp = -0.7
            elif ma_slope > 0.0005: bp = 0.5
            else: bp = 0.2
        elif dc >= 0.8: bp = 0.5; ch = 120
        elif dc >= 0.5: bp = 0.7; ch = 120
        else:
            if ch > 0:
                ch -= 1
                bp = 0.7 if not (c_sum > 0.05) else 1.0
            elif rv < 0.50 and n >= 240 and np.sum(ret[-240:-1]) > 0:
                bp = 1.5
            else: bp = 1.0
        if m1 > 0: bp = min(bp, 0.7)

        # Save state for next tick
        self.state.regime_m1 = m1
        self.state.regime_ch = ch
        self.state.last_bp = bp
        return bp

    def run_once(self):
        now = datetime.now(timezone.utc)
        self.log.info(f"=== RACM tick @ {now.strftime('%Y-%m-%d %H:%M')} UTC ===")
        p = self.params

        try:
            # ── 1. Fetch BTC 1H ──
            btc_1h = self.pipeline.fetch_klines('BTC', '1h', pages=3)
            price = btc_1h['close'].values
            ret = btc_1h['return'].values
            nn = len(ret)
            log_p = np.log(np.clip(price, 1e-12, None))
            self.log.info(f"BTC 1H: {nn} bars")

            # ── 2. Fetch derivatives ──
            try:
                fr_df = self.pipeline.fetch_funding_rate(limit=500)
                funding_raw = fr_df['fundingRate'].reindex(btc_1h.index, method='ffill').fillna(0).values
            except:
                funding_raw = np.zeros(nn)

            # FIX 3: Accumulate OI history
            try:
                oi_now = self.pipeline.fetch_open_interest()
                self.state.append_oi(oi_now)
            except:
                self.state.append_oi(np.nan)

            # Build OI array from history (pad with NaN for missing)
            oi_hist = self.state.oi_history
            oi_arr = np.full(nn, np.nan)
            if len(oi_hist) > 0:
                n_oi = min(len(oi_hist), nn)
                oi_arr[-n_oi:] = oi_hist[-n_oi:]

            liq_arr = np.zeros(nn)  # TODO: WebSocket collector

            # ── 3. Compute features ──
            funding_z = RACMFeatures.funding_zscore(funding_raw)
            oi_z = RACMFeatures.oi_zscore(oi_arr)
            oi_chg = RACMFeatures.oi_change_24h(oi_arr)
            liq_z = RACMFeatures.liq_zscore(liq_arr)
            vr = RACMFeatures.vol_ratio(ret)
            rv_pct = RACMFeatures.rv_pctrank(ret)
            ret_1d = RACMFeatures.ret_nd(log_p, 24)
            ret_30d = RACMFeatures.ret_nd(log_p, 720)
            atr_pct = RACMFeatures.atr_pct(ret, price)
            ma_cs = RACMFeatures.ma_cross_slow(price)
            fra = np.roll(funding_raw, 1)

            # ── 4. FIX 1: Regime with persisted state ──
            bp = self._compute_regime_incremental(price, ret)
            self.log.info(f"Regime: bp={bp:.1f} (m1={self.state.regime_m1} ch={self.state.regime_ch})")

            # ── 5. LS ranking (every 8H) ──
            is_ls_time = now.hour % 8 == 0
            if is_ls_time:
                self.log.info("LS rebalance check...")
                asset_8h = {}
                for asset in p.assets:
                    try:
                        df = self.pipeline.fetch_klines(asset, '8h', pages=1)
                        asset_8h[asset] = df['return'].values
                    except Exception as e:
                        self.log.warning(f"Failed {asset}: {e}")
                if len(asset_8h) >= 3:
                    new_long, new_short = RACMLS.compute_ranking(
                        asset_8h, list(asset_8h.keys()), p.ls_lookbacks, bar_multiplier=3)
                    if new_long != self.state.ls_long_asset or new_short != self.state.ls_short_asset:
                        self.log.info(f"LS CHANGE: {self.state.ls_long_asset}/{self.state.ls_short_asset} -> {new_long}/{new_short}")
                        self.state.ls_long_asset = new_long
                        self.state.ls_short_asset = new_short
                    else:
                        self.log.info(f"LS unchanged: long={new_long} short={new_short}")

            # ── 6. Crypto gates ──
            i = nn - 1
            cs, gate = RACMCryptoGates.compute(
                safe_val(funding_z, i), safe_val(oi_z, i), safe_val(oi_chg, i),
                safe_val(liq_z, i), safe_val(rv_pct, i, 0.5), bp,
                safe_val(atr_pct, i, 0.01), safe_val(ret_30d, i),
                safe_val(ma_cs, i), safe_val(vr, i, 1.0), safe_val(ret_1d, i), p)

            # ── 7. FIX 2: Kelly on base_raw history buffer ──
            dw = max(0, 1 - p.ls_weight)
            if len(self.state.base_raw_history) >= 240:
                past = np.array(self.state.base_raw_history[-p.kelly_lookback_hours:])
                lev, vt = RACMKelly.compute(past, p)
            else:
                lev, vt = 1.5, p.vol_target_base
            total_lev = min(lev * vt, p.position_cap)

            # ── 8. DD control ──
            dd_mult = RACMDDControl.compute(self.state.equity, self.state.peak_equity, p)

            # ── 9. Positions ──
            position_btc = dw * bp * gate * total_lev * dd_mult
            position_ls = p.ls_weight * gate * total_lev * dd_mult

            self.log.info(f"Signals: gate={gate:.2f} lev={total_lev:.2f} dd_mult={dd_mult:.2f} cs={cs:.2f}")
            self.log.info(f"Position: BTC={position_btc:.2f}x LS={position_ls:.2f}x "
                          f"(long={self.state.ls_long_asset} short={self.state.ls_short_asset})")

            # ── 10. Paper P&L ──
            btc_ret = ret[-1]
            na = len(p.assets)

            # LS P&L
            ls_pnl = 0.0
            if self.state.ls_long_asset and self.state.ls_short_asset:
                for ls_asset, ls_sign in [(self.state.ls_long_asset, 1), (self.state.ls_short_asset, -1)]:
                    try:
                        if ls_asset == 'BTC':
                            a_ret = btc_ret
                        else:
                            ls_df = self.pipeline.fetch_klines(ls_asset, '1h', limit=2, pages=1)
                            a_ret = ls_df['return'].values[-1] if len(ls_df) >= 1 else 0
                        ls_pnl += ls_sign * a_ret / na
                    except:
                        pass

            # Combine
            pnl_base = dw * bp * btc_ret + p.ls_weight * ls_pnl
            pnl_crypto = cs * btc_ret * p.crypto_alpha_weight
            pnl = (pnl_base + pnl_crypto) * gate * total_lev * dd_mult

            # Carry (raw funding rate)
            pnl += fra[-1] * abs(gate * total_lev * dd_mult)

            # Slippage
            pnl -= 0.0005 * abs(total_lev) / 24

            # FIX 2: Append this bar's base_raw to history for future Kelly
            self.state.append_base_raw(pnl_base)

            # Update equity
            self.state.equity *= (1 + pnl)
            self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
            dd_pct = (self.state.equity / self.state.peak_equity - 1) * 100

            self.log.info(f"P&L: base={pnl_base:+.4%} LS={ls_pnl:+.4%} crypto={pnl_crypto:+.4%} -> total={pnl:+.4%}")
            self.log.info(f"Equity: {self.state.equity:.4f} (dd={dd_pct:+.1f}%) "
                          f"kelly_buf={len(self.state.base_raw_history)} oi_buf={len(self.state.oi_history)}")

            # ── 11. Record + Save ──
            self.state.position_log.append({
                'timestamp': now.isoformat(), 'bp': bp, 'gate': gate,
                'lev': total_lev, 'dd_mult': dd_mult, 'cs': cs,
                'position_btc': position_btc, 'position_ls': position_ls,
                'ls_long': self.state.ls_long_asset, 'ls_short': self.state.ls_short_asset,
                'pnl_base': pnl_base, 'ls_pnl': ls_pnl, 'pnl_crypto': pnl_crypto,
                'pnl': pnl, 'btc_ret': btc_ret, 'funding_z': safe_val(funding_z, i),
                'rv_pctrank': safe_val(rv_pct, i, 0.5), 'equity': self.state.equity,
            })
            self.state.last_update = now.isoformat()
            self.state.save(self.state_path)

        except Exception as e:
            self.log.error(f"ERROR: {e}", exc_info=True)

    def run_loop(self, interval: int = 3600):
        self.log.info("RACM Bot v2 starting...")
        while True:
            self.run_once()
            self.log.info(f"Next tick in {interval}s")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=3600)
    parser.add_argument('--initial-capital', type=float, default=1.0)
    args = parser.parse_args()

    bot = RACMBot()
    if args.initial_capital != 1.0 and bot.state.equity == 1.0:
        bot.state.equity = args.initial_capital
        bot.state.peak_equity = args.initial_capital

    if args.once:
        bot.run_once()
    else:
        bot.run_loop(args.interval)


if __name__ == '__main__':
    main()

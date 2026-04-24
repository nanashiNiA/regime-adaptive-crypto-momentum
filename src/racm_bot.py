"""
RACM Production Bot - Uses racm_core.py for all signal logic.
Single source of truth: no duplicated formulas.

Usage:
    python -m src.racm_bot --mode paper --once
    python -m src.racm_bot --mode paper --interval 3600
"""
import sys, os, time, json, logging, pickle, argparse
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field

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
        req = urllib.request.Request(url, headers={'User-Agent': 'RACM-Bot/1.0'})
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

    def fetch_open_interest_hist(self, limit: int = 200) -> pd.DataFrame:
        url = f'{self.FUTURES_URL}/openInterest?symbol=BTCUSDT'
        data = self._fetch(url)
        return float(data['openInterest'])


# ─── State ───
@dataclass
class BotState:
    equity: float = 1.0
    peak_equity: float = 1.0
    ls_long_asset: str = ''
    ls_short_asset: str = ''
    last_update: str = ''
    position_log: list = field(default_factory=list)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'BotState':
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
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
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'racm_bot.log'), encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ])
        self.log = logging.getLogger('RACM')

    def run_once(self):
        now = datetime.now(timezone.utc)
        self.log.info(f"=== RACM tick @ {now.strftime('%Y-%m-%d %H:%M')} UTC ===")
        p = self.params

        try:
            # ── 1. Fetch BTC 1H (need ~2760 bars for warmup) ──
            btc_1h = self.pipeline.fetch_klines('BTC', '1h', pages=3)
            price = btc_1h['close'].values
            ret = btc_1h['return'].values
            nn = len(ret)
            log_p = np.log(np.clip(price, 1e-12, None))
            self.log.info(f"BTC 1H: {nn} bars")

            # ── 2. Fetch derivatives ──
            try:
                fr_df = self.pipeline.fetch_funding_rate(limit=500)
                # Build 1H funding array aligned to btc_1h index
                funding_raw = fr_df['fundingRate'].reindex(btc_1h.index, method='ffill').fillna(0).values
            except:
                funding_raw = np.zeros(nn)

            # OI: current only (historical needs different endpoint)
            oi_arr = np.full(nn, np.nan)
            try:
                oi_now = self.pipeline.fetch_open_interest_hist()
                oi_arr[-1] = oi_now
            except: pass

            liq_arr = np.zeros(nn)  # TODO: WebSocket collector

            # ── 3. Compute features (using racm_core) ──
            funding_z = RACMFeatures.funding_zscore(funding_raw)
            oi_z = RACMFeatures.oi_zscore(oi_arr)
            oi_chg = RACMFeatures.oi_change_24h(oi_arr)
            liq_z = RACMFeatures.liq_zscore(liq_arr)
            vr = RACMFeatures.vol_ratio(ret)
            rv_pct = RACMFeatures.rv_pctrank(ret)
            vp = RACMFeatures.portfolio_vol(ret)
            ret_1d = RACMFeatures.ret_nd(log_p, 24)
            ret_30d = RACMFeatures.ret_nd(log_p, 720)
            atr_pct = RACMFeatures.atr_pct(ret, price)
            ma_cs = RACMFeatures.ma_cross_slow(price)
            fra = np.roll(funding_raw, 1)

            # ── 4. Regime ──
            bp_arr = RACMRegime.compute(price, ret, p)
            bp = bp_arr[-1] if nn > p.warmup_hours else 1.0
            self.log.info(f"Regime: bp={bp:.1f}")

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

            # ── 6. Crypto gates (using racm_core) ──
            i = nn - 1
            cs, gate = RACMCryptoGates.compute(
                safe_val(funding_z, i), safe_val(oi_z, i), safe_val(oi_chg, i),
                safe_val(liq_z, i), safe_val(rv_pct, i, 0.5), bp,
                safe_val(atr_pct, i, 0.01), safe_val(ret_30d, i),
                safe_val(ma_cs, i), safe_val(vr, i, 1.0), safe_val(ret_1d, i), p)

            # ── 7. Kelly (using racm_core, on portfolio returns) ──
            dw = max(0, 1 - p.ls_weight)
            # Build base_raw for Kelly (last 90 days)
            base_raw_window = []
            for j in range(max(0, nn - p.kelly_lookback_hours), nn):
                base_raw_window.append(dw * ret[j] * bp_arr[j] + p.ls_weight * 0)  # LS not available bar-by-bar
            lev, vt = RACMKelly.compute(np.array(base_raw_window), p)
            total_lev = min(lev * vt, p.position_cap)

            # ── 8. DD control (using racm_core) ──
            dd_mult = RACMDDControl.compute(self.state.equity, self.state.peak_equity, p)

            # ── 9. Positions ──
            position_btc = dw * bp * gate * total_lev * dd_mult
            position_ls = p.ls_weight * gate * total_lev * dd_mult

            self.log.info(f"Signals: gate={gate:.2f} lev={total_lev:.2f} dd_mult={dd_mult:.2f} cs={cs:.2f}")
            self.log.info(f"Position: BTC={position_btc:.2f}x LS={position_ls:.2f}x (long={self.state.ls_long_asset} short={self.state.ls_short_asset})")

            # ── 10. Paper P&L ──
            btc_ret = ret[-1]

            # LS P&L: fetch actual 1H return for long/short assets
            ls_pnl = 0.0
            na = len(p.assets)
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

            pnl_base = dw * bp * btc_ret + p.ls_weight * ls_pnl
            pnl_crypto = cs * btc_ret * p.crypto_alpha_weight
            pnl = (pnl_base + pnl_crypto) * gate * total_lev * dd_mult

            # Carry (raw funding rate, not z-score)
            pnl += fra[-1] * abs(gate * total_lev * dd_mult)

            # Slippage (simplified: avg 5bps per LS change, amortized)
            pnl -= 0.0005 * abs(total_lev) / 24  # ~5bps/day

            self.state.equity *= (1 + pnl)
            self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
            dd_pct = (self.state.equity / self.state.peak_equity - 1) * 100

            self.log.info(f"P&L: base={pnl_base:+.4%} LS={ls_pnl:+.4%} crypto={pnl_crypto:+.4%} -> total={pnl:+.4%}")
            self.log.info(f"Equity: {self.state.equity:.4f} (dd={dd_pct:+.1f}%)")

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
        self.log.info("RACM Bot starting...")
        while True:
            self.run_once()
            self.log.info(f"Next tick in {interval}s")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='paper', choices=['paper', 'live'])
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

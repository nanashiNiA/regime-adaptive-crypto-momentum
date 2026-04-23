"""
RACM Production Bot - Regime-Adaptive Cross-Sectional Crypto Momentum
Phase 1: Paper trading with Binance data pipeline

Usage:
    # First run (builds warmup from history):
    python -m src.racm_bot --mode paper --initial-capital 1000000

    # Subsequent runs (loads state):
    python -m src.racm_bot --mode paper
"""
import sys, os, time, json, logging, pickle, argparse
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import urllib.request

if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

# ─── Config ───
@dataclass
class RACMConfig:
    # Assets
    assets: list = field(default_factory=lambda: ['BTC','ETH','SOL','XRP','DOGE','LINK'])
    slippage_bps: dict = field(default_factory=lambda: {'BTC':3,'ETH':3,'SOL':5,'XRP':5,'DOGE':5,'LINK':8})

    # LS Momentum
    ls_lookbacks_days: list = field(default_factory=lambda: [60, 90])
    ls_rebalance_hours: int = 8
    ls_weight: float = 0.80  # LW

    # Regime
    regime_ma_days: int = 110
    regime_ma_short_days: int = 20
    regime_skew_threshold: float = -0.5
    regime_dd_threshold: float = -0.12
    regime_check_hours: int = 1

    # Crypto quality gates
    funding_z_long: float = -1.5
    funding_z_short: float = 2.0
    oi_z_threshold: float = -1.5
    liq_z_threshold: float = 2.0
    grind_atr_threshold: float = 0.006
    crash_danger_count: int = 3

    # Position sizing
    kelly_fraction: float = 0.15
    position_cap: float = 3.0
    vol_target_base: float = 1.5
    kelly_lookback_days: int = 90

    # DD control
    dd_level1: float = -0.15
    dd_mult1: float = 0.7
    dd_level2: float = -0.22
    dd_mult2: float = 0.4
    dd_level3: float = -0.30
    dd_mult3: float = 0.1

    # Execution
    fee_bps: float = 0.0  # Lighter.xyz = 0
    crypto_alpha_weight: float = 0.10

    # State
    state_dir: str = 'data/bot_state'
    log_dir: str = 'logs/racm_bot'


# ─── Data Pipeline ───
class BinanceDataPipeline:
    """Fetch real-time and historical data from Binance API"""

    BASE_URL = 'https://api.binance.com/api/v3'
    FUTURES_URL = 'https://fapi.binance.com/fapi/v1'

    def __init__(self, config: RACMConfig):
        self.config = config

    def _fetch(self, url):
        req = urllib.request.Request(url, headers={'User-Agent': 'RACM-Bot/1.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())

    def fetch_klines(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV klines"""
        url = f'{self.BASE_URL}/klines?symbol={symbol}USDT&interval={interval}&limit={limit}'
        data = self._fetch(url)
        df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume',
                                          'close_time','quote_vol','trades','taker_buy_base',
                                          'taker_buy_quote','ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['return'] = df['close'].pct_change()
        return df.dropna(subset=['return'])

    def fetch_funding_rate(self, limit: int = 500) -> pd.DataFrame:
        """Fetch BTC perpetual funding rate"""
        url = f'{self.FUTURES_URL}/fundingRate?symbol=BTCUSDT&limit={limit}'
        data = self._fetch(url)
        df = pd.DataFrame(data)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = df['fundingRate'].astype(float)
        df = df.set_index('fundingTime').sort_index()
        return df

    def fetch_open_interest(self) -> float:
        """Fetch current BTC open interest"""
        url = f'{self.FUTURES_URL}/openInterest?symbol=BTCUSDT'
        data = self._fetch(url)
        return float(data['openInterest'])

    def fetch_all_1h(self, lookback_bars: int = 2700) -> pd.DataFrame:
        """Fetch BTC 1H data with multiple pages if needed"""
        pages = (lookback_bars // 1000) + 1
        all_data = []
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        for _ in range(pages):
            url = f'{self.BASE_URL}/klines?symbol=BTCUSDT&interval=1h&limit=1000&endTime={end_time}'
            data = self._fetch(url)
            if not data: break
            all_data.extend(data)
            end_time = int(data[0][0]) - 1
        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume',
                                              'close_time','quote_vol','trades','taker_buy_base',
                                              'taker_buy_quote','ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['return'] = df['close'].pct_change()
        return df.dropna(subset=['return'])

    def fetch_asset_8h(self, symbol: str, lookback_bars: int = 800) -> pd.DataFrame:
        """Fetch 8H klines for an asset"""
        pages = (lookback_bars // 1000) + 1
        all_data = []
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        for _ in range(pages):
            url = f'{self.BASE_URL}/klines?symbol={symbol}USDT&interval=8h&limit=1000&endTime={end_time}'
            data = self._fetch(url)
            if not data: break
            all_data.extend(data)
            end_time = int(data[0][0]) - 1
        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume',
                                              'close_time','quote_vol','trades','taker_buy_base',
                                              'taker_buy_quote','ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
        for col in ['close']:
            df[col] = df[col].astype(float)
        df['return'] = df['close'].pct_change()
        return df.dropna(subset=['return'])


# ─── State Management ───
@dataclass
class BotState:
    """Persistent state across restarts"""
    equity: float = 1.0
    peak_equity: float = 1.0
    regime_m1: int = 0
    regime_ch: int = 0
    last_bp: float = 1.0
    ls_long_asset: str = ''
    ls_short_asset: str = ''
    last_update: str = ''
    trade_log: list = field(default_factory=list)
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


# ─── Signal Engine ───
class RACMSignalEngine:
    """Compute all signals from data"""

    def __init__(self, config: RACMConfig):
        self.config = config

    def compute_regime(self, price: np.ndarray, ret: np.ndarray, state: BotState) -> tuple:
        """Compute regime position multiplier (bp_h) for current bar"""
        n = len(price)
        if n < self.config.regime_ma_days * 24 + 120:
            return 1.0, state.regime_m1, state.regime_ch

        ma_long_window = self.config.regime_ma_days * 24
        ma_short_window = self.config.regime_ma_short_days * 24

        ma_long = np.mean(price[max(0, n-ma_long_window):n-1])
        ma_short = np.mean(price[max(0, n-ma_short_window):n-1])
        skew_val = pd.Series(ret[max(0, n-720):n-1]).skew() if n > 720 else 0
        peak = np.max(price[max(0, n-1080):n-1])
        dd = (price[-2] - peak) / peak if peak > 0 else 0
        rv = np.std(ret[max(0, n-720):n-1]) * np.sqrt(365*24) if n > 720 else 1.0

        # MA slope (5 days)
        if n > ma_long_window + 120:
            ma_l_now = np.mean(price[max(0,n-ma_long_window):n-1])
            ma_l_old = np.mean(price[max(0,n-ma_long_window-120):n-121])
            ma_slope = (ma_l_now - ma_l_old) / ma_l_old if ma_l_old > 0 else 0
        else:
            ma_slope = 0

        c_sum = np.sum(ret[max(0, n-720):n-1])

        # Danger composite
        dl = 0; ds = 0
        if price[-2] < ma_long: dl += 1
        if price[-2] < ma_short: ds += 1
        if not np.isnan(skew_val) and skew_val < self.config.regime_skew_threshold: dl += 1; ds += 1
        if dd < self.config.regime_dd_threshold: dl += 1; ds += 1
        dc = dl * 0.3 + ds * 0.7

        m1 = state.regime_m1
        ch = state.regime_ch
        if n >= 3 and ret[-3] < -0.006: m1 = 4
        if m1 > 0: m1 -= 1

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

        return bp, m1, ch

    def compute_ls_ranking(self, asset_8h_data: dict) -> tuple:
        """Compute LS momentum ranking from 8H data"""
        config = self.config
        rankings = {}
        for lb_days in config.ls_lookbacks_days:
            lb = lb_days * 3  # 8H bars per day = 3
            scores = {}
            for name, df in asset_8h_data.items():
                if len(df) < lb + 10: continue
                ret = df['return'].values
                mom = np.sum(ret[-lb-1:-1])
                vol = np.mean(np.abs(ret[-30-1:-1])) + 1e-10
                scores[name] = mom / vol
            if len(scores) >= 3:
                sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                rankings[lb_days] = (sorted_assets[0][0], sorted_assets[-1][0])

        if not rankings:
            return '', ''
        # Use first lookback's ranking
        long_asset = list(rankings.values())[0][0]
        short_asset = list(rankings.values())[0][1]
        return long_asset, short_asset

    def compute_crypto_gates(self, funding_z: float, oi_zscore: float,
                             liq_zscore: float, rv_pctrank: float,
                             bp: float, atr_pct: float, ret_30d: float,
                             ma_cross_slow: float) -> tuple:
        """Compute crypto quality gates and alpha signal"""
        config = self.config

        # Crypto alpha
        cs = 0
        if funding_z < -2.0: cs += 0.5
        elif funding_z < config.funding_z_long: cs += 0.3
        if oi_zscore < config.oi_z_threshold and ret_30d < -0.02: cs += 0.3
        if funding_z > config.funding_z_short: cs -= 0.3
        if liq_zscore > config.liq_z_threshold: cs -= 0.3
        cs = np.clip(cs, -1, 1)

        # Gate multiplier
        n_danger = int(rv_pctrank > 0.95) + int(liq_zscore > config.liq_z_threshold) + int(bp < 0.3)
        grind = (atr_pct < config.grind_atr_threshold and ret_30d < -0.05
                 and ma_cross_slow < 0 and rv_pctrank < 0.50)
        gate = 1.0
        if n_danger >= config.crash_danger_count: gate = 0.1
        elif grind: gate = 0.5
        elif rv_pctrank > 0.50 and atr_pct > 0: gate = 0.85  # vol hedge

        return cs, gate

    def compute_kelly(self, past_returns: np.ndarray) -> tuple:
        """Compute Kelly leverage and vol target"""
        config = self.config
        if len(past_returns) < 240:
            return 1.5, config.vol_target_base

        mu = np.mean(past_returns)
        var = np.var(past_returns) + 1e-10
        lev = np.clip(mu / var * config.kelly_fraction, 1.0, config.position_cap)

        vol_p = np.std(past_returns) * np.sqrt(365 * 24)
        if vol_p > 0:
            vt = np.clip(config.vol_target_base / vol_p, 0.5, config.position_cap / lev)
        else:
            vt = 1.0

        return lev, vt


# ─── Main Bot ───
class RACMBot:
    def __init__(self, config: RACMConfig):
        self.config = config
        self.pipeline = BinanceDataPipeline(config)
        self.engine = RACMSignalEngine(config)
        self.state_path = os.path.join(config.state_dir, 'racm_state.pkl')
        self.state = BotState.load(self.state_path)

        # Setup logging
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config.log_dir, 'racm_bot.log'), encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('RACM')

    def run_once(self):
        """Run one iteration of the bot (called every hour)"""
        now = datetime.now(timezone.utc)
        self.logger.info(f"=== RACM tick @ {now.strftime('%Y-%m-%d %H:%M')} UTC ===")

        try:
            # 1. Fetch data
            self.logger.info("Fetching BTC 1H data...")
            btc_1h = self.pipeline.fetch_all_1h(lookback_bars=2700)
            price = btc_1h['close'].values
            ret = btc_1h['return'].values

            # 2. Regime detection
            bp, m1, ch = self.engine.compute_regime(price, ret, self.state)
            self.state.regime_m1 = m1
            self.state.regime_ch = ch
            self.state.last_bp = bp
            self.logger.info(f"Regime: bp={bp:.1f}")

            # 3. LS ranking (every 8H)
            is_ls_time = now.hour % self.config.ls_rebalance_hours == 0
            if is_ls_time:
                self.logger.info("LS rebalance check...")
                asset_data = {}
                for asset in self.config.assets:
                    try:
                        df = self.pipeline.fetch_asset_8h(asset, lookback_bars=300)
                        asset_data[asset] = df
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch {asset}: {e}")
                if len(asset_data) >= 3:
                    new_long, new_short = self.engine.compute_ls_ranking(asset_data)
                    if new_long != self.state.ls_long_asset or new_short != self.state.ls_short_asset:
                        old = f"{self.state.ls_long_asset}/{self.state.ls_short_asset}"
                        self.logger.info(f"LS CHANGE: {old} -> {new_long}/{new_short}")
                        self.state.ls_long_asset = new_long
                        self.state.ls_short_asset = new_short
                    else:
                        self.logger.info(f"LS unchanged: long={new_long} short={new_short}")

            # 4. Crypto quality gates
            # Simplified: use funding rate from API
            try:
                fr_df = self.pipeline.fetch_funding_rate(limit=200)
                fr_vals = fr_df['fundingRate'].values
                fr_mean = np.mean(fr_vals[-168:]) if len(fr_vals) >= 168 else np.mean(fr_vals)
                fr_std = np.std(fr_vals[-168:]) if len(fr_vals) >= 168 else np.std(fr_vals)
                funding_z = (fr_vals[-1] - fr_mean) / (fr_std + 1e-10) if len(fr_vals) > 0 else 0
            except:
                funding_z = 0

            # Simplified features for gates
            rv_pctrank = 0.5  # TODO: compute from data
            atr_pct = np.mean(np.abs(ret[-24:])) / (price[-1] + 1e-12) if len(ret) >= 24 else 0.01
            ret_30d = np.sum(ret[-720:]) if len(ret) >= 720 else 0
            ma_cross_slow = 0  # TODO: compute

            cs, gate = self.engine.compute_crypto_gates(
                funding_z, 0, 0, rv_pctrank, bp, atr_pct, ret_30d, ma_cross_slow)

            # 5. Kelly sizing
            n = len(ret)
            kelly_lb = self.config.kelly_lookback_days * 24
            base_raw = ret[max(0, n-kelly_lb):n]  # simplified
            lev, vt = self.engine.compute_kelly(base_raw)
            total_lev = min(lev * vt, self.config.position_cap)

            # 6. DD control
            dd_eq = (self.state.equity - self.state.peak_equity) / self.state.peak_equity
            dd_mult = 1.0
            if dd_eq < self.config.dd_level3: dd_mult = self.config.dd_mult3
            elif dd_eq < self.config.dd_level2: dd_mult = self.config.dd_mult2
            elif dd_eq < self.config.dd_level1: dd_mult = self.config.dd_mult1

            # 7. Final position
            final_position = gate * total_lev * dd_mult
            dw = max(0, 1 - self.config.ls_weight)
            position_btc = dw * bp * final_position
            position_ls = self.config.ls_weight * final_position

            # 8. Log
            self.logger.info(f"Position: regime_bp={bp:.2f} gate={gate:.2f} lev={total_lev:.2f} dd_mult={dd_mult:.2f}")
            self.logger.info(f"Final: BTC={position_btc:.2f}x LS={position_ls:.2f}x (long={self.state.ls_long_asset} short={self.state.ls_short_asset})")
            self.logger.info(f"Equity: {self.state.equity:.4f} (peak={self.state.peak_equity:.4f} dd={dd_eq:.2%})")
            self.logger.info(f"Crypto: funding_z={funding_z:.2f} cs={cs:.2f}")

            # 9. Paper trade P&L (using last 1H return)
            pnl = ret[-1] * position_btc  # simplified
            self.state.equity *= (1 + pnl)
            self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

            # 10. Record
            record = {
                'timestamp': now.isoformat(),
                'bp': bp, 'gate': gate, 'lev': total_lev, 'dd_mult': dd_mult,
                'position_btc': position_btc, 'position_ls': position_ls,
                'ls_long': self.state.ls_long_asset, 'ls_short': self.state.ls_short_asset,
                'funding_z': funding_z, 'pnl': pnl,
                'equity': self.state.equity, 'dd': dd_eq,
            }
            self.state.position_log.append(record)
            self.state.last_update = now.isoformat()

            # 11. Save state
            self.state.save(self.state_path)
            self.logger.info(f"State saved. P&L this bar: {pnl:+.4%}")

        except Exception as e:
            self.logger.error(f"ERROR: {e}", exc_info=True)

    def run_loop(self, interval_seconds: int = 3600):
        """Run continuously (every hour)"""
        self.logger.info("RACM Bot starting continuous loop...")
        while True:
            self.run_once()
            self.logger.info(f"Sleeping {interval_seconds}s until next tick...")
            time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description='RACM Production Bot')
    parser.add_argument('--mode', default='paper', choices=['paper', 'live'])
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--initial-capital', type=float, default=1.0)
    parser.add_argument('--interval', type=int, default=3600, help='Seconds between ticks')
    args = parser.parse_args()

    config = RACMConfig()
    bot = RACMBot(config)

    if args.initial_capital != 1.0 and bot.state.equity == 1.0:
        bot.state.equity = args.initial_capital
        bot.state.peak_equity = args.initial_capital
        bot.logger.info(f"Initial capital set to {args.initial_capital}")

    if args.once:
        bot.run_once()
    else:
        bot.run_loop(args.interval)


if __name__ == '__main__':
    main()

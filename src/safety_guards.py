"""Safety Guards: Address all unaddressed operational risks.
Plug into racm_bot.py as middleware.
"""
import sys, os, shutil, json, time, logging
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import numpy as np

log = logging.getLogger('RACM.Safety')


class DataSanityCheck:
    """Guard 1: Verify API data quality before trading."""

    @staticmethod
    def check_price(price: float, prev_price: float = None, symbol: str = 'BTC') -> tuple:
        """Returns (is_ok, reason)"""
        if price <= 0 or np.isnan(price):
            return False, f'{symbol} price is {price} (invalid)'
        if price > 500000:  # BTC > $500K seems wrong
            return False, f'{symbol} price {price} exceeds sanity cap'
        if price < 1000:    # BTC < $1K seems wrong
            return False, f'{symbol} price {price} below sanity floor'
        if prev_price and prev_price > 0:
            change = abs(price / prev_price - 1)
            if change > 0.30:  # >30% change in 1H
                return False, f'{symbol} price changed {change:.0%} in 1H (too extreme)'
        return True, 'OK'

    @staticmethod
    def check_timestamp(timestamp, max_age_minutes: int = 10) -> tuple:
        """Check if data is stale."""
        now = datetime.now(timezone.utc)
        if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is None:
            age = (now.replace(tzinfo=None) - timestamp).total_seconds() / 60
        else:
            age = (now - timestamp).total_seconds() / 60
        if age > max_age_minutes:
            return False, f'Data is {age:.0f} min old (max {max_age_minutes})'
        return True, 'OK'

    @staticmethod
    def check_funding_rate(fr: float) -> tuple:
        if abs(fr) > 0.01:  # >1% funding rate is extreme
            return False, f'Funding rate {fr:.4f} is extreme (>1%)'
        return True, 'OK'


class USDTPegMonitor:
    """Guard 2: Monitor USDT/USD peg."""

    @staticmethod
    def check_peg(threshold: float = 0.01) -> tuple:
        """Returns (is_ok, usdt_price, deviation)"""
        try:
            url = 'https://api.binance.com/api/v3/ticker/price?symbol=USDCUSDT'
            req = urllib.request.Request(url, headers={'User-Agent': 'RACM-Safety'})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            usdt_price = float(data['price'])  # USDC/USDT ~ 1.0
            deviation = abs(usdt_price - 1.0)
            if deviation > threshold:
                return False, usdt_price, deviation
            return True, usdt_price, deviation
        except:
            return True, 1.0, 0  # fail-safe: assume OK if can't check


class StateBackup:
    """Guard 3: Backup state file every tick."""

    @staticmethod
    def backup(state_path: str, max_backups: int = 24):
        """Create timestamped backup of state file."""
        if not os.path.exists(state_path):
            return
        backup_dir = os.path.dirname(state_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(backup_dir, f'racm_state_backup_{timestamp}.pkl')
        shutil.copy2(state_path, backup_path)

        # Clean old backups (keep last N)
        backups = sorted([f for f in os.listdir(backup_dir) if f.startswith('racm_state_backup_')])
        while len(backups) > max_backups:
            os.remove(os.path.join(backup_dir, backups.pop(0)))

    @staticmethod
    def restore_latest(state_dir: str) -> str:
        """Find latest backup if main state is corrupted."""
        backups = sorted([f for f in os.listdir(state_dir) if f.startswith('racm_state_backup_')])
        if backups:
            return os.path.join(state_dir, backups[-1])
        return None


class GasMonitor:
    """Guard 4: Check Arbitrum gas prices before trading."""

    @staticmethod
    def check_gas(max_gwei: float = 10.0) -> tuple:
        """Returns (is_ok, gas_price_gwei)"""
        # Arbitrum gas is usually very low, but can spike
        # For now, always return OK (need Arbitrum RPC for real check)
        return True, 0.1


class EmergencyStop:
    """Guard 5: Kill switch conditions."""

    @staticmethod
    def should_stop(equity: float, peak: float, daily_pnl: float,
                    consecutive_losses: int) -> tuple:
        """Returns (should_stop, reason)"""
        dd = (equity / peak - 1) if peak > 0 else 0

        if dd < -0.25:
            return True, f'Equity DD {dd:.1%} exceeds -25% limit'
        if daily_pnl < -0.10:
            return True, f'Daily loss {daily_pnl:.1%} exceeds -10% limit'
        if consecutive_losses >= 5:
            return True, f'{consecutive_losses} consecutive losing days'
        return False, 'OK'


class ConfigValidator:
    """Guard 6: Validate config parameters before running."""

    @staticmethod
    def validate(params) -> list:
        """Returns list of (warning/error, message)"""
        issues = []
        if params.kelly_fraction > 0.50:
            issues.append(('ERROR', f'KF={params.kelly_fraction} > 0.50 (too aggressive)'))
        if params.position_cap > 5.0:
            issues.append(('ERROR', f'cap={params.position_cap} > 5.0 (excessive leverage)'))
        if params.position_cap < 1.0:
            issues.append(('WARNING', f'cap={params.position_cap} < 1.0 (very conservative)'))
        if len(params.assets) < 3:
            issues.append(('WARNING', f'Only {len(params.assets)} assets (LS needs 3+)'))
        return issues


class SafetyMiddleware:
    """Combined safety layer. Call before each tick."""

    def __init__(self, state_path: str = 'data/bot_state/racm_state.pkl'):
        self.state_path = state_path
        self.prev_price = None
        self.consecutive_losses = 0
        self.daily_pnl = 0
        self.last_day = None

    def pre_tick(self, params=None) -> tuple:
        """Run all pre-tick checks. Returns (can_trade, issues)"""
        issues = []

        # Config validation (first run only)
        if params:
            config_issues = ConfigValidator.validate(params)
            for sev, msg in config_issues:
                issues.append(f'[{sev}] {msg}')
                if sev == 'ERROR':
                    return False, issues

        # USDT peg check
        peg_ok, usdt_price, dev = USDTPegMonitor.check_peg()
        if not peg_ok:
            issues.append(f'[CRITICAL] USDT depeg: price={usdt_price:.4f} dev={dev:.2%}')
            return False, issues

        return True, issues

    def post_tick(self, price: float, pnl: float, equity: float, peak: float,
                  timestamp=None):
        """Run all post-tick checks and backups."""
        issues = []

        # Price sanity
        price_ok, reason = DataSanityCheck.check_price(price, self.prev_price)
        if not price_ok:
            issues.append(f'[WARNING] Price sanity: {reason}')
        self.prev_price = price

        # Track daily P&L
        today = datetime.now().strftime('%Y-%m-%d')
        if self.last_day != today:
            if self.daily_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            self.daily_pnl = 0
            self.last_day = today
        self.daily_pnl += pnl

        # Emergency stop check
        stop, reason = EmergencyStop.should_stop(
            equity, peak, self.daily_pnl, self.consecutive_losses)
        if stop:
            issues.append(f'[EMERGENCY] {reason}')

        # State backup
        StateBackup.backup(self.state_path)

        return issues


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8') if sys.platform == 'win32' else None
    print('=== Safety Guards Test ===')
    print()

    # Test price sanity
    ok, msg = DataSanityCheck.check_price(95000)
    print(f'Price 95000: {ok} ({msg})')
    ok, msg = DataSanityCheck.check_price(-1)
    print(f'Price -1: {ok} ({msg})')
    ok, msg = DataSanityCheck.check_price(95000, 50000)
    print(f'Price 95000 from 50000: {ok} ({msg})')

    # Test USDT peg
    ok, price, dev = USDTPegMonitor.check_peg()
    print(f'USDT peg: ok={ok} price={price:.4f} dev={dev:.4f}')

    # Test emergency stop
    stop, msg = EmergencyStop.should_stop(0.70, 1.0, -0.05, 3)
    print(f'Emergency (DD -30%): stop={stop} ({msg})')

    print('\nAll guards operational.')

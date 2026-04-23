"""V24: 15min LS rebalance test
Fetch 15min data for 6 assets (150 API pages each ~14min total)
Compare LS @ 15min, 30min, 1H, 8H on same data
"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request,time

print("="*70)
print("  15-MINUTE LS REBALANCE TEST")
print("="*70)

# === Fetch 15min data for all 6 assets ===
def fc_deep(sym, interval, pages=150):
    """Fetch with many pages for deep history"""
    r = []; et = int(pd.Timestamp.now().timestamp()*1000)
    for p in range(pages):
        u = f'https://api.binance.com/api/v3/klines?symbol={sym}&interval={interval}&limit=1000&endTime={et}'
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent':'M'}), timeout=15) as rr:
                d = json.loads(rr.read())
        except:
            break
        if not d: break
        for k in d:
            r.append({'timestamp': pd.Timestamp(k[0], unit='ms'), 'close': float(k[4])})
        et = int(d[0][0]) - 1
        if (p+1) % 50 == 0:
            print(f"    {sym} {interval}: {p+1}/{pages} pages...")
            time.sleep(0.5)  # rate limit
    df = pd.DataFrame(r).drop_duplicates('timestamp').set_index('timestamp').sort_index()
    df['return'] = df['close'].pct_change()
    return df.dropna()

print("\n[1] Fetching 15min data (this takes ~15 min)...")
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'LINKUSDT']
data_15m = {}
for sym in SYMBOLS:
    name = sym.replace('USDT','')
    df = fc_deep(sym, '15m', 150)
    data_15m[name] = df
    print(f"  {name}: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

# Common index for all 6 assets
c_15m = data_15m['BTC'].index
for name in data_15m:
    if name != 'BTC': c_15m = c_15m.intersection(data_15m[name].index)
print(f"\n  Common 15min bars: {len(c_15m)} ({c_15m[0]} to {c_15m[-1]})")
n_days = (c_15m[-1] - c_15m[0]).days
n_years = n_days / 365.25
print(f"  Period: {n_days} days ({n_years:.1f} years)")

# Also resample to 30min, 1H, 8H from 15min data
def resample_from_15m(data_dict, c_idx, target_freq):
    resampled = {}
    for name in data_dict:
        df = data_dict[name].loc[c_idx]
        rs = df.resample(target_freq).agg({'close': 'last'}).dropna()
        rs['return'] = rs['close'].pct_change()
        rs = rs.dropna()
        resampled[name] = rs
    c = resampled['BTC'].index
    for name in resampled:
        if name != 'BTC': c = c.intersection(resampled[name].index)
    return resampled, c

# === Run LS at each frequency ===
print("\n[2] Computing LS at multiple frequencies...")

results = {}
for freq_label, freq_str, bars_per_day in [('15min','15min',96), ('30min','30min',48), ('1H','1h',24), ('4H','4h',6), ('8H','8h',3)]:
    if freq_label == '15min':
        asset_ret = {n: data_15m[n].loc[c_15m, 'return'].values for n in data_15m}
        c_idx = c_15m
    else:
        resampled, c_rs = resample_from_15m(data_15m, c_15m, freq_str)
        asset_ret = {n: resampled[n].loc[c_rs, 'return'].values for n in resampled}
        c_idx = c_rs

    na = len(asset_ret)
    n_bars = len(c_idx)
    a_vol = {n: np.roll(pd.Series(np.abs(r)).rolling(30, min_periods=10).mean().values, 1) for n, r in asset_ret.items()}

    # LS momentum (lookback in days, converted to bars)
    lp = np.zeros(n_bars)
    ls_changes = 0; prev_l = ''; prev_s = ''
    for lb_days in [60, 90]:
        lb = lb_days * bars_per_day
        w = 0.5
        am = {n: np.roll(pd.Series(r).rolling(lb, min_periods=lb//3).sum().values, 1) for n, r in asset_ret.items()}
        start = max(200, lb + 10)
        for i in range(start, n_bars):
            moms = [(am[n][i]/(a_vol[n][i]+1e-10), n, asset_ret[n][i]) for n in asset_ret if not np.isnan(am[n][i])]
            if len(moms) < 3: continue
            moms.sort(key=lambda x: x[0], reverse=True)
            lp[i] += (moms[0][2]/na - moms[-1][2]/na) * w
            if lb_days == 60:
                nl = moms[0][1]; ns = moms[-1][1]
                if i > start and (nl != prev_l or ns != prev_s) and prev_l:
                    ls_changes += 1
                prev_l = nl; prev_s = ns

    # Equity (simple, no regime/crypto for fair comparison)
    KF = 0.15; POS_CAP = 3.0; VB = 1.5
    kelly_lb = int(90 * bars_per_day)  # 90 days in bars
    eq = np.ones(n_bars); e = 1.0; pk_e = 1.0
    warmup = max(start, kelly_lb + 100)
    for i in range(warmup, n_bars):
        pnl = lp[i]
        past = lp[max(warmup, i-kelly_lb):i]
        if len(past) < 50: lev = 1.5
        else:
            mu = np.mean(past); var = np.var(past) + 1e-10
            lev = np.clip(mu/var*KF, 1.0, POS_CAP)
        pnl *= lev
        # DD control
        dd_eq = (e - pk_e)/pk_e if pk_e > 0 else 0
        if dd_eq < -0.30: pnl *= 0.1
        elif dd_eq < -0.22: pnl *= 0.4
        elif dd_eq < -0.15: pnl *= 0.7
        e *= (1+pnl); eq[i] = e; pk_e = max(pk_e, e)

    # Monthly returns (WFS)
    eq_s = pd.Series(eq, index=c_idx)
    eq_m = eq_s.resample('ME').last()
    monthly = eq_m.pct_change().dropna() * 100
    wfs = monthly.mean() * 12
    win_m = (monthly > 0).sum()
    total_m = len(monthly)
    pk_arr = np.maximum.accumulate(eq); maxdd = (eq/pk_arr - 1).min() * 100
    total_days_freq = (c_idx[-1] - c_idx[warmup]).days
    chg_per_day = ls_changes / total_days_freq if total_days_freq > 0 else 0
    slip_est = chg_per_day * 2 * 5 / 10000 * 365 * 100  # 5bps avg

    results[freq_label] = {
        'wfs': wfs, 'win': win_m, 'total': total_m, 'maxdd': maxdd,
        'changes': chg_per_day, 'slip': slip_est, 'bars': n_bars
    }
    print(f"\n  LS @ {freq_label} ({bars_per_day}x/day):")
    print(f"    WFS: {wfs:.0f}%  Win: {win_m}/{total_m}  MaxDD: {maxdd:+.1f}%")
    print(f"    LS changes: {chg_per_day:.1f}/day  Est. slip: {slip_est:.0f}%/yr")
    print(f"    NET WFS: {wfs - slip_est:.0f}%")

# Summary
print(f"\n{'='*70}")
print(f"  SUMMARY (pure LS, same data, same period)")
print(f"{'='*70}")
print(f"\n  {'Freq':<8} {'WFS':>6} {'Slip':>6} {'NET':>6} {'Win':>8} {'MaxDD':>7} {'Chg/d':>6}")
print(f"  {'-'*50}")
for freq in ['15min','30min','1H','4H','8H']:
    r = results[freq]
    net = r['wfs'] - r['slip']
    tag = ' <--best' if net == max(results[f]['wfs']-results[f]['slip'] for f in results) else ''
    print(f"  {freq:<8} {r['wfs']:>5.0f}% {r['slip']:>5.0f}% {net:>5.0f}% {r['win']:>3}/{r['total']:<3} {r['maxdd']:>+6.1f}% {r['changes']:>5.1f}{tag}")

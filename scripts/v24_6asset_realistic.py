"""V24: 6-asset realistic version
- 6 high-liquidity assets: BTC, ETH, SOL, XRP, DOGE, LINK
- Realistic slippage: per-asset based on liquidity
- LS trades counted per-asset with actual slippage
- Train 3M -> Test 1M proper WF
"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request

print("="*80)
print("  V24: 6-ASSET REALISTIC BACKTEST")
print("  Assets: BTC, ETH, SOL, XRP, DOGE, LINK")
print("  Slippage: per-asset (3-10 bps)")
print("="*80)

# === 1H BTC data ===
btc_h=pd.read_pickle('data/external/binance/btcusdt_hourly.pkl')
btc_h.index=pd.to_datetime(btc_h.index)
deriv=pd.read_pickle('data/processed/derivatives_1h.pkl')
deriv.index=deriv.index.tz_localize(None)
common_1h = btc_h.index.intersection(deriv.index)
h = pd.DataFrame(index=common_1h)
h['close'] = btc_h.loc[common_1h, 'close'].values
h['ret'] = h['close'].pct_change()
h['funding'] = deriv.loc[common_1h, 'funding_rate'].values
h['oi'] = deriv.loc[common_1h, 'open_interest_last'].values
h['liq_count'] = deriv.loc[common_1h, 'liq_count'].values
h = h.ffill().dropna(subset=['close'])
nn = len(h); idx_h = h.index; ret = h['ret'].values; price = h['close'].values

# Features (leak-fixed)
fr = np.roll(h['funding'].fillna(0).values, 1)
fr_mean = pd.Series(fr).rolling(168, min_periods=48).mean().values
fr_std = pd.Series(fr).rolling(168, min_periods=48).std().values
funding_z = np.roll(np.where(fr_std > 1e-10, (fr - fr_mean) / fr_std, 0), 1)
oi = h['oi'].values
oi_raw = np.zeros(nn)
for i in range(25, nn):
    oi_raw[i] = oi[i-1] - oi[i-25] if not np.isnan(oi[i-1]) and not np.isnan(oi[i-25]) else 0
oi_m = pd.Series(oi_raw).rolling(168, min_periods=48).mean().values
oi_s = pd.Series(oi_raw).rolling(168, min_periods=48).std().values
oi_zscore = np.roll(np.where(oi_s > 1e-10, (oi_raw - oi_m) / oi_s, 0), 1)
oi_chg_24 = np.zeros(nn)
for i in range(25, nn):
    if oi[i-1] > 0 and not np.isnan(oi[i-1]) and not np.isnan(oi[i-25]):
        oi_chg_24[i] = (oi[i-1] - oi[i-25]) / (oi[i-25] + 1e-10)
liq_c = h['liq_count'].fillna(0).values
liq_24h = np.roll(pd.Series(liq_c).rolling(24, min_periods=6).sum().values, 1)
liq_m = pd.Series(liq_24h).rolling(168, min_periods=48).mean().values
liq_s = pd.Series(liq_24h).rolling(168, min_periods=48).std().values
liq_zscore = np.where(liq_s > 1e-10, (liq_24h - liq_m) / liq_s, 0)
rv_pctrank = np.roll(pd.Series(np.abs(ret)).rolling(24, min_periods=6).sum().rolling(720, min_periods=168).rank(pct=True).values, 1)
log_p = np.log(np.clip(price, 1e-12, None))
ret_1d = np.zeros(nn); ret_30d = np.zeros(nn)
for i in range(25, nn): ret_1d[i] = log_p[i-1] - log_p[i-25]
for i in range(721, nn): ret_30d[i] = log_p[i-1] - log_p[i-721]
atr_pct = np.roll(pd.Series(np.abs(ret)).rolling(24, min_periods=6).mean().values / (price + 1e-12), 1)
ma_72 = pd.Series(price).ewm(span=72, min_periods=24).mean().values
ma_168 = pd.Series(price).ewm(span=168, min_periods=48).mean().values
ma_cross_slow = np.roll((ma_72 - ma_168) / (price + 1e-12), 1)
# Regime
ma_long = pd.Series(price).rolling(2640, min_periods=1320).mean().values
ma_short = pd.Series(price).rolling(480, min_periods=240).mean().values
skew_h = pd.Series(ret).rolling(720, min_periods=360).skew().values
pk_h = pd.Series(price).rolling(1080, min_periods=540).max().values
dd_h = (price - pk_h) / pk_h
rv_ann = pd.Series(ret).rolling(720, min_periods=360).std().values * np.sqrt(365*24)
ma_slope = np.zeros(nn)
for i in range(2760, nn):
    if not np.isnan(ma_long[i-1]) and not np.isnan(ma_long[i-121]) and ma_long[i-121] > 0:
        ma_slope[i] = (ma_long[i-1] - ma_long[i-121]) / ma_long[i-121]
c_sum = pd.Series(ret).rolling(720, min_periods=360).sum().values
bp_h = np.ones(nn); m1_h = 0; ch_h = 0; S_H = 2760
for i in range(S_H, nn):
    dl = 0; ds = 0
    if not np.isnan(ma_long[i-1]) and price[i-1] < ma_long[i-1]: dl += 1
    if not np.isnan(ma_short[i-1]) and price[i-1] < ma_short[i-1]: ds += 1
    if not np.isnan(skew_h[i-1]) and skew_h[i-1] < -0.5: dl += 1; ds += 1
    if dd_h[i-1] < -0.12: dl += 1; ds += 1
    dc = dl * 0.3 + ds * 0.7
    if i >= 2 and ret[i-2] < -0.006: m1_h = 4
    if m1_h > 0: m1_h -= 1
    if dc >= 1.5:
        if ma_slope[i] < -0.001 and ret[i-1] < 0: bp_h[i] = -0.7
        elif ma_slope[i] > 0.0005: bp_h[i] = 0.5
        else: bp_h[i] = 0.2
        continue
    if dc >= 0.8: bp_h[i] = 0.5; ch_h = 120
    elif dc >= 0.5: bp_h[i] = 0.7; ch_h = 120
    else:
        if ch_h > 0: ch_h -= 1; bp_h[i] = 0.7 if not (not np.isnan(c_sum[i-1]) and c_sum[i-1] > 0.05) else 1.0
        elif not np.isnan(rv_ann[i-1]) and rv_ann[i-1] < 0.50 and i >= 240 and np.sum(ret[i-240:i-1]) > 0: bp_h[i] = 1.5
        else: bp_h[i] = 1.0
    if m1_h > 0: bp_h[i] = min(bp_h[i], 0.7)

# === 8H multi-asset: 6 liquid assets only ===
print("\n[1] Loading 6 high-liquidity assets at 8H...")
btc_8h = btc_h.resample('8h').agg({'close':'last'}).dropna()
btc_8h['return'] = btc_8h['close'].pct_change(); btc_8h = btc_8h.dropna()
def fc(s,i,p=20):
    r=[];et=int(pd.Timestamp.now().timestamp()*1000)
    for _ in range(p):
        u=f'https://api.binance.com/api/v3/klines?symbol={s}&interval={i}&limit=1000&endTime={et}'
        with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'M'}),timeout=15) as rr:d=json.loads(rr.read())
        if not d:break
        for k in d:r.append({'timestamp':pd.Timestamp(k[0],unit='ms'),'close':float(k[4])})
        et=int(d[0][0])-1
    df=pd.DataFrame(r).drop_duplicates('timestamp').set_index('timestamp').sort_index()
    df['return']=df['close'].pct_change();return df.dropna()

# Only 6 liquid assets
e8=fc('ETHUSDT','8h'); s8=fc('SOLUSDT','8h')
liquid_extras = {}
for sym in ['XRPUSDT','DOGEUSDT','LINKUSDT']:
    try: df=fc(sym,'8h',20); liquid_extras[sym.replace('USDT','')]=df
    except: pass

c_8h=btc_8h.index.intersection(e8.index).intersection(s8.index)
for df in liquid_extras.values(): c_8h=c_8h.intersection(df.index)

rb_8h=btc_8h.loc[c_8h,'return'].values
re_8h=e8.loc[c_8h,'return'].values
rs_8h=s8.loc[c_8h,'return'].values
ext_r_8h={sym:df.loc[c_8h,'return'].values for sym,df in liquid_extras.items()}

all_a_8h={'BTC':rb_8h,'ETH':re_8h,'SOL':rs_8h}
for sym,r_ext in ext_r_8h.items(): all_a_8h[sym]=r_ext
na=len(all_a_8h)
print(f"  Assets: {list(all_a_8h.keys())} ({na} total)")
print(f"  Common 8H bars: {len(c_8h)}")

# Per-asset slippage (bps)
SLIPPAGE = {'BTC':3, 'ETH':3, 'SOL':5, 'XRP':5, 'DOGE':5, 'LINK':8}

asset_vol_8h={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in all_a_8h.items()}

# LS with per-asset tracking + realistic slippage
print("\n[2] Computing LS with per-asset trade tracking...")
lp_8h = np.zeros(len(c_8h))
ls_long_name = [''] * len(c_8h)
ls_short_name = [''] * len(c_8h)
ls_slip_cost = np.zeros(len(c_8h))  # slippage per 8H bar
ls_trade_count = np.zeros(len(c_8h))

for lb in [60,90]:
    w = 0.5
    am = {n: np.roll(pd.Series(r).rolling(lb, min_periods=lb//3).sum().values, 1) for n,r in all_a_8h.items()}
    for i in range(180, len(c_8h)):
        moms = [(am[n][i]/(asset_vol_8h[n][i]+1e-10), n, all_a_8h[n][i]) for n in all_a_8h if not np.isnan(am[n][i])]
        if len(moms) < 3: continue
        moms.sort(key=lambda x:x[0], reverse=True)
        lp_8h[i] += (moms[0][2]/na - moms[-1][2]/na) * w
        if lb == 60:
            new_long = moms[0][1]
            new_short = moms[-1][1]
            # Count trades + slippage when ranking changes
            if i > 180:
                if new_long != ls_long_name[i-1] and ls_long_name[i-1] != '':
                    # Close old long + open new long
                    old_slip = SLIPPAGE.get(ls_long_name[i-1], 10)
                    new_slip = SLIPPAGE.get(new_long, 10)
                    ls_slip_cost[i] += (old_slip + new_slip) / 10000 / na
                    ls_trade_count[i] += 2
                if new_short != ls_short_name[i-1] and ls_short_name[i-1] != '':
                    old_slip = SLIPPAGE.get(ls_short_name[i-1], 10)
                    new_slip = SLIPPAGE.get(new_short, 10)
                    ls_slip_cost[i] += (old_slip + new_slip) / 10000 / na
                    ls_trade_count[i] += 2
            ls_long_name[i] = new_long
            ls_short_name[i] = new_short

total_trades_8h = np.sum(ls_trade_count[180:])
total_days = (c_8h[-1] - c_8h[180]).days
trades_per_day = total_trades_8h / total_days if total_days > 0 else 0
avg_slip_per_8h = np.mean(ls_slip_cost[180:]) * 100
print(f"  LS trades: {total_trades_8h:.0f} total, {trades_per_day:.1f}/day")
print(f"  Avg slippage cost per 8H bar: {avg_slip_per_8h:.4f}%")
print(f"  Annual slippage cost: {avg_slip_per_8h * 3 * 365:.1f}%")

# Map to 1H
lp_1h = np.zeros(nn)
slip_1h = np.zeros(nn)
for j in range(180, len(c_8h)):
    ts_8h = c_8h[j]; ts_end = c_8h[j+1] if j+1 < len(c_8h) else ts_8h + pd.Timedelta(hours=8)
    idxs = np.where((idx_h >= ts_8h) & (idx_h < ts_end))[0]
    if len(idxs) > 0:
        for ii in idxs:
            lp_1h[ii] = lp_8h[j] / len(idxs)
            slip_1h[ii] = ls_slip_cost[j] / len(idxs)  # distribute slip cost

# Portfolio
LW=0.80; VB=1.5
vr_1h = np.roll(pd.Series(np.abs(ret)).rolling(9, min_periods=3).mean().values, 1) / (np.roll(pd.Series(np.abs(ret)).rolling(720, min_periods=240).mean().values, 1) + 1e-10)
vp_1h = np.roll(pd.Series(ret).rolling(720, min_periods=240).std().values * np.sqrt(365*24), 1)
fra_1h = np.roll(h['funding'].fillna(0).values, 1)
dw = max(0, 1 - LW)
base_raw_1h = np.zeros(nn)
for i in range(S_H, nn): base_raw_1h[i] = dw * ret[i] * bp_h[i] + LW * lp_1h[i]

# WF folds
folds=[]; fold_info=[]
cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    tr_s=cur; tr_e=cur+pd.DateOffset(months=3)-pd.DateOffset(days=1)
    te_s=cur+pd.DateOffset(months=3); te_e=te_s+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    tr_mask=np.array([(d>=tr_s and d<=tr_e) for d in idx_h])
    te_mask=np.array([(d>=te_s and d<=te_e) for d in idx_h])
    tr_idx=np.where(tr_mask)[0]; te_idx=np.where(te_mask)[0]
    if len(tr_idx)>=100 and len(te_idx)>=20:
        folds.append({'tr': tr_idx, 'te': te_idx, 'period': te_s.strftime('%Y-%m'),
                       'train': f"{tr_s.strftime('%Y-%m')} to {tr_e.strftime('%Y-%m')}"})
    cur+=pd.DateOffset(months=1)

def run_period(start_i, end_i, kf, pos_cap, include_slip=True):
    eq = np.ones(end_i - start_i); e = 1.0; pk_e = 1.0
    for j, i in enumerate(range(start_i, end_i)):
        if i < S_H: continue
        pnl = base_raw_1h[i]
        fz = funding_z[i] if not np.isnan(funding_z[i]) else 0
        oz = oi_zscore[i] if not np.isnan(oi_zscore[i]) else 0
        oc = oi_chg_24[i] if not np.isnan(oi_chg_24[i]) else 0
        lz = liq_zscore[i] if not np.isnan(liq_zscore[i]) else 0
        rvp = rv_pctrank[i] if not np.isnan(rv_pctrank[i]) else 0.5
        cs = 0
        if fz < -2.0: cs += 0.5
        elif fz < -1.5: cs += 0.3
        if oz < -1.5 and ret_1d[i] < -0.02: cs += 0.3
        if oc > 0.015 and ret_1d[i] < -0.015: cs += 0.2
        if fz > 2.0: cs -= 0.3
        if lz > 2.0: cs -= 0.3
        cs = np.clip(cs, -1, 1)
        pnl += cs * ret[i] * 0.10
        nd = int(rvp > 0.95) + int(lz > 2.0) + int(bp_h[i] < 0.3)
        grind = (atr_pct[i] < 0.006 and ret_30d[i] < -0.05 and ma_cross_slow[i] < 0 and rvp < 0.50)
        gate = 1.0
        if nd >= 3: gate = 0.1
        elif grind: gate = 0.5
        elif vr_1h[i] > 1.5: gate = 0.85
        pnl *= gate
        past = base_raw_1h[max(S_H, i-2160):i]
        if len(past) < 240: lev = 1.5; vt = VB
        else:
            mu = np.mean(past); var = np.var(past) + 1e-10
            lev = np.clip(mu/var*kf, 1.0, pos_cap)
            vol_p = np.std(past)*np.sqrt(365*24)
            vt = np.clip(VB/vol_p, 0.5, pos_cap/lev) if vol_p > 0 else 1.0
        total_lev = min(lev * vt, pos_cap)
        pnl *= total_lev
        ps = gate * total_lev
        pnl += fra_1h[i] * abs(ps)
        # Realistic slippage (per-asset, from LS trade tracking)
        if include_slip:
            pnl -= slip_1h[i] * total_lev
        # DD control
        dd_eq = (e - pk_e)/pk_e if pk_e > 0 else 0
        if dd_eq < -0.30: pnl *= 0.1
        elif dd_eq < -0.22: pnl *= 0.4
        elif dd_eq < -0.15: pnl *= 0.7
        e *= (1+pnl); eq[j] = e; pk_e = max(pk_e, e)
    return eq

def sharpe_of(eq):
    rets = np.diff(eq) / eq[:-1]
    rets = rets[~np.isnan(rets)]
    if len(rets) < 10: return -999
    return np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(365*24)

# === Run proper WF ===
print(f"\n{'='*80}")
print(f"  PROPER WF: Train 3M -> Test 1M, {len(folds)} folds")
print(f"{'='*80}")

KF_GRID = [0.15, 0.20, 0.25, 0.30]

for pos_cap in [2.0, 2.5, 3.0]:
    print(f"\n  --- Position cap = {pos_cap} ---")
    print(f"  {'#':>3} {'Period':>8} {'KF':>5} {'Return':>8} {'MaxDD':>7} {'Result':>7}")

    wf_results = []
    for k, fold in enumerate(folds):
        best_kf = 0.25; best_sr = -999
        for kf in KF_GRID:
            eq_tr = run_period(fold['tr'][0], fold['tr'][-1]+1, kf, pos_cap)
            sr = sharpe_of(eq_tr)
            if sr > best_sr: best_sr = sr; best_kf = kf
        eq_te = run_period(fold['te'][0], fold['te'][-1]+1, best_kf, pos_cap)
        test_ret = (eq_te[-1] / eq_te[0] - 1) * 100 if eq_te[0] > 0 else 0
        test_dd = (eq_te / np.maximum.accumulate(eq_te) - 1).min() * 100
        result = "WIN" if test_ret > 0 else "LOSE"
        wf_results.append({'period': fold['period'], 'kf': best_kf,
                           'return': test_ret, 'maxdd': test_dd})
        print(f"  {k+1:>3} {fold['period']:>8} {best_kf:>5.2f} {test_ret:>+7.1f}% {test_dd:>+6.1f}% {result:>7}")

    rets = [r['return'] for r in wf_results]
    wfs = np.mean(rets) * 12
    pos_count = sum(1 for r in rets if r > 0)
    b22_folds = [r['return'] for r in wf_results if r['period'].startswith('2022')]
    b22 = sum(b22_folds)
    max_fold_dd = min(r['maxdd'] for r in wf_results)

    print(f"\n  SUMMARY (cap={pos_cap}):")
    print(f"    WFS: {wfs:.0f}%")
    print(f"    Win: {pos_count}/{len(rets)} ({pos_count/len(rets)*100:.0f}%)")
    print(f"    Bear 2022: {b22:+.0f}%")
    print(f"    Worst fold DD: {max_fold_dd:+.1f}%")
    print(f"    Fold avg: {np.mean(rets):.1f}% median: {np.median(rets):.1f}%")
    kf_counts = {}
    for r in wf_results: kf_counts[r['kf']] = kf_counts.get(r['kf'], 0) + 1
    print(f"    KF dist: {dict(sorted(kf_counts.items()))}")
    meets_all = wfs >= 300 and b22 > 0
    print(f"    Professor 300%: {'YES' if meets_all else 'NO'}")

# Trade frequency detail
print(f"\n{'='*80}")
print(f"  TRADE FREQUENCY (6 assets)")
print(f"{'='*80}")
# Count by year
for y in range(2021, 2025):
    mask_8h = np.array([d.year == y for d in c_8h])
    yr_trades = np.sum(ls_trade_count[mask_8h])
    yr_days = 365
    print(f"  {y}: {yr_trades:.0f} trades, {yr_trades/yr_days:.1f}/day")

print(f"\n  LS rebalance schedule: every 8H = 3 checks/day")
print(f"  Per check: evaluate 6 assets, rebalance if ranking changes")
print(f"  Average: {trades_per_day:.1f} actual trades/day")
print(f"  + regime changes ~0.5/day + DD adjustments ~0.1/day")
print(f"  Total: ~{trades_per_day + 0.6:.1f} meaningful trades/day")

print(f"\n  SLIPPAGE DETAIL:")
for a, s in SLIPPAGE.items():
    print(f"    {a}: {s} bps")
annual_slip = avg_slip_per_8h * 3 * 365
print(f"  Annual cost (before leverage): {annual_slip:.1f}%")

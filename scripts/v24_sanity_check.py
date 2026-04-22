"""V24 Sanity check: verify leverage and PnL distribution at 1H"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request

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

# Quick features
fr = h['funding'].fillna(0).values
fr_mean = pd.Series(fr).rolling(168, min_periods=48).mean().values
fr_std = pd.Series(fr).rolling(168, min_periods=48).std().values
funding_z = np.roll(np.where(fr_std > 1e-10, (fr - fr_mean) / fr_std, 0), 1)

oi = h['oi'].values
oi_chg_24 = np.zeros(nn)
for i in range(25, nn):
    if oi[i-1] > 0 and not np.isnan(oi[i-1]) and not np.isnan(oi[i-25]):
        oi_chg_24[i] = (oi[i-1] - oi[i-25]) / (oi[i-25] + 1e-10)
oi_raw = np.zeros(nn)
for i in range(25, nn):
    oi_raw[i] = oi[i-1] - oi[i-25] if not np.isnan(oi[i-1]) and not np.isnan(oi[i-25]) else 0
oi_mean_arr = pd.Series(oi_raw).rolling(168, min_periods=48).mean().values
oi_std_arr = pd.Series(oi_raw).rolling(168, min_periods=48).std().values
oi_zscore = np.where(oi_std_arr > 1e-10, (oi_raw - oi_mean_arr) / oi_std_arr, 0)

liq_c = h['liq_count'].fillna(0).values
liq_24h = np.roll(pd.Series(liq_c).rolling(24, min_periods=6).sum().values, 1)
liq_mean_arr = pd.Series(liq_24h).rolling(168, min_periods=48).mean().values
liq_std_arr = pd.Series(liq_24h).rolling(168, min_periods=48).std().values
liq_zscore = np.where(liq_std_arr > 1e-10, (liq_24h - liq_mean_arr) / liq_std_arr, 0)

rv_pctrank = np.roll(pd.Series(np.abs(ret)).rolling(24, min_periods=6).sum().rolling(720, min_periods=168).rank(pct=True).values, 1)
ret_30d = np.zeros(nn)
log_p = np.log(np.clip(price, 1e-12, None))
for i in range(721, nn): ret_30d[i] = log_p[i-1] - log_p[i-721]
ret_1d = np.zeros(nn)
for i in range(25, nn): ret_1d[i] = log_p[i-1] - log_p[i-25]
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

# LS at 8H
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
e8=fc('ETHUSDT','8h');s8=fc('SOLUSDT','8h')
extras={}
for sym in ['AVAXUSDT','DOGEUSDT','LINKUSDT','AAVEUSDT','LTCUSDT','XRPUSDT','BCHUSDT']:
    try:df=fc(sym,'8h',20);extras[sym.replace('USDT','')]=df
    except:pass
c_8h=btc_8h.index.intersection(e8.index).intersection(s8.index)
for df in extras.values():c_8h=c_8h.intersection(df.index)
rb_8h=btc_8h.loc[c_8h,'return'].values;re_8h=e8.loc[c_8h,'return'].values;rs_8h=s8.loc[c_8h,'return'].values
ext_r_8h={sym:df.loc[c_8h,'return'].values for sym,df in extras.items()}
all_a_8h={'BTC':rb_8h,'ETH':re_8h,'SOL':rs_8h}
for sym,r_ext in ext_r_8h.items():all_a_8h[sym]=r_ext
na=len(all_a_8h)
asset_vol_8h={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in all_a_8h.items()}
lp_8h=np.zeros(len(c_8h))
for lb in [60,90]:
    w=0.5
    am={n:np.roll(pd.Series(r).rolling(lb,min_periods=lb//3).sum().values,1) for n,r in all_a_8h.items()}
    for i in range(180,len(c_8h)):
        moms=[(am[n][i]/(asset_vol_8h[n][i]+1e-10),n,all_a_8h[n][i]) for n in all_a_8h if not np.isnan(am[n][i])]
        if len(moms)<3:continue
        moms.sort(key=lambda x:x[0],reverse=True)
        lp_8h[i]+=(moms[0][2]/na-moms[-1][2]/na)*w
lp_1h = np.zeros(nn)
for j in range(180, len(c_8h)):
    ts_8h = c_8h[j]
    ts_end = c_8h[j+1] if j+1 < len(c_8h) else ts_8h + pd.Timedelta(hours=8)
    mask_1h = (idx_h >= ts_8h) & (idx_h < ts_end)
    idxs = np.where(mask_1h)[0]
    if len(idxs) > 0:
        for ii in idxs: lp_1h[ii] = lp_8h[j] / len(idxs)

CAP=2.5;LW=0.80;KF=0.25;LC=4.0;VB=1.5
vs_1h = np.roll(pd.Series(np.abs(ret)).rolling(9, min_periods=3).mean().values, 1)
vl_1h = np.roll(pd.Series(np.abs(ret)).rolling(720, min_periods=240).mean().values, 1)
vr_1h = vs_1h / (vl_1h + 1e-10)
vp_1h = np.roll(pd.Series(ret).rolling(720, min_periods=240).std().values * np.sqrt(365*24), 1)
fra_1h = np.roll(h['funding'].fillna(0).values, 1)
dw = max(0, 1 - LW)
base_raw_1h = np.zeros(nn)
for i in range(S_H, nn): base_raw_1h[i] = dw * ret[i] * bp_h[i] + LW * lp_1h[i]

print("="*80)
print("  V24 SANITY CHECK: leverage and PnL distribution at 1H")
print("="*80)

# Track detailed stats
levs = []; vts_arr = []; pnls = []; total_positions = []
eq = np.ones(nn); e = 1.0; pk_e = 1.0

for i in range(S_H, nn):
    pnl = base_raw_1h[i]
    fz = funding_z[i] if not np.isnan(funding_z[i]) else 0
    oz = oi_zscore[i] if not np.isnan(oi_zscore[i]) else 0
    oc = oi_chg_24[i] if not np.isnan(oi_chg_24[i]) else 0
    lz = liq_zscore[i] if not np.isnan(liq_zscore[i]) else 0
    rvp = rv_pctrank[i] if not np.isnan(rv_pctrank[i]) else 0.5

    # Crypto signals
    crypto_signal = 0
    if fz < -2.0: crypto_signal += 0.5
    elif fz < -1.5: crypto_signal += 0.3
    if oz < -1.5 and ret_1d[i] < -0.02: crypto_signal += 0.3
    if oc > 0.015 and ret_1d[i] < -0.015: crypto_signal += 0.2
    if fz > 2.0: crypto_signal -= 0.3
    if lz > 2.0: crypto_signal -= 0.3
    crypto_signal = np.clip(crypto_signal, -1, 1)
    pnl += crypto_signal * ret[i] * 0.10

    # Quality gates
    n_danger = 0
    if rvp > 0.95: n_danger += 1
    if lz > 2.0: n_danger += 1
    if bp_h[i] < 0.3: n_danger += 1
    grind = (atr_pct[i] < 0.006 and ret_30d[i] < -0.05 and ma_cross_slow[i] < 0 and rvp < 0.50)

    gate_mult = 1.0
    if n_danger >= 3: gate_mult = 0.1
    elif grind: gate_mult = 0.5
    elif vr_1h[i] > 1.5: gate_mult = 0.85
    pnl *= gate_mult

    # Kelly
    past = base_raw_1h[max(S_H, i-2160):i]
    if len(past) < 240: lev = 2.0; vt = VB
    else:
        mu = np.mean(past); var = np.var(past) + 1e-10
        lev = np.clip(mu/var*KF, 1.0, LC)
        vol_p = np.std(past)*np.sqrt(365*24)
        vt = np.clip(VB/vol_p, 0.5, 3.0) if vol_p > 0 else VB
    pnl *= lev
    vts_val = np.clip(vt/vp_1h[i], 0.2, CAP) if not np.isnan(vp_1h[i]) and vp_1h[i] > 0 else 1.0
    pnl *= vts_val
    ps = gate_mult * lev * vts_val
    pnl += fra_1h[i] * abs(ps)

    # DD control
    dd_eq = (e - pk_e)/pk_e if pk_e > 0 else 0
    if dd_eq < -0.30: pnl *= 0.1
    elif dd_eq < -0.22: pnl *= 0.4
    elif dd_eq < -0.15: pnl *= 0.7

    levs.append(lev)
    vts_arr.append(vts_val)
    total_positions.append(gate_mult * lev * vts_val)
    pnls.append(pnl)

    e *= (1+pnl); eq[i] = e; pk_e = max(pk_e, e)

levs = np.array(levs); vts_arr = np.array(vts_arr)
total_pos = np.array(total_positions); pnls = np.array(pnls)

print(f"\n  LEVERAGE STATS:")
print(f"    Kelly lev: mean={np.mean(levs):.2f}, median={np.median(levs):.2f}, max={np.max(levs):.2f}")
print(f"    VT scale:  mean={np.mean(vts_arr):.2f}, median={np.median(vts_arr):.2f}, max={np.max(vts_arr):.2f}")
print(f"    Total pos:  mean={np.mean(total_pos):.2f}, median={np.median(total_pos):.2f}, max={np.max(total_pos):.2f}")

print(f"\n  PNL STATS (per 1H bar):")
print(f"    mean={np.mean(pnls)*100:.4f}%, std={np.std(pnls)*100:.4f}%")
print(f"    min={np.min(pnls)*100:.3f}%, max={np.max(pnls)*100:.3f}%")
print(f"    >1%: {np.sum(pnls>0.01)} bars, >5%: {np.sum(pnls>0.05)} bars, >10%: {np.sum(pnls>0.10)} bars")
print(f"    <-1%: {np.sum(pnls<-0.01)} bars, <-5%: {np.sum(pnls<-0.05)} bars")

# Daily PnL (sum of 24 1H bars)
daily_eq = eq[::24]  # sample every 24 bars
daily_ret = np.diff(daily_eq) / daily_eq[:-1]
print(f"\n  DAILY PNL STATS:")
print(f"    mean={np.mean(daily_ret)*100:.3f}%/day, std={np.std(daily_ret)*100:.3f}%/day")
print(f"    Annualized mean: {np.mean(daily_ret)*365*100:.0f}%")
print(f"    Annualized vol: {np.std(daily_ret)*np.sqrt(365)*100:.0f}%")
print(f"    Daily Sharpe: {np.mean(daily_ret)/np.std(daily_ret)*np.sqrt(365):.2f}")

# Check base_raw_1h distribution
base_active = base_raw_1h[S_H:]
print(f"\n  BASE_RAW_1H STATS:")
print(f"    mean={np.mean(base_active)*100:.5f}%, std={np.std(base_active)*100:.4f}%")
print(f"    non-zero: {np.count_nonzero(base_active)}/{len(base_active)}")
print(f"    mean(abs): {np.mean(np.abs(base_active))*100:.5f}%")

# Check LS distribution at 1H
ls_active = lp_1h[S_H:]
print(f"\n  LS_1H STATS:")
print(f"    mean={np.mean(ls_active)*100:.5f}%, std={np.std(ls_active)*100:.5f}%")
print(f"    non-zero: {np.count_nonzero(ls_active)}/{len(ls_active)}")

# Key issue: is the leverage * VT product too high?
product = levs * vts_arr
print(f"\n  LEV * VT PRODUCT:")
print(f"    mean={np.mean(product):.2f}, median={np.median(product):.2f}, max={np.max(product):.2f}")
print(f"    >5x: {np.sum(product>5)} bars ({np.sum(product>5)/len(product)*100:.1f}%)")
print(f"    >8x: {np.sum(product>8)} bars ({np.sum(product>8)/len(product)*100:.1f}%)")
print(f"    >10x: {np.sum(product>10)} bars ({np.sum(product>10)/len(product)*100:.1f}%)")

# Verify: at 8H, Kelly gives similar leverage?
print(f"\n  COMPARISON: Kelly at 8H vs 1H")
print(f"  8H lookback=270 bars vs 1H lookback=2160 bars (both = 90 days)")
# Compute Kelly at 8H scale using 8H base_raw
base_8h_test = base_raw_1h.reshape(-1, 8).mean(axis=1) * 8 if nn % 8 == 0 else None
if base_8h_test is not None:
    past_8h = base_8h_test[-270:]
    mu_8h = np.mean(past_8h); var_8h = np.var(past_8h)
    lev_8h = np.clip(mu_8h/var_8h * KF, 1.0, LC)
    print(f"  8H: mu={mu_8h:.6f} var={var_8h:.8f} lev={lev_8h:.2f}")
past_1h = base_raw_1h[-2160:]
mu_1h = np.mean(past_1h); var_1h = np.var(past_1h)
lev_1h_test = np.clip(mu_1h/var_1h * KF, 1.0, LC)
print(f"  1H: mu={mu_1h:.6f} var={var_1h:.8f} lev={lev_1h_test:.2f}")

# Year-by-year
print(f"\n  YEAR-BY-YEAR (compound):")
for y in range(2021, 2025):
    mask = np.array([d.year == y for d in idx_h]); idx_y = np.where(mask)[0]
    if len(idx_y) > 100:
        yr = (eq[idx_y[-1]]/eq[max(1,idx_y[0]-1)]-1)*100
        avg_lev = np.mean(levs[idx_y[0]-S_H:idx_y[-1]-S_H+1]) if idx_y[0] >= S_H else 0
        avg_pos = np.mean(total_pos[idx_y[0]-S_H:idx_y[-1]-S_H+1]) if idx_y[0] >= S_H else 0
        print(f"    {y}: {yr:+.0f}% (avg_lev={avg_lev:.1f} avg_pos={avg_pos:.1f}x)")

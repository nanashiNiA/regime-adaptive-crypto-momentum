"""V24: 2025 OOS test
Fetch BTC 2025 1H from Binance API, combine with derivatives_1h (has 2025 data)
Run model on 2025 as pure out-of-sample
"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request

print("="*80)
print("  V24: 2025 OUT-OF-SAMPLE TEST")
print("="*80)

# === Fetch BTC 1H for 2025 from Binance API ===
print("\n[1] Fetching BTC 1H 2025 from Binance...")
def fetch_btc_1h(start_ts, end_ts):
    rows = []
    et = end_ts
    while et > start_ts:
        u = f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000&endTime={et}'
        with urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent':'M'}), timeout=15) as rr:
            d = json.loads(rr.read())
        if not d: break
        for k in d:
            rows.append({'timestamp': pd.Timestamp(k[0], unit='ms'),
                        'open': float(k[1]), 'high': float(k[2]),
                        'low': float(k[3]), 'close': float(k[4])})
        et = int(d[0][0]) - 1
        if et <= start_ts: break
    df = pd.DataFrame(rows).drop_duplicates('timestamp').set_index('timestamp').sort_index()
    return df

start_2025 = int(pd.Timestamp('2024-06-01').timestamp() * 1000)  # fetch from mid-2024 for warmup
end_2025 = int(pd.Timestamp.now().timestamp() * 1000)
btc_api = fetch_btc_1h(start_2025, end_2025)
print(f"  Fetched: {len(btc_api)} bars ({btc_api.index[0]} to {btc_api.index[-1]})")

# Combine with local file
btc_h_local = pd.read_pickle('data/external/binance/btcusdt_hourly.pkl')
btc_h_local.index = pd.to_datetime(btc_h_local.index)
# API data has same columns, merge
btc_combined = pd.concat([btc_h_local, btc_api[~btc_api.index.isin(btc_h_local.index)]])
btc_combined = btc_combined.sort_index()
print(f"  Combined: {len(btc_combined)} bars ({btc_combined.index[0]} to {btc_combined.index[-1]})")

# Load derivatives (has 2025 data)
deriv = pd.read_pickle('data/processed/derivatives_1h.pkl')
deriv.index = deriv.index.tz_localize(None)
print(f"  Derivatives: {len(deriv)} bars ({deriv.index[0]} to {deriv.index[-1]})")

# Common index
common_1h = btc_combined.index.intersection(deriv.index)
print(f"  Common: {len(common_1h)} bars ({common_1h[0]} to {common_1h[-1]})")

# Build features
h = pd.DataFrame(index=common_1h)
h['close'] = btc_combined.loc[common_1h, 'close'].values
h['high'] = btc_combined.loc[common_1h, 'high'].values if 'high' in btc_combined.columns else btc_combined.loc[common_1h, 'close'].values
h['low'] = btc_combined.loc[common_1h, 'low'].values if 'low' in btc_combined.columns else btc_combined.loc[common_1h, 'close'].values
h['ret'] = h['close'].pct_change()
h['funding'] = deriv.loc[common_1h, 'funding_rate'].values
h['oi'] = deriv.loc[common_1h, 'open_interest_last'].values
h['liq_count'] = deriv.loc[common_1h, 'liq_count'].values
h = h.ffill().dropna(subset=['close'])
nn = len(h); idx_h = h.index; ret = h['ret'].values; price = h['close'].values
print(f"  Final frame: {nn} bars")

# All features (same as v24)
fr = np.roll(h['funding'].fillna(0).values, 1)
fr_mean = pd.Series(fr).rolling(168, min_periods=48).mean().values
fr_std = pd.Series(fr).rolling(168, min_periods=48).std().values
funding_z = np.roll(np.where(fr_std > 1e-10, (fr - fr_mean) / fr_std, 0), 1)
oi_val = h['oi'].values
oi_raw = np.zeros(nn)
for i in range(25, nn):
    oi_raw[i] = oi_val[i-1] - oi_val[i-25] if not np.isnan(oi_val[i-1]) and not np.isnan(oi_val[i-25]) else 0
oi_m = pd.Series(oi_raw).rolling(168, min_periods=48).mean().values
oi_s = pd.Series(oi_raw).rolling(168, min_periods=48).std().values
oi_zscore = np.roll(np.where(oi_s > 1e-10, (oi_raw - oi_m) / oi_s, 0), 1)
oi_chg_24 = np.zeros(nn)
for i in range(25, nn):
    if oi_val[i-1] > 0 and not np.isnan(oi_val[i-1]) and not np.isnan(oi_val[i-25]):
        oi_chg_24[i] = (oi_val[i-1] - oi_val[i-25]) / (oi_val[i-25] + 1e-10)
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

# LS at 8H (6 assets)
btc_8h = btc_combined.resample('8h').agg({'close':'last'}).dropna()
btc_8h['return'] = btc_8h['close'].pct_change(); btc_8h = btc_8h.dropna()
def fc(s,i,p=25):
    r=[];et=int(pd.Timestamp.now().timestamp()*1000)
    for _ in range(p):
        u=f'https://api.binance.com/api/v3/klines?symbol={s}&interval={i}&limit=1000&endTime={et}'
        with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'M'}),timeout=15) as rr:d=json.loads(rr.read())
        if not d:break
        for k in d:r.append({'timestamp':pd.Timestamp(k[0],unit='ms'),'close':float(k[4])})
        et=int(d[0][0])-1
    df=pd.DataFrame(r).drop_duplicates('timestamp').set_index('timestamp').sort_index()
    df['return']=df['close'].pct_change();return df.dropna()
e8=fc('ETHUSDT','8h',25);s8=fc('SOLUSDT','8h',25)
liquid_extras={}
for sym in ['XRPUSDT','DOGEUSDT','LINKUSDT']:
    try:df=fc(sym,'8h',25);liquid_extras[sym.replace('USDT','')]=df
    except:pass
c_8h=btc_8h.index.intersection(e8.index).intersection(s8.index)
for df in liquid_extras.values():c_8h=c_8h.intersection(df.index)
all_a_8h={'BTC':btc_8h.loc[c_8h,'return'].values,'ETH':e8.loc[c_8h,'return'].values,'SOL':s8.loc[c_8h,'return'].values}
for sym,df in liquid_extras.items():all_a_8h[sym]=df.loc[c_8h,'return'].values
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
    ts_8h=c_8h[j];ts_end=c_8h[j+1] if j+1<len(c_8h) else ts_8h+pd.Timedelta(hours=8)
    idxs=np.where((idx_h>=ts_8h)&(idx_h<ts_end))[0]
    if len(idxs)>0:
        for ii in idxs:lp_1h[ii]=lp_8h[j]/len(idxs)

LW=0.80;VB=1.5;KF=0.15;POS_CAP=3.0
vr_1h=np.roll(pd.Series(np.abs(ret)).rolling(9,min_periods=3).mean().values,1)/(np.roll(pd.Series(np.abs(ret)).rolling(720,min_periods=240).mean().values,1)+1e-10)
vp_1h=np.roll(pd.Series(ret).rolling(720,min_periods=240).std().values*np.sqrt(365*24),1)
fra_1h=np.roll(h['funding'].fillna(0).values,1)
dw=max(0,1-LW)
base_raw_1h=np.zeros(nn)
for i in range(S_H,nn):base_raw_1h[i]=dw*ret[i]*bp_h[i]+LW*lp_1h[i]

# Run full model
eq=np.ones(nn);e=1.0;pk_e=1.0
for i in range(S_H,nn):
    pnl=base_raw_1h[i]
    fz=funding_z[i] if not np.isnan(funding_z[i]) else 0
    oz=oi_zscore[i] if not np.isnan(oi_zscore[i]) else 0
    oc=oi_chg_24[i] if not np.isnan(oi_chg_24[i]) else 0
    lz=liq_zscore[i] if not np.isnan(liq_zscore[i]) else 0
    rvp=rv_pctrank[i] if not np.isnan(rv_pctrank[i]) else 0.5
    cs=0
    if fz<-2.0:cs+=0.5
    elif fz<-1.5:cs+=0.3
    if oz<-1.5 and ret_1d[i]<-0.02:cs+=0.3
    if oc>0.015 and ret_1d[i]<-0.015:cs+=0.2
    if fz>2.0:cs-=0.3
    if lz>2.0:cs-=0.3
    cs=np.clip(cs,-1,1)
    pnl+=cs*ret[i]*0.10
    nd=int(rvp>0.95)+int(lz>2.0)+int(bp_h[i]<0.3)
    grind=(atr_pct[i]<0.006 and ret_30d[i]<-0.05 and ma_cross_slow[i]<0 and rvp<0.50)
    gate=1.0
    if nd>=3:gate=0.1
    elif grind:gate=0.5
    elif vr_1h[i]>1.5:gate=0.85
    pnl*=gate
    past=base_raw_1h[max(S_H,i-2160):i]
    if len(past)<240:lev=1.5;vt=VB
    else:
        mu=np.mean(past);var=np.var(past)+1e-10
        lev=np.clip(mu/var*KF,1.0,POS_CAP)
        vol_p=np.std(past)*np.sqrt(365*24)
        vt=np.clip(VB/vol_p,0.5,POS_CAP/lev) if vol_p>0 else 1.0
    total_lev=min(lev*vt,POS_CAP)
    pnl*=total_lev;ps=gate*total_lev
    pnl+=fra_1h[i]*abs(ps)
    dd_eq=(e-pk_e)/pk_e if pk_e>0 else 0
    if dd_eq<-0.30:pnl*=0.1
    elif dd_eq<-0.22:pnl*=0.4
    elif dd_eq<-0.15:pnl*=0.7
    pnl-=0.0005*abs(total_lev)/24  # 5bps avg slippage for 6 assets
    e*=(1+pnl);eq[i]=e;pk_e=max(pk_e,e)

# === Results ===
print(f"\n{'='*80}")
print(f"  2025 OOS RESULTS")
print(f"{'='*80}")

# Monthly breakdown for 2025
print(f"\n  Monthly breakdown:")
for m in range(1, 13):
    mask = np.array([d.year == 2025 and d.month == m for d in idx_h])
    idx_m = np.where(mask)[0]
    if len(idx_m) > 20:
        m_ret = (eq[idx_m[-1]] / eq[max(1, idx_m[0]-1)] - 1) * 100
        m_eq = eq[idx_m] / eq[max(1, idx_m[0]-1)]
        m_dd = (m_eq / np.maximum.accumulate(m_eq) - 1).min() * 100
        btc_ret = (price[idx_m[-1]] / price[idx_m[0]] - 1) * 100
        result = "WIN" if m_ret > 0 else "LOSE"
        print(f"    2025-{m:02d}: Model {m_ret:+7.1f}% BTC {btc_ret:+6.1f}% DD {m_dd:+5.1f}% {result}")

# 2025 full year
i25 = np.where(np.array([d.year == 2025 for d in idx_h]))[0]
if len(i25) > 100:
    y25_ret = (eq[i25[-1]] / eq[max(1, i25[0]-1)] - 1) * 100
    y25_eq = eq[i25] / eq[max(1, i25[0]-1)]
    y25_dd = (y25_eq / np.maximum.accumulate(y25_eq) - 1).min() * 100
    btc25 = (price[i25[-1]] / price[i25[0]] - 1) * 100
    print(f"\n  2025 Full: Model {y25_ret:+.0f}% vs BTC {btc25:+.0f}%")
    print(f"  2025 MaxDD: {y25_dd:+.1f}%")
    # WFS for 2025 folds
    wf25 = []
    cur = pd.Timestamp('2024-10-01')
    while cur + pd.DateOffset(months=4) <= idx_h[-1] + pd.DateOffset(days=15):
        te_s = cur + pd.DateOffset(months=3)
        te_e = te_s + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        te_m = np.array([(d >= te_s and d <= te_e) for d in idx_h])
        te_idx = np.where(te_m)[0]
        if len(te_idx) >= 20 and te_s.year >= 2025:
            f_ret = (eq[te_idx[-1]] / eq[max(0, te_idx[0]-1)] - 1) * 100
            wf25.append(f_ret)
            print(f"    OOS fold {te_s.strftime('%Y-%m')}: {f_ret:+.1f}%")
        cur += pd.DateOffset(months=1)
    if wf25:
        print(f"\n  2025 OOS WFS (mean*12): {np.mean(wf25)*12:.0f}%")
        print(f"  2025 OOS positive: {sum(1 for r in wf25 if r>0)}/{len(wf25)}")
else:
    print(f"\n  Not enough 2025 data (only {len(i25)} bars)")

# Year-by-year comparison
print(f"\n  YEAR-BY-YEAR:")
for y in range(2020, 2027):
    mask = np.array([d.year == y for d in idx_h]); idx_y = np.where(mask)[0]
    if len(idx_y) > 100:
        yr = (eq[idx_y[-1]] / eq[max(1, idx_y[0]-1)] - 1) * 100
        btc_yr = (price[idx_y[-1]] / price[idx_y[0]] - 1) * 100
        tag = " [OOS]" if y >= 2025 else ""
        print(f"    {y}: Model {yr:+.0f}% vs BTC {btc_yr:+.0f}%{tag}")

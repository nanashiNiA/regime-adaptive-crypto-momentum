"""V24: LS rebalance frequency test
Current: 8H (3x/day). Test: 4H, 2H, 1H
Question: Does more frequent LS rebalancing improve WFS?
"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request

print("="*70)
print("  LS REBALANCE FREQUENCY TEST")
print("  8H (current) vs 4H vs 2H vs 1H")
print("="*70)

# === 1H BTC + derivatives (same as always) ===
btc_h=pd.read_pickle('data/external/binance/btcusdt_hourly.pkl')
btc_h.index=pd.to_datetime(btc_h.index)
deriv=pd.read_pickle('data/processed/derivatives_1h.pkl')
deriv.index=deriv.index.tz_localize(None)
common_1h=btc_h.index.intersection(deriv.index)
h=pd.DataFrame(index=common_1h)
h['close']=btc_h.loc[common_1h,'close'].values
h['ret']=h['close'].pct_change()
h['funding']=deriv.loc[common_1h,'funding_rate'].values
h['oi']=deriv.loc[common_1h,'open_interest_last'].values
h['liq_count']=deriv.loc[common_1h,'liq_count'].values
h=h.ffill().dropna(subset=['close'])
nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values

# 1H features (abbreviated, same as v24)
fr=np.roll(h['funding'].fillna(0).values,1)
fr_mean=pd.Series(fr).rolling(168,min_periods=48).mean().values
fr_std=pd.Series(fr).rolling(168,min_periods=48).std().values
funding_z=np.roll(np.where(fr_std>1e-10,(fr-fr_mean)/fr_std,0),1)
oi_val=h['oi'].values;oi_raw=np.zeros(nn)
for i in range(25,nn):oi_raw[i]=oi_val[i-1]-oi_val[i-25] if not np.isnan(oi_val[i-1]) and not np.isnan(oi_val[i-25]) else 0
oi_m=pd.Series(oi_raw).rolling(168,min_periods=48).mean().values;oi_s=pd.Series(oi_raw).rolling(168,min_periods=48).std().values
oi_zscore=np.roll(np.where(oi_s>1e-10,(oi_raw-oi_m)/oi_s,0),1)
oi_chg_24=np.zeros(nn)
for i in range(25,nn):
    if oi_val[i-1]>0 and not np.isnan(oi_val[i-1]) and not np.isnan(oi_val[i-25]):oi_chg_24[i]=(oi_val[i-1]-oi_val[i-25])/(oi_val[i-25]+1e-10)
liq_c=h['liq_count'].fillna(0).values;liq_24h=np.roll(pd.Series(liq_c).rolling(24,min_periods=6).sum().values,1)
liq_m=pd.Series(liq_24h).rolling(168,min_periods=48).mean().values;liq_s=pd.Series(liq_24h).rolling(168,min_periods=48).std().values
liq_zscore=np.where(liq_s>1e-10,(liq_24h-liq_m)/liq_s,0)
rv_pctrank=np.roll(pd.Series(np.abs(ret)).rolling(24,min_periods=6).sum().rolling(720,min_periods=168).rank(pct=True).values,1)
log_p=np.log(np.clip(price,1e-12,None));ret_1d=np.zeros(nn);ret_30d=np.zeros(nn)
for i in range(25,nn):ret_1d[i]=log_p[i-1]-log_p[i-25]
for i in range(721,nn):ret_30d[i]=log_p[i-1]-log_p[i-721]
atr_pct=np.roll(pd.Series(np.abs(ret)).rolling(24,min_periods=6).mean().values/(price+1e-12),1)
ma_72=pd.Series(price).ewm(span=72,min_periods=24).mean().values;ma_168=pd.Series(price).ewm(span=168,min_periods=48).mean().values
ma_cross_slow=np.roll((ma_72-ma_168)/(price+1e-12),1)
ma_long=pd.Series(price).rolling(2640,min_periods=1320).mean().values;ma_short=pd.Series(price).rolling(480,min_periods=240).mean().values
skew_h=pd.Series(ret).rolling(720,min_periods=360).skew().values;pk_h=pd.Series(price).rolling(1080,min_periods=540).max().values
dd_h=(price-pk_h)/pk_h;rv_ann=pd.Series(ret).rolling(720,min_periods=360).std().values*np.sqrt(365*24)
ma_slope=np.zeros(nn)
for i in range(2760,nn):
    if not np.isnan(ma_long[i-1]) and not np.isnan(ma_long[i-121]) and ma_long[i-121]>0:ma_slope[i]=(ma_long[i-1]-ma_long[i-121])/ma_long[i-121]
c_sum=pd.Series(ret).rolling(720,min_periods=360).sum().values
bp_h=np.ones(nn);m1_h=0;ch_h=0;S_H=2760
for i in range(S_H,nn):
    dl=0;ds=0
    if not np.isnan(ma_long[i-1]) and price[i-1]<ma_long[i-1]:dl+=1
    if not np.isnan(ma_short[i-1]) and price[i-1]<ma_short[i-1]:ds+=1
    if not np.isnan(skew_h[i-1]) and skew_h[i-1]<-0.5:dl+=1;ds+=1
    if dd_h[i-1]<-0.12:dl+=1;ds+=1
    dc=dl*0.3+ds*0.7
    if i>=2 and ret[i-2]<-0.006:m1_h=4
    if m1_h>0:m1_h-=1
    if dc>=1.5:
        if ma_slope[i]<-0.001 and ret[i-1]<0:bp_h[i]=-0.7
        elif ma_slope[i]>0.0005:bp_h[i]=0.5
        else:bp_h[i]=0.2
        continue
    if dc>=0.8:bp_h[i]=0.5;ch_h=120
    elif dc>=0.5:bp_h[i]=0.7;ch_h=120
    else:
        if ch_h>0:ch_h-=1;bp_h[i]=0.7 if not(not np.isnan(c_sum[i-1]) and c_sum[i-1]>0.05) else 1.0
        elif not np.isnan(rv_ann[i-1]) and rv_ann[i-1]<0.50 and i>=240 and np.sum(ret[i-240:i-1])>0:bp_h[i]=1.5
        else:bp_h[i]=1.0
    if m1_h>0:bp_h[i]=min(bp_h[i],0.7)

# === Fetch multi-asset data at multiple frequencies ===
print("\n[1] Fetching 6 assets at 1H, 2H, 4H, 8H...")
def fc(s, interval, p=25):
    r=[];et=int(pd.Timestamp.now().timestamp()*1000)
    for _ in range(p):
        u=f'https://api.binance.com/api/v3/klines?symbol={s}&interval={interval}&limit=1000&endTime={et}'
        with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'M'}),timeout=15) as rr:d=json.loads(rr.read())
        if not d:break
        for k in d:r.append({'timestamp':pd.Timestamp(k[0],unit='ms'),'close':float(k[4])})
        et=int(d[0][0])-1
    df=pd.DataFrame(r).drop_duplicates('timestamp').set_index('timestamp').sort_index()
    df['return']=df['close'].pct_change();return df.dropna()

ASSETS = ['ETHUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','LINKUSDT']
INTERVALS = {'1h': 1, '2h': 2, '4h': 4, '8h': 8}
SLIPPAGE = {'BTC':3,'ETH':3,'SOL':5,'XRP':5,'DOGE':5,'LINK':8}

# Fetch all intervals
asset_data = {}
for interval in ['1h','2h','4h','8h']:
    asset_data[interval] = {}
    # BTC from local (resample)
    btc_rs = btc_h.resample(f'{INTERVALS[interval]}h').agg({'close':'last'}).dropna()
    btc_rs['return'] = btc_rs['close'].pct_change()
    btc_rs = btc_rs.dropna()
    asset_data[interval]['BTC'] = btc_rs
    for sym in ASSETS:
        try:
            df = fc(sym, interval, 25)
            asset_data[interval][sym.replace('USDT','')] = df
        except:
            pass
    print(f"  {interval}: {list(asset_data[interval].keys())}, BTC bars={len(btc_rs)}")

# Common features
LW=0.80;VB=1.5;KF=0.15;POS_CAP=3.0;dw=max(0,1-LW)
vr_1h=np.roll(pd.Series(np.abs(ret)).rolling(9,min_periods=3).mean().values,1)/(np.roll(pd.Series(np.abs(ret)).rolling(720,min_periods=240).mean().values,1)+1e-10)
vp_1h=np.roll(pd.Series(ret).rolling(720,min_periods=240).std().values*np.sqrt(365*24),1)
fra_1h=np.roll(h['funding'].fillna(0).values,1)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    te_s=cur+pd.DateOffset(months=3);te_e=te_s+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    te_m=np.array([(d>=te_s and d<=te_e) for d in idx_h])
    if te_m.sum()>=20:folds.append(np.where(te_m)[0])
    cur+=pd.DateOffset(months=1)

def eval_wfs(eq):
    wf_r=[]
    for idx in folds:
        if len(idx)<10:continue
        wf_r.append((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100)
    return np.mean(wf_r)*12 if wf_r else 0, wf_r

# === Test each LS frequency ===
print("\n[2] Running models at each LS frequency...")

for interval in ['8h','4h','2h','1h']:
    hours = INTERVALS[interval]
    data = asset_data[interval]
    na = len(data)

    # Common index
    c_idx = data['BTC'].index
    for a in data:
        if a != 'BTC': c_idx = c_idx.intersection(data[a].index)

    a_ret = {a: data[a].loc[c_idx,'return'].values for a in data}
    a_vol = {a: np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for a,r in a_ret.items()}

    # LS at this frequency
    # Adjust lookback: 60d/90d in days -> bars at this frequency
    bars_per_day = 24 // hours
    lb_bars = [(60 * bars_per_day // 3, 60), (90 * bars_per_day // 3, 90)]  # (bars, days)
    # Actually keep lookback in terms of bars at 8H equivalent
    # 60 days at 8H = 60*3 = 180 bars. At 4H = 60*6 = 360 bars.
    lp_freq = np.zeros(len(c_idx))
    ls_changes = 0
    prev_long = ''; prev_short = ''
    for lb_days in [60, 90]:
        lb = lb_days * bars_per_day
        w = 0.5
        am = {a: np.roll(pd.Series(r).rolling(lb, min_periods=lb//3).sum().values, 1) for a,r in a_ret.items()}
        start = max(180, lb)
        for i in range(start, len(c_idx)):
            moms = [(am[a][i]/(a_vol[a][i]+1e-10), a, a_ret[a][i]) for a in a_ret if not np.isnan(am[a][i])]
            if len(moms) < 3: continue
            moms.sort(key=lambda x: x[0], reverse=True)
            lp_freq[i] += (moms[0][2]/na - moms[-1][2]/na) * w
            if lb_days == 60:
                nl = moms[0][1]; ns = moms[-1][1]
                if i > start and (nl != prev_long or ns != prev_short) and prev_long:
                    ls_changes += 1
                prev_long = nl; prev_short = ns

    # Map to 1H
    lp_1h_freq = np.zeros(nn)
    for j in range(max(180, lb), len(c_idx)):
        ts = c_idx[j]
        ts_end = c_idx[j+1] if j+1 < len(c_idx) else ts + pd.Timedelta(hours=hours)
        idxs = np.where((idx_h >= ts) & (idx_h < ts_end))[0]
        if len(idxs) > 0:
            for ii in idxs:
                lp_1h_freq[ii] = lp_freq[j] / len(idxs)

    # Build base_raw
    base_raw = np.zeros(nn)
    for i in range(S_H, nn):
        base_raw[i] = dw * ret[i] * bp_h[i] + LW * lp_1h_freq[i]

    # Run model
    eq = np.ones(nn); e = 1.0; pk_e = 1.0
    for i in range(S_H, nn):
        pnl = base_raw[i]
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
        cs=np.clip(cs,-1,1);pnl+=cs*ret[i]*0.10
        nd=int(rvp>0.95)+int(lz>2.0)+int(bp_h[i]<0.3)
        grind=(atr_pct[i]<0.006 and ret_30d[i]<-0.05 and ma_cross_slow[i]<0 and rvp<0.50)
        gate=1.0
        if nd>=3:gate=0.1
        elif grind:gate=0.5
        elif vr_1h[i]>1.5:gate=0.85
        pnl*=gate
        past=base_raw[max(S_H,i-2160):i]
        if len(past)<240:lev=1.5;vt=VB
        else:
            mu=np.mean(past);var=np.var(past)+1e-10
            lev=np.clip(mu/var*KF,1.0,POS_CAP);vol_p=np.std(past)*np.sqrt(365*24)
            vt=np.clip(VB/vol_p,0.5,POS_CAP/lev) if vol_p>0 else 1.0
        total_lev=min(lev*vt,POS_CAP);pnl*=total_lev;ps=gate*total_lev
        pnl+=fra_1h[i]*abs(ps)
        dd_eq=(e-pk_e)/pk_e if pk_e>0 else 0
        if dd_eq<-0.30:pnl*=0.1
        elif dd_eq<-0.22:pnl*=0.4
        elif dd_eq<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk_e=max(pk_e,e)

    # Evaluate
    pk=np.maximum.accumulate(eq);maxdd=(eq/pk-1).min()*100
    wfs, wf_r = eval_wfs(eq)
    win = sum(1 for r in wf_r if r > 0)
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(1,i22[0]-1)]-1)*100 if len(i22)>60 else -100

    total_days = (c_idx[-1] - c_idx[max(180,lb)]).days
    changes_per_day = ls_changes / total_days if total_days > 0 else 0
    checks_per_day = 24 / hours
    slip_annual = changes_per_day * 2 * 5 / 10000 * 365 * 100  # rough avg 5bps

    tag = " <-- current" if interval == '8h' else ""
    print(f"\n  LS @ {interval} ({checks_per_day:.0f}x/day):{tag}")
    print(f"    WFS: {wfs:.0f}%  Win: {win}/{len(wf_r)}  MaxDD: {maxdd:+.1f}%  Bear22: {b22:+.0f}%")
    print(f"    LS changes: {changes_per_day:.1f}/day  Checks: {checks_per_day:.0f}/day")
    print(f"    Est. slippage: {slip_annual:.1f}%/year")
    print(f"    Year-by-year:")
    for y in range(2021,2025):
        yr_folds = [r for r, idx in zip(wf_r, folds) if idx_h[idx[0]].year == y or (idx_h[idx[0]].month >= 4 and idx_h[idx[0]].year == y)]
        # simpler
        mask = np.array([d.year==y for d in idx_h]);iy=np.where(mask)[0]
        if len(iy)>100:
            yr=(eq[iy[-1]]/eq[max(1,iy[0]-1)]-1)*100
            print(f"      {y}: {yr:+.0f}%")

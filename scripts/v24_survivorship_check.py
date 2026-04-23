"""V24 Survivorship bias check:
Compare 6-asset (includes SOL/DOGE) vs 4-asset (BTC/ETH/XRP/LINK only)
If 2021 returns collapse with 4 assets, survivorship bias is confirmed.
"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request

print("="*80)
print("  SURVIVORSHIP BIAS CHECK")
print("  6 assets (BTC/ETH/SOL/XRP/DOGE/LINK) vs")
print("  4 assets (BTC/ETH/XRP/LINK - established coins only)")
print("  3 assets (BTC/ETH/LINK - most conservative)")
print("="*80)

# === 1H BTC + derivatives ===
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

# All 1H features (same as v24)
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

# === Fetch 8H data for multiple asset sets ===
print("\n[1] Fetching 8H data...")
btc_8h=btc_h.resample('8h').agg({'close':'last'}).dropna()
btc_8h['return']=btc_8h['close'].pct_change();btc_8h=btc_8h.dropna()
def fc(s,i,p=20):
    r=[];et=int(pd.Timestamp.now().timestamp()*1000)
    for _ in range(p):
        u=f'https://api.binance.com/api/v3/klines?symbol={s}&interval={i}&limit=1000&endTime={et}'
        with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'M'}),timeout=15) as rr:d=json.loads(rr.read())
        if not d:break
        for k in d:r.append({'timestamp':pd.Timestamp(k[0],unit='ms'),'close':float(k[4])})
        et=int(d[0][0])-1
    df=pd.DataFrame(r).drop_duplicates('timestamp').set_index('timestamp').sort_index();df['return']=df['close'].pct_change();return df.dropna()

all_8h = {'BTC': btc_8h}
for sym in ['ETHUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','LINKUSDT']:
    try:
        df = fc(sym,'8h',20)
        all_8h[sym.replace('USDT','')] = df
        print(f"  {sym}: {len(df)} bars")
    except Exception as ex:
        print(f"  {sym}: FAILED {ex}")

# Define asset sets
SETS = {
    '6-asset (current)': ['BTC','ETH','SOL','XRP','DOGE','LINK'],
    '4-asset (no SOL/DOGE)': ['BTC','ETH','XRP','LINK'],
    '3-asset (BTC/ETH/XRP)': ['BTC','ETH','XRP'],
}

vr_1h=np.roll(pd.Series(np.abs(ret)).rolling(9,min_periods=3).mean().values,1)/(np.roll(pd.Series(np.abs(ret)).rolling(720,min_periods=240).mean().values,1)+1e-10)
vp_1h=np.roll(pd.Series(ret).rolling(720,min_periods=240).std().values*np.sqrt(365*24),1)
fra_1h=np.roll(h['funding'].fillna(0).values,1)
LW=0.80;VB=1.5;KF=0.15;POS_CAP=3.0;dw=max(0,1-LW)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    te_s=cur+pd.DateOffset(months=3);te_e=te_s+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    te_m=np.array([(d>=te_s and d<=te_e) for d in idx_h])
    if te_m.sum()>=20:folds.append((np.where(te_m)[0], te_s.strftime('%Y-%m')))
    cur+=pd.DateOffset(months=1)

for set_name, asset_list in SETS.items():
    print(f"\n{'='*60}")
    print(f"  {set_name}: {asset_list}")
    print(f"{'='*60}")

    # Common 8H index
    c_8h = btc_8h.index
    for a in asset_list:
        if a != 'BTC':
            c_8h = c_8h.intersection(all_8h[a].index)

    a_returns = {a: all_8h[a].loc[c_8h,'return'].values for a in asset_list}
    na = len(asset_list)
    a_vol = {a: np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for a,r in a_returns.items()}

    # LS momentum
    lp_8h_set = np.zeros(len(c_8h))
    for lb in [60,90]:
        w=0.5
        am={a:np.roll(pd.Series(r).rolling(lb,min_periods=lb//3).sum().values,1) for a,r in a_returns.items()}
        for i in range(180,len(c_8h)):
            moms=[(am[a][i]/(a_vol[a][i]+1e-10),a,a_returns[a][i]) for a in asset_list if not np.isnan(am[a][i])]
            if len(moms)<3:continue
            moms.sort(key=lambda x:x[0],reverse=True)
            lp_8h_set[i]+=(moms[0][2]/na-moms[-1][2]/na)*w

    # Map to 1H
    lp_1h_set=np.zeros(nn)
    for j in range(180,len(c_8h)):
        ts_8h=c_8h[j];ts_end=c_8h[j+1] if j+1<len(c_8h) else ts_8h+pd.Timedelta(hours=8)
        idxs=np.where((idx_h>=ts_8h)&(idx_h<ts_end))[0]
        if len(idxs)>0:
            for ii in idxs:lp_1h_set[ii]=lp_8h_set[j]/len(idxs)

    base_raw=np.zeros(nn)
    for i in range(S_H,nn):base_raw[i]=dw*ret[i]*bp_h[i]+LW*lp_1h_set[i]

    # Run model
    eq=np.ones(nn);e=1.0;pk_e=1.0
    for i in range(S_H,nn):
        pnl=base_raw[i]
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
    wf_r=[]
    for fidx, period in folds:
        if len(fidx)<10:continue
        wf_r.append(((eq[fidx[-1]]/eq[max(0,fidx[0]-1)]-1)*100, period))
    wfs=np.mean([r for r,_ in wf_r])*12
    win=sum(1 for r,_ in wf_r if r>0)

    print(f"\n  WFS: {wfs:.0f}%  Win: {win}/{len(wf_r)}  MaxDD: {maxdd:+.1f}%")

    # Year by year
    print(f"\n  Year-by-year:")
    for y in range(2021,2025):
        yr_folds=[(r,p) for r,p in wf_r if p.startswith(str(y))]
        if yr_folds:
            yr_sum=sum(r for r,_ in yr_folds)
            yr_win=sum(1 for r,_ in yr_folds if r>0)
            print(f"    {y}: sum={yr_sum:+.0f}%  avg={yr_sum/len(yr_folds):+.1f}%/m  win={yr_win}/{len(yr_folds)}")

    # Top/bottom folds
    sorted_folds = sorted(wf_r, key=lambda x:x[0], reverse=True)
    print(f"\n  Top 3 folds:")
    for r,p in sorted_folds[:3]:
        print(f"    {p}: {r:+.1f}%")
    print(f"  Bottom 3 folds:")
    for r,p in sorted_folds[-3:]:
        print(f"    {p}: {r:+.1f}%")

    # Outlier sensitivity
    rets_only = [r for r,_ in wf_r]
    print(f"\n  Outlier sensitivity:")
    print(f"    Full: WFS={np.mean(rets_only)*12:.0f}%")
    print(f"    Top1 removed: WFS={np.mean(sorted(rets_only)[:-1])*12:.0f}%")
    print(f"    Top3 removed: WFS={np.mean(sorted(rets_only)[:-3])*12:.0f}%")

    # t-test
    from scipy import stats as st
    t25,p25=st.ttest_1samp(rets_only,25)
    t0,p0=st.ttest_1samp(rets_only,0)
    print(f"    t-test mean>0%: p={p0/2:.4f}")
    print(f"    t-test mean>25%: p={p25/2:.4f}")

"""Robustness: Parameter sensitivity tests
1. Lookback sensitivity (30d, 60d, 90d, 120d, 180d)
2. Asset leave-one-out (remove each of 6 assets)
3. Regime parameter perturbation (MA, skew, DD)
"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request

# === Compact data loading (same as v24) ===
btc_h=pd.read_pickle('data/external/binance/btcusdt_hourly.pkl');btc_h.index=pd.to_datetime(btc_h.index)
deriv=pd.read_pickle('data/processed/derivatives_1h.pkl');deriv.index=deriv.index.tz_localize(None)
common_1h=btc_h.index.intersection(deriv.index)
h=pd.DataFrame(index=common_1h);h['close']=btc_h.loc[common_1h,'close'].values;h['ret']=h['close'].pct_change()
h['funding']=deriv.loc[common_1h,'funding_rate'].values;h['oi']=deriv.loc[common_1h,'open_interest_last'].values
h['liq_count']=deriv.loc[common_1h,'liq_count'].values;h=h.ffill().dropna(subset=['close'])
nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
# Crypto features
fr=np.roll(h['funding'].fillna(0).values,1);fr_m=pd.Series(fr).rolling(168,min_periods=48).mean().values;fr_s=pd.Series(fr).rolling(168,min_periods=48).std().values
funding_z=np.roll(np.where(fr_s>1e-10,(fr-fr_m)/fr_s,0),1)
oi_v=h['oi'].values;oi_raw=np.zeros(nn)
for i in range(25,nn):oi_raw[i]=oi_v[i-1]-oi_v[i-25] if not np.isnan(oi_v[i-1]) and not np.isnan(oi_v[i-25]) else 0
oi_m=pd.Series(oi_raw).rolling(168,min_periods=48).mean().values;oi_s=pd.Series(oi_raw).rolling(168,min_periods=48).std().values
oi_zscore=np.roll(np.where(oi_s>1e-10,(oi_raw-oi_m)/oi_s,0),1)
oi_chg_24=np.zeros(nn)
for i in range(25,nn):
    if oi_v[i-1]>0 and not np.isnan(oi_v[i-1]) and not np.isnan(oi_v[i-25]):oi_chg_24[i]=(oi_v[i-1]-oi_v[i-25])/(oi_v[i-25]+1e-10)
liq_c=h['liq_count'].fillna(0).values;liq_24h=np.roll(pd.Series(liq_c).rolling(24,min_periods=6).sum().values,1)
liq_m2=pd.Series(liq_24h).rolling(168,min_periods=48).mean().values;liq_s2=pd.Series(liq_24h).rolling(168,min_periods=48).std().values
liq_zscore=np.where(liq_s2>1e-10,(liq_24h-liq_m2)/liq_s2,0)
rv_pctrank=np.roll(pd.Series(np.abs(ret)).rolling(24,min_periods=6).sum().rolling(720,min_periods=168).rank(pct=True).values,1)
log_p=np.log(np.clip(price,1e-12,None));ret_1d=np.zeros(nn);ret_30d=np.zeros(nn)
for i in range(25,nn):ret_1d[i]=log_p[i-1]-log_p[i-25]
for i in range(721,nn):ret_30d[i]=log_p[i-1]-log_p[i-721]
atr_pct=np.roll(pd.Series(np.abs(ret)).rolling(24,min_periods=6).mean().values/(price+1e-12),1)
ma_72e=pd.Series(price).ewm(span=72,min_periods=24).mean().values;ma_168e=pd.Series(price).ewm(span=168,min_periods=48).mean().values
ma_cross_slow=np.roll((ma_72e-ma_168e)/(price+1e-12),1)
vr_1h=np.roll(pd.Series(np.abs(ret)).rolling(9,min_periods=3).mean().values,1)/(np.roll(pd.Series(np.abs(ret)).rolling(720,min_periods=240).mean().values,1)+1e-10)
vp_1h=np.roll(pd.Series(ret).rolling(720,min_periods=240).std().values*np.sqrt(365*24),1)
fra_1h=np.roll(h['funding'].fillna(0).values,1)
# 8H multi-asset
btc_8h=btc_h.resample('8h').agg({'close':'last'}).dropna();btc_8h['return']=btc_8h['close'].pct_change();btc_8h=btc_8h.dropna()
def fc(s,i,p=20):
    r=[];et=int(pd.Timestamp.now().timestamp()*1000)
    for _ in range(p):
        u=f'https://api.binance.com/api/v3/klines?symbol={s}&interval={i}&limit=1000&endTime={et}'
        with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'M'}),timeout=15) as rr:d=json.loads(rr.read())
        if not d:break
        for k in d:r.append({'timestamp':pd.Timestamp(k[0],unit='ms'),'close':float(k[4])})
        et=int(d[0][0])-1
    df=pd.DataFrame(r).drop_duplicates('timestamp').set_index('timestamp').sort_index();df['return']=df['close'].pct_change();return df.dropna()
all_8h_data={}
for sym in ['ETHUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','LINKUSDT']:
    try:all_8h_data[sym.replace('USDT','')]=fc(sym,'8h',20)
    except:pass
c_8h_all=btc_8h.index
for df in all_8h_data.values():c_8h_all=c_8h_all.intersection(df.index)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    te_s=cur+pd.DateOffset(months=3);te_e=te_s+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    te_m=np.array([(d>=te_s and d<=te_e) for d in idx_h])
    if te_m.sum()>=20:folds.append(np.where(te_m)[0])
    cur+=pd.DateOffset(months=1)

def build_regime(ma_days, skew_th, dd_th):
    ma_l=pd.Series(price).rolling(ma_days*24,min_periods=ma_days*12).mean().values
    ma_s=pd.Series(price).rolling(480,min_periods=240).mean().values
    skew=pd.Series(ret).rolling(720,min_periods=360).skew().values
    pk=pd.Series(price).rolling(1080,min_periods=540).max().values;dd=(price-pk)/pk
    rv=pd.Series(ret).rolling(720,min_periods=360).std().values*np.sqrt(365*24)
    ms=np.zeros(nn);cs=pd.Series(ret).rolling(720,min_periods=360).sum().values
    S=max(2760,ma_days*24+120)
    for i in range(ma_days*24+120,nn):
        if not np.isnan(ma_l[i-1]) and not np.isnan(ma_l[i-121]) and ma_l[i-121]>0:ms[i]=(ma_l[i-1]-ma_l[i-121])/ma_l[i-121]
    bp=np.ones(nn);m1=0;ch=0
    for i in range(S,nn):
        dl=0;ds=0
        if not np.isnan(ma_l[i-1]) and price[i-1]<ma_l[i-1]:dl+=1
        if not np.isnan(ma_s[i-1]) and price[i-1]<ma_s[i-1]:ds+=1
        if not np.isnan(skew[i-1]) and skew[i-1]<skew_th:dl+=1;ds+=1
        if dd[i-1]<dd_th:dl+=1;ds+=1
        dc=dl*0.3+ds*0.7
        if i>=2 and ret[i-2]<-0.006:m1=4
        if m1>0:m1-=1
        if dc>=1.5:
            if ms[i]<-0.001 and ret[i-1]<0:bp[i]=-0.7
            elif ms[i]>0.0005:bp[i]=0.5
            else:bp[i]=0.2
            continue
        if dc>=0.8:bp[i]=0.5;ch=120
        elif dc>=0.5:bp[i]=0.7;ch=120
        else:
            if ch>0:ch-=1;bp[i]=0.7 if not(not np.isnan(cs[i-1]) and cs[i-1]>0.05) else 1.0
            elif not np.isnan(rv[i-1]) and rv[i-1]<0.50 and i>=240 and np.sum(ret[i-240:i-1])>0:bp[i]=1.5
            else:bp[i]=1.0
        if m1>0:bp[i]=min(bp[i],0.7)
    return bp, S

def build_ls(asset_list, lookbacks):
    c8=c_8h_all
    a_r={'BTC':btc_8h.loc[c8,'return'].values}
    for a in asset_list:
        if a!='BTC' and a in all_8h_data:
            c8=c8.intersection(all_8h_data[a].index)
    a_r={'BTC':btc_8h.loc[c8,'return'].values}
    for a in asset_list:
        if a!='BTC':a_r[a]=all_8h_data[a].loc[c8,'return'].values
    na=len(a_r);av={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in a_r.items()}
    lp8=np.zeros(len(c8))
    for lb in lookbacks:
        w=1.0/len(lookbacks)
        am={n:np.roll(pd.Series(r).rolling(lb,min_periods=lb//3).sum().values,1) for n,r in a_r.items()}
        for i in range(max(lookbacks)+10,len(c8)):
            moms=[(am[n][i]/(av[n][i]+1e-10),n,a_r[n][i]) for n in a_r if not np.isnan(am[n][i])]
            if len(moms)<3:continue
            moms.sort(key=lambda x:x[0],reverse=True)
            lp8[i]+=(moms[0][2]/na-moms[-1][2]/na)*w
    lp1=np.zeros(nn)
    for j in range(max(lookbacks)+10,len(c8)):
        ts=c8[j];te=c8[j+1] if j+1<len(c8) else ts+pd.Timedelta(hours=8)
        idxs=np.where((idx_h>=ts)&(idx_h<te))[0]
        if len(idxs)>0:
            for ii in idxs:lp1[ii]=lp8[j]/len(idxs)
    return lp1

def run_model(bp, S, lp1, LW=0.80, KF=0.15, CAP=3.0):
    dw=max(0,1-LW);base=np.zeros(nn)
    for i in range(S,nn):base[i]=dw*ret[i]*bp[i]+LW*lp1[i]
    eq=np.ones(nn);e=1.0;pk_e=1.0
    for i in range(S,nn):
        pnl=base[i]
        fz=funding_z[i] if not np.isnan(funding_z[i]) else 0
        oz=oi_zscore[i] if not np.isnan(oi_zscore[i]) else 0
        lz=liq_zscore[i] if not np.isnan(liq_zscore[i]) else 0
        rvp=rv_pctrank[i] if not np.isnan(rv_pctrank[i]) else 0.5
        cs=0
        if fz<-2.0:cs+=0.5
        elif fz<-1.5:cs+=0.3
        if oz<-1.5 and ret_1d[i]<-0.02:cs+=0.3
        if fz>2.0:cs-=0.3;
        if lz>2.0:cs-=0.3
        cs=np.clip(cs,-1,1);pnl+=cs*ret[i]*0.10
        nd=int(rvp>0.95)+int(lz>2.0)+int(bp[i]<0.3)
        grind=(atr_pct[i]<0.006 and ret_30d[i]<-0.05 and ma_cross_slow[i]<0 and rvp<0.50)
        gate=1.0
        if nd>=3:gate=0.1
        elif grind:gate=0.5
        elif vr_1h[i]>1.5:gate=0.85
        pnl*=gate
        past=base[max(S,i-2160):i]
        if len(past)<240:lev=1.5;vt=1.5
        else:
            mu=np.mean(past);var=np.var(past)+1e-10
            lev=np.clip(mu/var*KF,1.0,CAP);vol_p=np.std(past)*np.sqrt(365*24)
            vt=np.clip(1.5/vol_p,0.5,CAP/lev) if vol_p>0 else 1.0
        tl=min(lev*vt,CAP);pnl*=tl;pnl+=fra_1h[i]*abs(gate*tl)
        dd_eq=(e-pk_e)/pk_e if pk_e>0 else 0
        if dd_eq<-0.30:pnl*=0.1
        elif dd_eq<-0.22:pnl*=0.4
        elif dd_eq<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk_e=max(pk_e,e)
    wf_r=[]
    for idx in folds:
        if len(idx)<10:continue
        wf_r.append((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100)
    wfs=np.mean(wf_r)*12 if wf_r else 0
    win=sum(1 for r in wf_r if r>0)
    return wfs, win, len(wf_r)

print("="*70)
print("  [1] LOOKBACK SENSITIVITY")
print("="*70)
bp_default, S_def = build_regime(110, -0.5, -0.12)
ASSETS_6=['BTC','ETH','SOL','XRP','DOGE','LINK']
for lbs, label in [([30],[30]), ([45],[45]), ([60],[60]), ([90],[90]), ([120],[120]), ([180],[180]),
                    ([60,90],'60+90 (current)'), ([30,60,90],'30+60+90'), ([60,120],'60+120')]:
    lp1 = build_ls(ASSETS_6, lbs if isinstance(lbs,list) else [lbs])
    wfs, win, nf = run_model(bp_default, S_def, lp1)
    tag = ' <--' if label == '60+90 (current)' else ''
    print(f"  lb={label}: WFS={wfs:.0f}% Win={win}/{nf}{tag}")

print(f"\n{'='*70}")
print(f"  [2] ASSET LEAVE-ONE-OUT")
print(f"{'='*70}")
lp1_full = build_ls(ASSETS_6, [60,90])
wfs_full, win_full, nf = run_model(bp_default, S_def, lp1_full)
print(f"  All 6: WFS={wfs_full:.0f}% Win={win_full}/{nf}")
for remove in ASSETS_6:
    subset = [a for a in ASSETS_6 if a != remove]
    lp1_sub = build_ls(subset, [60,90])
    wfs_sub, win_sub, _ = run_model(bp_default, S_def, lp1_sub)
    diff = wfs_sub - wfs_full
    print(f"  Remove {remove:>5}: WFS={wfs_sub:.0f}% ({diff:+.0f}%) Win={win_sub}/{nf}")

print(f"\n{'='*70}")
print(f"  [3] REGIME PARAMETER PERTURBATION")
print(f"{'='*70}")
lp1_def = build_ls(ASSETS_6, [60,90])
for ma, sk, dd, label in [
    (110,-0.5,-0.12,'CURRENT'),
    (80,-0.5,-0.12,'MA80'), (90,-0.5,-0.12,'MA90'), (130,-0.5,-0.12,'MA130'), (150,-0.5,-0.12,'MA150'), (200,-0.5,-0.12,'MA200'),
    (110,-0.3,-0.12,'skew-0.3'), (110,-0.7,-0.12,'skew-0.7'), (110,-1.0,-0.12,'skew-1.0'),
    (110,-0.5,-0.08,'DD-8%'), (110,-0.5,-0.10,'DD-10%'), (110,-0.5,-0.15,'DD-15%'), (110,-0.5,-0.20,'DD-20%'),
]:
    bp_test, S_test = build_regime(ma, sk, dd)
    wfs_test, win_test, nf = run_model(bp_test, S_test, lp1_def)
    tag = ' <--' if label == 'CURRENT' else ''
    print(f"  {label:<12}: WFS={wfs_test:.0f}% Win={win_test}/{nf}{tag}")

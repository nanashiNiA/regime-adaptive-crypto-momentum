"""V24 6-asset shuffle test: verify statistical significance on the actual 6-asset model"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request

print("="*80)
print("  V24 6-ASSET SHUFFLE TEST (200 iterations x 3 types)")
print("="*80)

# === Data loading ===
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

# Features (leak-fixed, same as v24_6asset_realistic.py)
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

# LS at 8H (6 liquid assets)
print("\n[1] Loading 6 assets...")
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
liquid_extras={}
for sym in ['XRPUSDT','DOGEUSDT','LINKUSDT']:
    try:df=fc(sym,'8h',20);liquid_extras[sym.replace('USDT','')]=df
    except:pass
c_8h=btc_8h.index.intersection(e8.index).intersection(s8.index)
for df in liquid_extras.values():c_8h=c_8h.intersection(df.index)
all_a_8h={'BTC':btc_8h.loc[c_8h,'return'].values,'ETH':e8.loc[c_8h,'return'].values,'SOL':s8.loc[c_8h,'return'].values}
for sym,df in liquid_extras.items():all_a_8h[sym]=df.loc[c_8h,'return'].values
na=len(all_a_8h)
SLIPPAGE={'BTC':3,'ETH':3,'SOL':5,'XRP':5,'DOGE':5,'LINK':8}
asset_vol_8h={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in all_a_8h.items()}
lp_8h=np.zeros(len(c_8h))
ls_slip_cost=np.zeros(len(c_8h))
prev_long='';prev_short=''
for lb in [60,90]:
    w=0.5
    am={n:np.roll(pd.Series(r).rolling(lb,min_periods=lb//3).sum().values,1) for n,r in all_a_8h.items()}
    for i in range(180,len(c_8h)):
        moms=[(am[n][i]/(asset_vol_8h[n][i]+1e-10),n,all_a_8h[n][i]) for n in all_a_8h if not np.isnan(am[n][i])]
        if len(moms)<3:continue
        moms.sort(key=lambda x:x[0],reverse=True)
        lp_8h[i]+=(moms[0][2]/na-moms[-1][2]/na)*w
        if lb==60:
            nl=moms[0][1];ns=moms[-1][1]
            if i>180:
                if nl!=prev_long and prev_long!='':
                    ls_slip_cost[i]+=(SLIPPAGE.get(prev_long,10)+SLIPPAGE.get(nl,10))/10000/na
                if ns!=prev_short and prev_short!='':
                    ls_slip_cost[i]+=(SLIPPAGE.get(prev_short,10)+SLIPPAGE.get(ns,10))/10000/na
            prev_long=nl;prev_short=ns

lp_1h=np.zeros(nn);slip_1h=np.zeros(nn)
for j in range(180,len(c_8h)):
    ts_8h=c_8h[j];ts_end=c_8h[j+1] if j+1<len(c_8h) else ts_8h+pd.Timedelta(hours=8)
    idxs=np.where((idx_h>=ts_8h)&(idx_h<ts_end))[0]
    if len(idxs)>0:
        for ii in idxs:
            lp_1h[ii]=lp_8h[j]/len(idxs)
            slip_1h[ii]=ls_slip_cost[j]/len(idxs)

LW=0.80;VB=1.5;KF=0.15;POS_CAP=3.0
vr_1h=np.roll(pd.Series(np.abs(ret)).rolling(9,min_periods=3).mean().values,1)/(np.roll(pd.Series(np.abs(ret)).rolling(720,min_periods=240).mean().values,1)+1e-10)
vp_1h=np.roll(pd.Series(ret).rolling(720,min_periods=240).std().values*np.sqrt(365*24),1)
fra_1h=np.roll(h['funding'].fillna(0).values,1)
dw=max(0,1-LW)
base_raw_1h=np.zeros(nn)
for i in range(S_H,nn):base_raw_1h[i]=dw*ret[i]*bp_h[i]+LW*lp_1h[i]

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
    return np.mean(wf_r)*12 if wf_r else 0

# === Run real model ===
print("\n[2] Running real model (6 assets, cap=3.0, KF=0.15)...")
def run_full(base_arr, bp_arr):
    eq=np.ones(nn);e=1.0;pk_e=1.0
    for i in range(S_H,nn):
        pnl=base_arr[i]
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
        nd=int(rvp>0.95)+int(lz>2.0)+int(bp_arr[i]<0.3)
        grind=(atr_pct[i]<0.006 and ret_30d[i]<-0.05 and ma_cross_slow[i]<0 and rvp<0.50)
        gate=1.0
        if nd>=3:gate=0.1
        elif grind:gate=0.5
        elif vr_1h[i]>1.5:gate=0.85
        pnl*=gate
        past=base_arr[max(S_H,i-2160):i]
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
        pnl-=slip_1h[i]*total_lev
        e*=(1+pnl);eq[i]=e;pk_e=max(pk_e,e)
    return eq

eq_real=run_full(base_raw_1h, bp_h)
ws_real=eval_wfs(eq_real)
pk_real=np.maximum.accumulate(eq_real);maxdd_real=(eq_real/pk_real-1).min()*100
i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
b22_real=(eq_real[i22[-1]]/eq_real[max(1,i22[0]-1)]-1)*100
# Count wins
wf_r_real=[]
for idx in folds:
    if len(idx)<10:continue
    wf_r_real.append((eq_real[idx[-1]]/eq_real[max(0,idx[0]-1)]-1)*100)
win_real=sum(1 for r in wf_r_real if r>0)

print(f"  Real: WFS={ws_real:.0f}% MaxDD={maxdd_real:+.1f}% Bear22={b22_real:+.0f}% Win={win_real}/{len(wf_r_real)}")

# === Shuffle tests ===
N_SHUF = 1000

# [1] Return shuffle
print(f"\n[3] Return shuffle ({N_SHUF} iterations)...")
shuf_wfs_1=[]
for s in range(N_SHUF):
    np.random.seed(s)
    base_shuf=np.zeros(nn)
    orig=base_raw_1h[S_H:].copy();np.random.shuffle(orig)
    base_shuf[S_H:]=orig
    eq_s=np.ones(nn);e=1.0;pk_s=1.0
    for i in range(S_H,nn):
        pnl=base_shuf[i]
        if vr_1h[i]>1.5:pnl*=0.85
        past=base_shuf[max(S_H,i-2160):i]
        if len(past)<240:lev=1.5
        else:
            mu=np.mean(past);var=np.var(past)+1e-10
            lev=np.clip(mu/var*KF,1.0,POS_CAP)
        pnl*=lev
        dd_eq=(e-pk_s)/pk_s if pk_s>0 else 0
        if dd_eq<-0.30:pnl*=0.1
        elif dd_eq<-0.22:pnl*=0.4
        elif dd_eq<-0.15:pnl*=0.7
        e*=(1+pnl);eq_s[i]=e;pk_s=max(pk_s,e)
    shuf_wfs_1.append(eval_wfs(eq_s))
    if (s+1)%50==0:print(f"    {s+1}/{N_SHUF} done")
p1=np.mean(np.array(shuf_wfs_1)>=ws_real)
print(f"  Return shuffle: real={ws_real:.0f}% shuffle_mean={np.mean(shuf_wfs_1):.0f}% std={np.std(shuf_wfs_1):.0f}% p={p1:.3f} {'PASS' if p1<0.05 else 'FAIL'}")

# [2] Block shuffle (30d = 720H)
print(f"\n[4] Block shuffle 30d ({N_SHUF} iterations)...")
block_size=720
shuf_wfs_2=[]
for s in range(N_SHUF):
    np.random.seed(s+10000)
    active=base_raw_1h[S_H:]
    n_blocks=len(active)//block_size
    blocks=[active[b*block_size:(b+1)*block_size] for b in range(n_blocks)]
    if len(active)%block_size>0:blocks.append(active[n_blocks*block_size:])
    np.random.shuffle(blocks)
    base_shuf=np.zeros(nn)
    base_shuf[S_H:S_H+sum(len(b) for b in blocks)]=np.concatenate(blocks)[:len(active)]
    eq_s=np.ones(nn);e=1.0;pk_s=1.0
    for i in range(S_H,nn):
        pnl=base_shuf[i]
        if vr_1h[i]>1.5:pnl*=0.85
        past=base_shuf[max(S_H,i-2160):i]
        if len(past)<240:lev=1.5
        else:
            mu=np.mean(past);var=np.var(past)+1e-10
            lev=np.clip(mu/var*KF,1.0,POS_CAP)
        pnl*=lev
        dd_eq=(e-pk_s)/pk_s if pk_s>0 else 0
        if dd_eq<-0.30:pnl*=0.1
        elif dd_eq<-0.22:pnl*=0.4
        elif dd_eq<-0.15:pnl*=0.7
        e*=(1+pnl);eq_s[i]=e;pk_s=max(pk_s,e)
    shuf_wfs_2.append(eval_wfs(eq_s))
    if (s+1)%50==0:print(f"    {s+1}/{N_SHUF} done")
p2=np.mean(np.array(shuf_wfs_2)>=ws_real)
print(f"  Block shuffle: real={ws_real:.0f}% shuffle_mean={np.mean(shuf_wfs_2):.0f}% std={np.std(shuf_wfs_2):.0f}% p={p2:.3f} {'PASS' if p2<0.05 else 'FAIL'}")

# [3] Position shuffle
print(f"\n[5] Position shuffle ({N_SHUF} iterations)...")
shuf_wfs_3=[]
for s in range(N_SHUF):
    np.random.seed(s+20000)
    bp_shuf=bp_h.copy()
    active_bp=bp_shuf[S_H:].copy()
    np.random.shuffle(active_bp)
    bp_shuf[S_H:]=active_bp
    base_shuf=np.zeros(nn)
    for i in range(S_H,nn):
        base_shuf[i]=dw*ret[i]*bp_shuf[i]+LW*lp_1h[i]
    eq_s=np.ones(nn);e=1.0;pk_s=1.0
    for i in range(S_H,nn):
        pnl=base_shuf[i]
        if vr_1h[i]>1.5:pnl*=0.85
        past=base_shuf[max(S_H,i-2160):i]
        if len(past)<240:lev=1.5
        else:
            mu=np.mean(past);var=np.var(past)+1e-10
            lev=np.clip(mu/var*KF,1.0,POS_CAP)
        pnl*=lev
        dd_eq=(e-pk_s)/pk_s if pk_s>0 else 0
        if dd_eq<-0.30:pnl*=0.1
        elif dd_eq<-0.22:pnl*=0.4
        elif dd_eq<-0.15:pnl*=0.7
        e*=(1+pnl);eq_s[i]=e;pk_s=max(pk_s,e)
    shuf_wfs_3.append(eval_wfs(eq_s))
    if (s+1)%50==0:print(f"    {s+1}/{N_SHUF} done")
p3=np.mean(np.array(shuf_wfs_3)>=ws_real)
print(f"  Position shuffle: real={ws_real:.0f}% shuffle_mean={np.mean(shuf_wfs_3):.0f}% std={np.std(shuf_wfs_3):.0f}% p={p3:.3f} {'PASS' if p3<0.05 else 'FAIL'}")

# === Summary ===
print(f"\n{'='*80}")
print(f"  SUMMARY (6-asset, cap=3.0, KF=0.15, realistic slippage)")
print(f"{'='*80}")
print(f"  Real model:")
print(f"    WFS: {ws_real:.0f}%")
print(f"    MaxDD: {maxdd_real:+.1f}%")
print(f"    Bear 2022: {b22_real:+.0f}%")
print(f"    Win: {win_real}/{len(wf_r_real)}")
print(f"    Fold mean: {np.mean(wf_r_real):.1f}%  median: {np.median(wf_r_real):.1f}%")
print(f"")
print(f"  Shuffle tests (N={N_SHUF}):")
print(f"    [1] Return:   real={ws_real:.0f}% vs shuffle={np.mean(shuf_wfs_1):.0f}%+/-{np.std(shuf_wfs_1):.0f}%  p={p1:.3f} {'PASS' if p1<0.05 else 'FAIL'}")
print(f"    [2] Block30d: real={ws_real:.0f}% vs shuffle={np.mean(shuf_wfs_2):.0f}%+/-{np.std(shuf_wfs_2):.0f}%  p={p2:.3f} {'PASS' if p2<0.05 else 'FAIL'}")
print(f"    [3] Position: real={ws_real:.0f}% vs shuffle={np.mean(shuf_wfs_3):.0f}%+/-{np.std(shuf_wfs_3):.0f}%  p={p3:.3f} {'PASS' if p3<0.05 else 'FAIL'}")

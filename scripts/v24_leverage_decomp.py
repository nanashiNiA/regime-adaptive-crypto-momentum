"""Leverage decomposition: Is alpha real without leverage?"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request
from src.racm_core import RACMParams,RACMFeatures,RACMRegime,RACMCryptoGates,RACMKelly,RACMDDControl,RACMLS,safe_val

btc_h=pd.read_pickle('data/external/binance/btcusdt_hourly.pkl');btc_h.index=pd.to_datetime(btc_h.index)
deriv=pd.read_pickle('data/processed/derivatives_1h.pkl');deriv.index=deriv.index.tz_localize(None)
common_1h=btc_h.index.intersection(deriv.index)
h=pd.DataFrame(index=common_1h);h['close']=btc_h.loc[common_1h,'close'].values
h['ret']=h['close'].pct_change();h['funding']=deriv.loc[common_1h,'funding_rate'].values
h['oi']=deriv.loc[common_1h,'open_interest_last'].values;h['liq_count']=deriv.loc[common_1h,'liq_count'].values
h=h.ffill().dropna(subset=['close'])
nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values;log_p=np.log(np.clip(price,1e-12,None))
params=RACMParams();S=params.warmup_hours
funding_z=RACMFeatures.funding_zscore(h['funding'].fillna(0).values)
oi_z=RACMFeatures.oi_zscore(h['oi'].values);oi_chg=RACMFeatures.oi_change_24h(h['oi'].values)
liq_z=RACMFeatures.liq_zscore(h['liq_count'].fillna(0).values)
vr=RACMFeatures.vol_ratio(ret);rv_pct=RACMFeatures.rv_pctrank(ret)
ret_1d=RACMFeatures.ret_nd(log_p,24);ret_30d=RACMFeatures.ret_nd(log_p,720)
atr_pct_a=RACMFeatures.atr_pct(ret,price);ma_cs=RACMFeatures.ma_cross_slow(price)
fra=np.roll(h['funding'].fillna(0).values,1)
bp_h=RACMRegime.compute(price,ret,params)
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
all_8h={'BTC':btc_8h}
for sym in ['ETHUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','LINKUSDT']:
    try:all_8h[sym.replace('USDT','')]=fc(sym,'8h',20)
    except:pass
c_8h=btc_8h.index
for a in params.assets:
    if a!='BTC' and a in all_8h:c_8h=c_8h.intersection(all_8h[a].index)
a_ret_8h={a:all_8h[a].loc[c_8h,'return'].values for a in params.assets if a in all_8h}
lp_8h,_,_=RACMLS.compute_pnl_8h(a_ret_8h,params.ls_lookbacks,len(c_8h))
lp_1h=RACMLS.map_8h_to_1h(lp_8h,c_8h,idx_h,nn)
dw=max(0,1-params.ls_weight)
base_raw=np.zeros(nn)
for i in range(S,nn):base_raw[i]=dw*ret[i]*bp_h[i]+params.ls_weight*lp_1h[i]

folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    te_s=cur+pd.DateOffset(months=3);te_e=te_s+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    te_m=np.array([(d>=te_s and d<=te_e) for d in idx_h])
    if te_m.sum()>=20:folds.append(np.where(te_m)[0])
    cur+=pd.DateOffset(months=1)

def run(pnl_arr, lev_mult=1.0, use_kelly=False, use_dd=False, cap=3.0):
    eq=np.ones(nn);e=1.0;pk_e=1.0
    for i in range(S,nn):
        pnl=pnl_arr[i]
        if use_kelly:
            past=pnl_arr[max(S,i-2160):i]
            lev,vt=RACMKelly.compute(past,params)
            pnl*=min(lev*vt,cap)
        else:
            pnl*=lev_mult
        if use_dd:
            dd_mult=RACMDDControl.compute(e,pk_e,params);pnl*=dd_mult
        e*=(1+pnl);eq[i]=e;pk_e=max(pk_e,e)
    wf_r=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    return np.mean(wf_r)*12, sum(1 for r in wf_r if r>0), len(wf_r)

# Component arrays
ls_only = np.zeros(nn)
regime_only = np.zeros(nn)
for i in range(S,nn):
    ls_only[i] = lp_1h[i]
    regime_only[i] = ret[i] * bp_h[i]

print("="*70)
print("  LEVERAGE DECOMPOSITION")
print("="*70)

print("\n  --- What creates alpha? (all at 1x, no leverage) ---\n")
print("  %-45s %6s %6s" % ("Component", "WFS", "Win"))
print("  " + "-"*60)

w,n,t=run(np.array([ret[i] for i in range(nn)]),1.0)
print("  %-45s %+5.0f%% %2d/%d" % ("BTC Buy & Hold", w, n, t))
w,n,t=run(regime_only, 1.0)
print("  %-45s %+5.0f%% %2d/%d" % ("Regime x BTC only (no LS)", w, n, t))
w,n,t=run(ls_only, 1.0)
print("  %-45s %+5.0f%% %2d/%d" % ("LS only (no regime)", w, n, t))
w_base,n,t=run(base_raw, 1.0)
print("  %-45s %+5.0f%% %2d/%d" % ("Base signal (regime+LS blend, 1x)", w_base, n, t))

print("\n  --- How does leverage scale it? ---\n")
print("  %-45s %6s %6s %8s" % ("Config", "WFS", "Win", "vs 1x"))
print("  " + "-"*65)
for lev in [1.0, 1.5, 2.0, 2.5, 3.0]:
    w,n,t=run(base_raw, lev)
    ratio = w / w_base if w_base != 0 else 0
    print("  %-45s %+5.0f%% %2d/%d  %.1fx" % ("Fixed %.1fx" % lev, w, n, t, ratio))

w_kelly,n,t=run(base_raw, use_kelly=True, cap=3.0)
print("  %-45s %+5.0f%% %2d/%d  %.1fx" % ("Kelly adaptive (cap=3.0)", w_kelly, n, t, w_kelly/w_base if w_base else 0))
w_full,n,t=run(base_raw, use_kelly=True, use_dd=True, cap=3.0)
print("  %-45s %+5.0f%% %2d/%d  %.1fx" % ("Kelly + DD control (FULL)", w_full, n, t, w_full/w_base if w_base else 0))

print("\n  --- Kelly vs fixed: What does Kelly actually do? ---\n")
# Compare Kelly vs fixed at each year
print("  %-6s %8s %8s %8s" % ("Year", "Fixed 3x", "Kelly 3x", "Kelly advantage"))
for y in range(2021,2025):
    mask=np.array([d.year==y for d in idx_h]);iy=np.where(mask)[0]
    if len(iy)<100:continue
    # Fixed 3x
    eq_f=np.ones(nn);e=1.0
    for i in range(max(S,iy[0]),iy[-1]+1):e*=(1+base_raw[i]*3.0);eq_f[i]=e
    yr_f=(eq_f[iy[-1]]/eq_f[max(0,iy[0]-1)]-1)*100

    # Kelly 3x
    eq_k=np.ones(nn);e=1.0
    for i in range(max(S,iy[0]),iy[-1]+1):
        past=base_raw[max(S,i-2160):i]
        lev,vt=RACMKelly.compute(past,params)
        tl=min(lev*vt,3.0)
        e*=(1+base_raw[i]*tl);eq_k[i]=e
    yr_k=(eq_k[iy[-1]]/eq_k[max(0,iy[0]-1)]-1)*100
    print("  %-6d %+7.0f%% %+7.0f%% %+7.0f%%" % (y, yr_f, yr_k, yr_k-yr_f))

print("\n" + "="*70)
print("  VERDICT")
print("="*70)
print()
print("  Base signal at 1x: WFS = %+.0f%%" % w_base)
if w_base > 50:
    print("  -> ALPHA IS REAL without leverage")
    print("  -> Leverage amplifies %.0fx (1x->Kelly 3x)" % (w_full/w_base if w_base else 0))
    print()
    print("  Breakdown:")
    print("    Signal alpha (1x):  %+.0f%% = base" % w_base)
    print("    Leverage effect:    %+.0f%% = amplification" % (w_full - w_base))
    print("    Total:              %+.0f%%" % w_full)
    print("    Leverage share:     %.0f%% of total" % ((w_full-w_base)/w_full*100 if w_full else 0))

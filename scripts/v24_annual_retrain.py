"""Annual retraining test: Does recalibrating yearly improve performance?
Compare:
  A) FIXED: Same params entire period (current model)
  B) ANNUAL: Retrain cap/KF on prior year, apply to next year
  C) EXPANDING: Use ALL past data to optimize, apply to next year
"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request
from src.racm_core import RACMParams,RACMFeatures,RACMRegime,RACMCryptoGates,RACMKelly,RACMDDControl,RACMLS,safe_val

# === Load data (including 2025) ===
btc_h=pd.read_pickle('data/external/binance/btcusdt_hourly.pkl');btc_h.index=pd.to_datetime(btc_h.index)
deriv=pd.read_pickle('data/processed/derivatives_1h.pkl');deriv.index=deriv.index.tz_localize(None)
def fetch_btc_api():
    rows=[];et=int(pd.Timestamp.now().timestamp()*1000)
    for _ in range(20):
        u=f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000&endTime={et}'
        with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'M'}),timeout=15) as r:d=json.loads(r.read())
        if not d:break
        for k in d:rows.append({'timestamp':pd.Timestamp(k[0],unit='ms'),'close':float(k[4])})
        et=int(d[0][0])-1
    return pd.DataFrame(rows).drop_duplicates('timestamp').set_index('timestamp').sort_index()
btc_api=fetch_btc_api()
btc_combined=pd.concat([btc_h,btc_api[~btc_api.index.isin(btc_h.index)]]).sort_index()
common_1h=btc_combined.index.intersection(deriv.index)
h=pd.DataFrame(index=common_1h);h['close']=btc_combined.loc[common_1h,'close'].values
h['ret']=h['close'].pct_change();h['funding']=deriv.loc[common_1h,'funding_rate'].values
h['oi']=deriv.loc[common_1h,'open_interest_last'].values;h['liq_count']=deriv.loc[common_1h,'liq_count'].values
h=h.ffill().dropna(subset=['close'])
nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
log_p=np.log(np.clip(price,1e-12,None))
params=RACMParams()
S=params.warmup_hours
# Features
funding_z=RACMFeatures.funding_zscore(h['funding'].fillna(0).values)
oi_z=RACMFeatures.oi_zscore(h['oi'].values)
oi_chg=RACMFeatures.oi_change_24h(h['oi'].values)
liq_z=RACMFeatures.liq_zscore(h['liq_count'].fillna(0).values)
vr=RACMFeatures.vol_ratio(ret);rv_pct=RACMFeatures.rv_pctrank(ret)
ret_1d=RACMFeatures.ret_nd(log_p,24);ret_30d=RACMFeatures.ret_nd(log_p,720)
atr_pct_arr=RACMFeatures.atr_pct(ret,price);ma_cs=RACMFeatures.ma_cross_slow(price)
fra=np.roll(h['funding'].fillna(0).values,1)
bp_h=RACMRegime.compute(price,ret,params)
# LS
btc_8h=btc_combined.resample('8h').agg({'close':'last'}).dropna();btc_8h['return']=btc_8h['close'].pct_change();btc_8h=btc_8h.dropna()
def fc(s,i,p=25):
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
    try:all_8h[sym.replace('USDT','')]=fc(sym,'8h',25)
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

def run_year(year, kf, cap):
    """Run model for a specific year with given params"""
    start_d=pd.Timestamp(f'{year}-01-01');end_d=pd.Timestamp(f'{year}-12-31')
    mask=np.array([(d>=start_d and d<=end_d) for d in idx_h])
    year_idx=np.where(mask)[0]
    if len(year_idx)<100:return [],0,0
    p=RACMParams();p.kelly_fraction=kf;p.position_cap=cap
    eq=np.ones(nn);e=1.0;pk_e=1.0
    for i in range(max(S,year_idx[0]),min(nn,year_idx[-1]+1)):
        pnl=base_raw[i]
        cs,gate=RACMCryptoGates.compute(safe_val(funding_z,i),safe_val(oi_z,i),safe_val(oi_chg,i),
            safe_val(liq_z,i),safe_val(rv_pct,i,0.5),bp_h[i],safe_val(atr_pct_arr,i,0.01),
            safe_val(ret_30d,i),safe_val(ma_cs,i),safe_val(vr,i,1.0),safe_val(ret_1d,i),p)
        pnl+=cs*ret[i]*p.crypto_alpha_weight;pnl*=gate
        past=base_raw[max(S,i-p.kelly_lookback_hours):i]
        lev,vt=RACMKelly.compute(past,p)
        total_lev=min(lev*vt,p.position_cap);pnl*=total_lev
        pnl+=fra[i]*abs(gate*total_lev)
        dd_mult=RACMDDControl.compute(e,pk_e,p);pnl*=dd_mult
        e*=(1+pnl);eq[i]=e;pk_e=max(pk_e,e)
    # Monthly returns within year
    monthly=[]
    for m in range(1,13):
        mm=np.array([(d.year==year and d.month==m) for d in idx_h])
        im=np.where(mm)[0]
        if len(im)>20:
            mr=(eq[im[-1]]/eq[max(0,im[0]-1)]-1)*100 if eq[max(0,im[0]-1)]>0 else 0
            monthly.append(mr)
    yr_ret=(eq[year_idx[-1]]/eq[max(0,year_idx[0]-1)]-1)*100 if eq[max(0,year_idx[0]-1)]>0 else 0
    return monthly, yr_ret, sum(1 for r in monthly if r>0)

def optimize_on_period(start_year, end_year):
    """Find best KF and cap on a training period"""
    best_sharpe=-999;best_kf=0.15;best_cap=3.0
    for kf in [0.10,0.15,0.20,0.25,0.30]:
        for cap in [2.0,2.5,3.0,3.5]:
            all_m=[]
            for y in range(start_year, end_year+1):
                m,_,_=run_year(y,kf,cap)
                all_m.extend(m)
            if len(all_m)>3:
                sharpe=np.mean(all_m)/(np.std(all_m)+1e-10)
                if sharpe>best_sharpe:
                    best_sharpe=sharpe;best_kf=kf;best_cap=cap
    return best_kf,best_cap

print("="*70)
print("  ANNUAL RETRAINING TEST")
print("="*70)

# === A) FIXED: Same params all years ===
print("\n  [A] FIXED PARAMS (KF=0.15, cap=3.0) - current model")
print(f"  {'Year':<6} {'Return':>8} {'Win':>6} {'Mean/m':>8} {'WFS':>6}")
fixed_all=[]
for y in range(2021,2026):
    m,yr,w=run_year(y,0.15,3.0)
    wfs=np.mean(m)*12 if m else 0
    fixed_all.extend(m)
    print(f"  {y:<6} {yr:>+7.0f}% {w:>2}/{len(m):<2} {np.mean(m):>+7.1f}% {wfs:>5.0f}%")
print(f"  TOTAL  WFS={np.mean(fixed_all)*12:.0f}% Win={sum(1 for r in fixed_all if r>0)}/{len(fixed_all)}")

# === B) ANNUAL RETRAIN: optimize on prior year, apply to next ===
print("\n  [B] ANNUAL RETRAIN (optimize on prior year)")
print(f"  {'Year':<6} {'Train':>8} {'KF':>5} {'Cap':>5} {'Return':>8} {'Win':>6} {'WFS':>6}")
annual_all=[]
for y in range(2022,2026):
    kf,cap=optimize_on_period(y-1,y-1)
    m,yr,w=run_year(y,kf,cap)
    wfs=np.mean(m)*12 if m else 0
    annual_all.extend(m)
    print(f"  {y:<6} {y-1:>8} {kf:>5.2f} {cap:>5.1f} {yr:>+7.0f}% {w:>2}/{len(m):<2} {wfs:>5.0f}%")
print(f"  TOTAL  WFS={np.mean(annual_all)*12:.0f}% Win={sum(1 for r in annual_all if r>0)}/{len(annual_all)}")

# === C) EXPANDING: optimize on ALL past years ===
print("\n  [C] EXPANDING WINDOW (optimize on all past)")
print(f"  {'Year':<6} {'Train':>10} {'KF':>5} {'Cap':>5} {'Return':>8} {'Win':>6} {'WFS':>6}")
expand_all=[]
for y in range(2022,2026):
    kf,cap=optimize_on_period(2021,y-1)
    m,yr,w=run_year(y,kf,cap)
    wfs=np.mean(m)*12 if m else 0
    expand_all.extend(m)
    print(f"  {y:<6} {'2021-'+str(y-1):>10} {kf:>5.2f} {cap:>5.1f} {yr:>+7.0f}% {w:>2}/{len(m):<2} {wfs:>5.0f}%")
print(f"  TOTAL  WFS={np.mean(expand_all)*12:.0f}% Win={sum(1 for r in expand_all if r>0)}/{len(expand_all)}")

# === D) FIXED but lower cap (more conservative) ===
print("\n  [D] FIXED CONSERVATIVE (KF=0.15, cap=2.5)")
cons_all=[]
for y in range(2022,2026):
    m,yr,w=run_year(y,0.15,2.5)
    cons_all.extend(m)
print(f"  TOTAL  WFS={np.mean(cons_all)*12:.0f}% Win={sum(1 for r in cons_all if r>0)}/{len(cons_all)}")

# === Summary ===
print(f"\n{'='*70}")
print(f"  COMPARISON (2022-2025, common period)")
print(f"{'='*70}")
# Recompute fixed for 2022-2025 only
fixed_2225=[]
for y in range(2022,2026):
    m,_,_=run_year(y,0.15,3.0)
    fixed_2225.extend(m)

methods=[
    ('A) Fixed (KF=0.15, cap=3.0)', fixed_2225),
    ('B) Annual retrain', annual_all),
    ('C) Expanding window', expand_all),
    ('D) Fixed conservative (cap=2.5)', cons_all),
]
print(f"\n  {'Method':<35} {'WFS':>6} {'Win':>8} {'Mean/m':>8} {'Sharpe':>7}")
print(f"  {'-'*65}")
for name,rets in methods:
    wfs=np.mean(rets)*12;win=sum(1 for r in rets if r>0)
    sh=np.mean(rets)/(np.std(rets)+1e-10)*np.sqrt(12)
    print(f"  {name:<35} {wfs:>5.0f}% {win:>3}/{len(rets):<3} {np.mean(rets):>+7.1f}% {sh:>6.2f}")

print(f"\n  VERDICT:")
best_wfs=max((np.mean(r)*12,n) for n,r in methods)
print(f"  Best: {best_wfs[1]}")

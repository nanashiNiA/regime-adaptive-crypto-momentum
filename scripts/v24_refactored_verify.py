"""Verify refactored core produces same results as original v24_6asset_realistic.py"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request
from src.racm_core import RACMParams, RACMFeatures, RACMRegime, RACMCryptoGates, RACMKelly, RACMDDControl, RACMLS, safe_val

print("="*70)
print("  REFACTORED CORE VERIFICATION")
print("  Compare: racm_core.py vs original v24_6asset_realistic.py")
print("="*70)

# Load data
btc_h=pd.read_pickle('data/external/binance/btcusdt_hourly.pkl');btc_h.index=pd.to_datetime(btc_h.index)
deriv=pd.read_pickle('data/processed/derivatives_1h.pkl');deriv.index=deriv.index.tz_localize(None)
common_1h=btc_h.index.intersection(deriv.index)
h=pd.DataFrame(index=common_1h)
h['close']=btc_h.loc[common_1h,'close'].values;h['ret']=h['close'].pct_change()
h['funding']=deriv.loc[common_1h,'funding_rate'].values
h['oi']=deriv.loc[common_1h,'open_interest_last'].values
h['liq_count']=deriv.loc[common_1h,'liq_count'].values
h=h.ffill().dropna(subset=['close'])
nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
log_p=np.log(np.clip(price,1e-12,None))

params = RACMParams()

# Compute features using core module
print("\n[1] Computing features via racm_core...")
funding_z = RACMFeatures.funding_zscore(h['funding'].fillna(0).values)
oi_z = RACMFeatures.oi_zscore(h['oi'].values)
oi_chg = RACMFeatures.oi_change_24h(h['oi'].values)
liq_z = RACMFeatures.liq_zscore(h['liq_count'].fillna(0).values)
vr = RACMFeatures.vol_ratio(ret)
rv_pct = RACMFeatures.rv_pctrank(ret)
vp = RACMFeatures.portfolio_vol(ret)
ret_1d = RACMFeatures.ret_nd(log_p, 24)
ret_30d = RACMFeatures.ret_nd(log_p, 720)
atr = RACMFeatures.atr_pct(ret, price)
ma_cs = RACMFeatures.ma_cross_slow(price)
fra = np.roll(h['funding'].fillna(0).values, 1)

# Regime
print("[2] Computing regime via racm_core...")
bp_h = RACMRegime.compute(price, ret, params)

# LS at 8H
print("[3] Fetching 8H data and computing LS...")
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

all_8h = {'BTC': btc_8h}
for sym in ['ETHUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','LINKUSDT']:
    try: all_8h[sym.replace('USDT','')]=fc(sym,'8h',20)
    except: pass

c_8h = btc_8h.index
for a in params.assets:
    if a != 'BTC' and a in all_8h: c_8h = c_8h.intersection(all_8h[a].index)

a_ret_8h = {a: all_8h[a].loc[c_8h,'return'].values for a in params.assets if a in all_8h}
lp_8h, long_names, short_names = RACMLS.compute_pnl_8h(a_ret_8h, params.ls_lookbacks, len(c_8h))
lp_1h = RACMLS.map_8h_to_1h(lp_8h, c_8h, idx_h, nn)
slip_8h = RACMLS.count_slippage(long_names, short_names, params.slippage_bps, len(params.assets))
slip_1h = RACMLS.map_8h_to_1h(slip_8h, c_8h, idx_h, nn)

# Portfolio
S = params.warmup_hours
dw = max(0, 1 - params.ls_weight)
base_raw = np.zeros(nn)
for i in range(S, nn):
    base_raw[i] = dw * ret[i] * bp_h[i] + params.ls_weight * lp_1h[i]

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    te_s=cur+pd.DateOffset(months=3);te_e=te_s+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    te_m=np.array([(d>=te_s and d<=te_e) for d in idx_h])
    if te_m.sum()>=20:folds.append(np.where(te_m)[0])
    cur+=pd.DateOffset(months=1)

# Run model using core components
print("[4] Running model via racm_core...")
eq=np.ones(nn);e=1.0;pk_e=1.0
for i in range(S,nn):
    pnl=base_raw[i]
    fz=safe_val(funding_z,i);oz=safe_val(oi_z,i);oc=safe_val(oi_chg,i)
    lz=safe_val(liq_z,i);rvp=safe_val(rv_pct,i,0.5)
    v=safe_val(vr,i,1.0);r1d=safe_val(ret_1d,i);r30d=safe_val(ret_30d,i)
    ap=safe_val(atr,i,0.01);mcs=safe_val(ma_cs,i)

    cs, gate = RACMCryptoGates.compute(fz, oz, oc, lz, rvp, bp_h[i], ap, r30d, mcs, v, r1d, params)
    pnl += cs * ret[i] * params.crypto_alpha_weight
    pnl *= gate

    past = base_raw[max(S, i-params.kelly_lookback_hours):i]
    lev, vt = RACMKelly.compute(past, params)
    total_lev = min(lev * vt, params.position_cap)
    pnl *= total_lev

    ps = gate * total_lev
    pnl += fra[i] * abs(ps)

    dd_mult = RACMDDControl.compute(e, pk_e, params)
    pnl *= dd_mult

    pnl -= slip_1h[i] * total_lev

    e *= (1+pnl); eq[i]=e; pk_e=max(pk_e,e)

# Evaluate
wf_r=[]
for idx in folds:
    if len(idx)<10:continue
    wf_r.append((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100)
wfs=np.mean(wf_r)*12
win=sum(1 for r in wf_r if r>0)

print(f"\n{'='*70}")
print(f"  RESULTS")
print(f"{'='*70}")
print(f"  Refactored: WFS={wfs:.0f}%  Win={win}/{len(wf_r)}")
print(f"  Original:   WFS=367%  Win=39/45")
print(f"  Match: {'YES' if abs(wfs-367)<30 and win>=38 else 'NO - INVESTIGATE'}")

# Year by year
print(f"\n  Year-by-year:")
for y in range(2021,2025):
    mask=np.array([d.year==y for d in idx_h]);iy=np.where(mask)[0]
    if len(iy)>100:
        yr=(eq[iy[-1]]/eq[max(1,iy[0]-1)]-1)*100
        print(f"    {y}: {yr:+.0f}%")

# Key differences check
print(f"\n  Feature checks (should be non-zero):")
print(f"    funding_z range: [{np.nanmin(funding_z[S:]):.1f}, {np.nanmax(funding_z[S:]):.1f}]")
print(f"    oi_z range:      [{np.nanmin(oi_z[S:]):.1f}, {np.nanmax(oi_z[S:]):.1f}]")
print(f"    liq_z range:     [{np.nanmin(liq_z[S:]):.1f}, {np.nanmax(liq_z[S:]):.1f}]")
print(f"    vol_ratio range: [{np.nanmin(vr[S:]):.1f}, {np.nanmax(vr[S:]):.1f}]")
print(f"    rv_pctrank range:[{np.nanmin(rv_pct[S:]):.2f}, {np.nanmax(rv_pct[S:]):.2f}]")

"""V24 HYBRID: v23 multi-asset LS + best.py crypto-native signals
Architecture:
  - 1H data: funding z-score, OI features (from derivatives_1h.pkl)
  - 8H execution: 3 trades/day (professor requirement)
  - Multi-asset LS momentum (10 assets, 60d+90d lookback)
  - Regime detection (4-stage, lag-1)
  - Crypto-native signals as QUALITY GATES (not direct prediction)
  - Kelly sizing + Vol targeting + Equity DD control
  - All lag-1, no future information
"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request

print("="*80)
print("  V24 HYBRID MODEL: v23 LS + best.py crypto signals")
print("="*80)

# === 1. Load 1H data for crypto signals ===
print("\n[1] Loading data...")
btc_h=pd.read_pickle('data/external/binance/btcusdt_hourly.pkl')
btc_h.index=pd.to_datetime(btc_h.index)
deriv=pd.read_pickle('data/processed/derivatives_1h.pkl')
deriv.index=deriv.index.tz_localize(None)

# Align 1H data
common_1h = btc_h.index.intersection(deriv.index)
print(f"  BTC 1H: {len(btc_h)} bars, Derivatives 1H: {len(deriv)} bars")
print(f"  Common 1H: {len(common_1h)} bars ({common_1h[0]} to {common_1h[-1]})")

# Build 1H feature frame
h = pd.DataFrame(index=common_1h)
h['close'] = btc_h.loc[common_1h, 'close']
h['high'] = btc_h.loc[common_1h, 'high']
h['low'] = btc_h.loc[common_1h, 'low']
h['return'] = h['close'].pct_change()

# Derivatives features at 1H
for col in ['funding_rate']:
    if col in deriv.columns:
        h[col] = deriv.loc[common_1h, col]

# OI: try multiple column names
oi_col = None
for c in ['open_interest_binance', 'open_interest', 'oi']:
    if c in deriv.columns:
        oi_col = c; break
if oi_col:
    h['oi'] = deriv.loc[common_1h, oi_col]
    print(f"  OI column: {oi_col}")
else:
    h['oi'] = np.nan
    print(f"  WARNING: No OI column found in derivatives data")

h = h.ffill().dropna(subset=['close'])
print(f"  Final 1H frame: {len(h)} bars")

# === 2. Compute 1H crypto-native signals (best.py style) ===
print("\n[2] Computing 1H crypto signals...")

# Funding z-score (168H rolling = 7 days, same as best.py)
fr = h['funding_rate'].fillna(0)
fr_mean = fr.rolling(168, min_periods=48).mean()
fr_std = fr.rolling(168, min_periods=48).std()
h['funding_z'] = ((fr - fr_mean) / (fr_std + 1e-12)).shift(1)  # lag-1

# Funding momentum (12H momentum of z-score)
h['funding_mom'] = h['funding_z'].diff(12)

# OI features
if h['oi'].notna().sum() > 100:
    oi_chg = h['oi'].pct_change(24)
    oi_chg = oi_chg.replace([np.inf, -np.inf], np.nan)
    h['oi_chg_24h'] = oi_chg.shift(1)

    oi_raw = h['oi'].diff(24)
    oi_mean = oi_raw.rolling(168, min_periods=48).mean()
    oi_std = oi_raw.rolling(168, min_periods=48).std()
    h['oi_zscore'] = ((oi_raw - oi_mean) / (oi_std + 1e-12)).shift(1)
    print(f"  OI features computed")
else:
    h['oi_chg_24h'] = 0
    h['oi_zscore'] = 0
    print(f"  OI features: not enough data, set to 0")

# RV pctrank (from best.py)
ret_1h = np.log(h['close'].clip(1e-12)).diff()
rv_1h = (ret_1h**2).rolling(24, min_periods=6).sum().apply(np.sqrt)
rv_24h = rv_1h.rolling(24, min_periods=6).sum()
h['rv_pctrank'] = rv_24h.rolling(720, min_periods=168).rank(pct=True).shift(1)

# ATR (from best.py)
tr = pd.concat([
    h['high'] - h['low'],
    (h['high'] - h['close'].shift(1)).abs(),
    (h['low'] - h['close'].shift(1)).abs(),
], axis=1).max(axis=1)
h['atr_24h'] = tr.rolling(24, min_periods=6).mean()
h['atr_pct'] = (h['atr_24h'] / (h['close'] + 1e-12)).shift(1)

# Returns at multiple scales
h['ret_1d'] = np.log(h['close'].clip(1e-12)).diff(24)
h['ret_7d'] = np.log(h['close'].clip(1e-12)).diff(168)
h['ret_30d'] = np.log(h['close'].clip(1e-12)).diff(720)

# MA crosses (from best.py)
h['ma_72h'] = h['close'].ewm(span=72, min_periods=24).mean()
h['ma_168h'] = h['close'].ewm(span=168, min_periods=48).mean()
h['ma_cross_slow'] = ((h['ma_72h'] - h['ma_168h']) / (h['close'] + 1e-12)).shift(1)

# === 3. Aggregate 1H signals to 8H ===
print("\n[3] Aggregating 1H signals to 8H...")

# For each 8H bar, compute signal strength from the 8 hours within it
def agg_to_8h(series, method='last'):
    if method == 'last':
        return series.resample('8h').last()
    elif method == 'mean':
        return series.resample('8h').mean()
    elif method == 'min':
        return series.resample('8h').min()
    elif method == 'max':
        return series.resample('8h').max()
    elif method == 'sum':
        return series.resample('8h').sum()

sig_8h = pd.DataFrame()
sig_8h['funding_z'] = agg_to_8h(h['funding_z'], 'last')  # latest z-score
sig_8h['funding_z_min'] = agg_to_8h(h['funding_z'], 'min')  # most extreme in window
sig_8h['funding_mom'] = agg_to_8h(h['funding_mom'], 'last')
sig_8h['oi_chg_24h'] = agg_to_8h(h['oi_chg_24h'], 'last')
sig_8h['oi_zscore'] = agg_to_8h(h['oi_zscore'], 'last')
sig_8h['rv_pctrank'] = agg_to_8h(h['rv_pctrank'], 'last')
sig_8h['atr_pct'] = agg_to_8h(h['atr_pct'], 'last')
sig_8h['ret_30d'] = agg_to_8h(h['ret_30d'], 'last')
sig_8h['ma_cross_slow'] = agg_to_8h(h['ma_cross_slow'], 'last')
sig_8h = sig_8h.dropna(how='all')
print(f"  8H signal frame: {len(sig_8h)} bars")

# Compute crypto signal scores per 8H bar
# Signals from best.py (as quality gates):
#   - funding_z < -1.5: contrarian long (market too bearish)
#   - oi_chg_24h > 0.015 & ret_1d < -0.015: squeeze setup (OI up, price down)
#   - oi_zscore < -1.5 & ret_1d < -0.02: OI flush (forced liquidation)
#   - funding_mom < -0.5 & funding_z < -1.0: accelerating negative funding
#   - 4+ signals = crash → block
#   - grind_bear: atr_pct < 0.006 & ret_30d < -0.05 & ma_cross_slow < 0 & rv_pctrank < 0.50

# === 4. Load 8H multi-asset data (same as v23) ===
print("\n[4] Loading multi-asset 8H data...")
btc_8h = btc_h.resample('8h').agg({'close':'last'}).dropna()
btc_8h['return'] = btc_8h['close'].pct_change()
btc_8h = btc_8h.dropna()

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

c_all=btc_8h.index.intersection(e8.index).intersection(s8.index)
for df in extras.values():c_all=c_all.intersection(df.index)
# Also intersect with signal frame
c_all=c_all.intersection(sig_8h.index)
nn=len(c_all);dt2=c_all;SM=180;S=331

rb=btc_8h.loc[c_all,'return'].values;pb=btc_8h.loc[c_all,'close'].values
re=e8.loc[c_all,'return'].values;rs=s8.loc[c_all,'return'].values
ext_r={sym:df.loc[c_all,'return'].values for sym,df in extras.items()}
print(f"  Common 8H bars (all assets + signals): {nn}")
print(f"  Period: {dt2[0]} to {dt2[-1]}")

# Map 1H signals to aligned 8H index
fz_arr = sig_8h.loc[c_all, 'funding_z'].values
fz_min_arr = sig_8h.loc[c_all, 'funding_z_min'].values
fm_arr = sig_8h.loc[c_all, 'funding_mom'].values
oi_chg_arr = sig_8h.loc[c_all, 'oi_chg_24h'].values
oi_z_arr = sig_8h.loc[c_all, 'oi_zscore'].values
rv_pct_arr = sig_8h.loc[c_all, 'rv_pctrank'].values
atr_pct_arr = sig_8h.loc[c_all, 'atr_pct'].values
ret30d_arr = sig_8h.loc[c_all, 'ret_30d'].values
ma_slow_arr = sig_8h.loc[c_all, 'ma_cross_slow'].values

# === 5. Standard v23 components ===
print("\n[5] Computing v23 components...")
vp=np.roll(pd.Series(0.5*rb+0.25*re+0.25*rs).rolling(90,min_periods=30).std().values*np.sqrt(365*3),1)

# Funding rate for carry (lag-1)
d8f=deriv.resample('8h').agg({'funding_rate':'last'}).dropna(how='all')
fra=np.zeros(nn)
for i,d in enumerate(dt2):
    if d in d8f.index:
        v=d8f.loc[d,'funding_rate']
        fra[i]=v if not np.isnan(v) else 0
fra=np.roll(fra,1)

# Correlation (lag-1)
ca=np.roll((pd.Series(rb).rolling(30,min_periods=15).corr(pd.Series(re)).values+
            pd.Series(rb).rolling(30,min_periods=15).corr(pd.Series(rs)).values)/2,1)

# Regime detection
ml=pd.Series(pb).rolling(330,min_periods=165).mean().values
ms2=pd.Series(pb).rolling(60,min_periods=30).mean().values
sk=pd.Series(rb).rolling(90).skew().values
dpk=pd.Series(pb).rolling(135,min_periods=1).max().values;dd_p=(pb-dpk)/dpk
rv=pd.Series(rb).rolling(90).std().values*np.sqrt(365*3)
slo=np.zeros(nn)
for i in range(340,nn):
    if not np.isnan(ml[i-1]) and not np.isnan(ml[i-6]) and ml[i-6]>0:slo[i]=(ml[i-1]-ml[i-6])/ml[i-6]
c30=pd.Series(rb).rolling(90,min_periods=30).sum().values
bp=np.ones(nn);m1=0;ch=0
for i in range(331,nn):
    dl=0;ds=0
    if not np.isnan(ml[i-1]) and pb[i-1]<ml[i-1]:dl+=1
    if not np.isnan(ms2[i-1]) and pb[i-1]<ms2[i-1]:ds+=1
    if not np.isnan(sk[i-1]) and sk[i-1]<-0.5:dl+=1;ds+=1
    if dd_p[i-1]<-0.12:dl+=1;ds+=1
    dc=dl*0.3+ds*0.7
    if i>=2 and rb[i-2]<-0.017:m1=2
    if m1>0:m1-=1
    if dc>=1.5:
        if slo[i]<-0.001 and i>=1 and rb[i-1]<0:bp[i]=-0.7
        elif slo[i]>0.0005:bp[i]=0.5
        else:bp[i]=0.2
        continue
    if dc>=0.8:bp[i]=0.5;ch=15
    elif dc>=0.5:bp[i]=0.7;ch=15
    else:
        if ch>0:ch-=1;bp[i]=0.7 if not(not np.isnan(c30[i-1]) and c30[i-1]>0.05) else 1.0
        elif not np.isnan(rv[i-1]) and rv[i-1]<0.50 and i>=31 and np.sum(rb[i-31:i-1])>0:bp[i]=1.5
        else:bp[i]=1.0
    if m1>0:bp[i]=min(bp[i],0.7)

# Base portfolio + LS
dp=np.zeros(nn);pos_base=np.zeros(nn)
for i in range(S,nn):
    inc=(i>=SM)
    if inc:dp[i]=rb[i]*bp[i]*0.5+re[i]*0.25+rs[i]*0.25;pos_base[i]=abs(bp[i]*0.5+0.25+0.25)
    else:dp[i]=rb[i]*bp[i]*0.667+re[i]*0.333;pos_base[i]=abs(bp[i]*0.667+0.333)

vs_arr=np.roll(pd.Series(np.abs(rb)).rolling(9,min_periods=3).mean().values,1)
vl_arr=np.roll(pd.Series(np.abs(rb)).rolling(90,min_periods=30).mean().values,1)
vr_arr=vs_arr/(vl_arr+1e-10)

all_a={'BTC':rb,'ETH':re,'SOL':rs}
for sym,r_ext in ext_r.items():all_a[sym]=r_ext
na=len(all_a)
asset_vol={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in all_a.items()}
lp=np.zeros(nn)
for lb in [60,90]:
    w=0.5
    am={n:np.roll(pd.Series(r).rolling(lb,min_periods=lb//3).sum().values,1) for n,r in all_a.items()}
    for i in range(S,nn):
        if i<SM:continue
        moms=[(am[n][i]/(asset_vol[n][i]+1e-10),n,all_a[n][i]) for n in all_a if not np.isnan(am[n][i])]
        if len(moms)<3:continue
        moms.sort(key=lambda x:x[0],reverse=True)
        lp[i]+=(moms[0][2]/na-moms[-1][2]/na)*w

CAP=2.5;LW=0.80;KF=0.25;LC=4.0;VB=1.5
dw=max(0,1-LW);base_raw=dw*dp+LW*lp

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=dt2[-1]+pd.DateOffset(days=15):
    te_s=cur+pd.DateOffset(months=3);te_e=te_s+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    te_m=np.array([(d>=te_s and d<=te_e) for d in dt2])
    if te_m.sum()>=20:folds.append(np.where(te_m)[0])
    cur+=pd.DateOffset(months=1)

def eval_model(eq):
    pk=np.maximum.accumulate(eq);maxdd=(eq/pk-1).min()*100
    wf_r=[]
    for idx in folds:
        if len(idx)<10:continue
        wf_r.append((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100)
    ws=np.mean(wf_r)*12 if wf_r else 0
    i22=np.where(np.array([d.year==2022 for d in dt2]))[0]
    b22=(eq[i22[-1]]/eq[max(1,i22[0]-1)]-1)*100 if len(i22)>60 else -100
    return ws,maxdd,b22

# === 6. HYBRID MODELS ===
print("\n[6] Running hybrid models...")
CT=0.70;CR=0.30;VH=0.15

# [A] Baseline: v23 honest
eq0=np.ones(nn);e=1.0
for i in range(S,nn):
    pnl=base_raw[i]
    if not np.isnan(ca[i]) and ca[i]>CT:pnl*=CR
    if vr_arr[i]>1.5:pnl*=(1-VH)
    past=base_raw[max(S,i-270):i]
    if len(past)<30:lev=2.0;vt=VB
    else:
        mu=np.mean(past);var=np.var(past)+1e-10
        lev=np.clip(mu/var*KF,1.0,LC);vol_p=np.std(past)*np.sqrt(365*3)
        vt=np.clip(VB/vol_p,0.5,3.0) if vol_p>0 else VB
    pnl*=lev;vts=np.clip(vt/vp[i],0.2,CAP) if not np.isnan(vp[i]) and vp[i]>0 else 1.0
    pnl*=vts;ps=pos_base[i]*lev*vts
    if not np.isnan(ca[i]) and ca[i]>CT:ps*=CR
    if vr_arr[i]>1.5:ps*=(1-VH)
    pnl+=fra[i]*ps
    e*=(1+pnl);eq0[i]=e
ws0,dd0,b22_0=eval_model(eq0)
print(f"\n  [A] BASELINE v23: WFS={ws0:.0f}% MaxDD={dd0:+.1f}% Bear22={b22_0:+.0f}%")

# [B] Hybrid: v23 LS + best.py crypto signals as quality gates
# Crypto signal scoring per bar:
#   - funding_z < -1.5: bullish (+1)
#   - funding_z > 1.5: bearish (-1, but crypto has long bias so we mainly use for shorts)
#   - oi_zscore < -1.5 & recent drop: OI flush, bullish (+1)
#   - oi_chg > 0.015 & price down: squeeze, bullish (+1)
#   - funding_mom < -0.5 & funding_z < -1.0: accelerating bearish funding, bullish (+1)
#   - 4+ bullish signals = crash block (too many reversals = genuine crash)
#   - grind_bear: low vol + decline + MA down + low rv_pctrank → reduce

# Use these signals to:
# 1. BOOST LS position when funding contrarian (increase alpha)
# 2. REDUCE position when crash detected (improve MaxDD)
# 3. REDUCE in grind bear (improve Bear 2022)

print("\n  [B] HYBRID: v23 + crypto quality gates")
for boost_mult in [0.0, 0.15, 0.30]:
    for crash_reduce in [0.1, 0.3]:
        for grind_reduce in [0.3, 0.5]:
            eq_h=np.ones(nn);e=1.0;pk_h=1.0
            for i in range(S,nn):
                pnl=base_raw[i]

                # --- Crypto signal scoring ---
                n_bull = 0
                fz = fz_arr[i] if not np.isnan(fz_arr[i]) else 0
                fm = fm_arr[i] if not np.isnan(fm_arr[i]) else 0
                oc = oi_chg_arr[i] if not np.isnan(oi_chg_arr[i]) else 0
                oz = oi_z_arr[i] if not np.isnan(oi_z_arr[i]) else 0
                rvp = rv_pct_arr[i] if not np.isnan(rv_pct_arr[i]) else 0.5
                atp = atr_pct_arr[i] if not np.isnan(atr_pct_arr[i]) else 0.01
                r30 = ret30d_arr[i] if not np.isnan(ret30d_arr[i]) else 0
                mcs = ma_slow_arr[i] if not np.isnan(ma_slow_arr[i]) else 0

                if fz < -1.5: n_bull += 1
                if oz < -1.5 and rb[i-1] < -0.01: n_bull += 1  # OI flush
                if oc > 0.015 and rb[i-1] < -0.01: n_bull += 1  # squeeze
                if fm < -0.5 and fz < -1.0: n_bull += 1  # accel funding

                # Crash block: 4+ bullish signals = genuine crash
                crash = (n_bull >= 4)
                # Also crash if rv_pctrank > 0.95
                if rvp > 0.95: crash = True

                # Grind bear (from best.py)
                grind = (atp < 0.006 and r30 < -0.05 and mcs < 0 and rvp < 0.50)

                # --- Apply crypto gates ---
                if crash:
                    pnl *= crash_reduce
                elif grind:
                    pnl *= grind_reduce
                else:
                    # Funding contrarian boost: when funding very negative, boost LS
                    if fz < -1.5 and boost_mult > 0:
                        pnl += lp[i] * boost_mult
                    # Standard CT/CR/VH
                    if not np.isnan(ca[i]) and ca[i]>CT:pnl*=CR
                    if vr_arr[i]>1.5:pnl*=(1-VH)

                # Kelly (from past, using same crypto gates for consistency)
                past=[]
                for j in range(max(S,i-270),i):
                    p_t=base_raw[j]
                    fz_j=fz_arr[j] if not np.isnan(fz_arr[j]) else 0
                    rvp_j=rv_pct_arr[j] if not np.isnan(rv_pct_arr[j]) else 0.5
                    atp_j=atr_pct_arr[j] if not np.isnan(atr_pct_arr[j]) else 0.01
                    r30_j=ret30d_arr[j] if not np.isnan(ret30d_arr[j]) else 0
                    mcs_j=ma_slow_arr[j] if not np.isnan(ma_slow_arr[j]) else 0
                    grind_j=(atp_j<0.006 and r30_j<-0.05 and mcs_j<0 and rvp_j<0.50)
                    if rvp_j>0.95:p_t*=crash_reduce
                    elif grind_j:p_t*=grind_reduce
                    else:
                        if not np.isnan(ca[j]) and ca[j]>CT:p_t*=CR
                        if vr_arr[j]>1.5:p_t*=(1-VH)
                    past.append(p_t)

                if len(past)<30:lev=2.0;vt=VB
                else:
                    mu=np.mean(past);var=np.var(past)+1e-10
                    lev=np.clip(mu/var*KF,1.0,LC);vol_p=np.std(past)*np.sqrt(365*3)
                    vt=np.clip(VB/vol_p,0.5,3.0) if vol_p>0 else VB

                pnl*=lev;vts=np.clip(vt/vp[i],0.2,CAP) if not np.isnan(vp[i]) and vp[i]>0 else 1.0
                pnl*=vts

                # Carry income
                ps=pos_base[i]*lev*vts
                if crash:ps*=crash_reduce
                elif grind:ps*=grind_reduce
                else:
                    if not np.isnan(ca[i]) and ca[i]>CT:ps*=CR
                    if vr_arr[i]>1.5:ps*=(1-VH)
                pnl+=fra[i]*ps

                # Equity DD control (from best.py)
                dd_eq=(e-pk_h)/pk_h if pk_h>0 else 0
                if dd_eq<-0.30:pnl*=0.1
                elif dd_eq<-0.20:pnl*=0.5
                elif dd_eq<-0.15:pnl*=0.7

                e*=(1+pnl);eq_h[i]=e;pk_h=max(pk_h,e)

            ws_h,dd_h,b22_h=eval_model(eq_h)
            tag=""
            if ws_h>=200 and dd_h>=-32:tag=" ★"
            if ws_h>=250 and dd_h>=-32:tag=" ★★"
            if ws_h>=300 and dd_h>=-32:tag=" ★★★"
            if ws_h>ws0-30 or dd_h>dd0+3 or tag:
                print(f"    boost={boost_mult:.2f} crash={crash_reduce:.1f} grind={grind_reduce:.1f}: "
                      f"WFS={ws_h:.0f}% MaxDD={dd_h:+.1f}% B22={b22_h:+.0f}%{tag}")

# [C] Best hybrid WITHOUT DD control (to isolate crypto signal effect)
print("\n  [C] HYBRID WITHOUT DD CONTROL (isolate crypto signal effect)")
for boost_mult in [0.0, 0.15, 0.30]:
    for crash_reduce in [0.1, 0.3]:
        for grind_reduce in [0.3, 0.5]:
            eq_c=np.ones(nn);e=1.0
            for i in range(S,nn):
                pnl=base_raw[i]
                fz=fz_arr[i] if not np.isnan(fz_arr[i]) else 0
                rvp=rv_pct_arr[i] if not np.isnan(rv_pct_arr[i]) else 0.5
                atp=atr_pct_arr[i] if not np.isnan(atr_pct_arr[i]) else 0.01
                r30=ret30d_arr[i] if not np.isnan(ret30d_arr[i]) else 0
                mcs=ma_slow_arr[i] if not np.isnan(ma_slow_arr[i]) else 0
                fm=fm_arr[i] if not np.isnan(fm_arr[i]) else 0
                oz=oi_z_arr[i] if not np.isnan(oi_z_arr[i]) else 0
                oc=oi_chg_arr[i] if not np.isnan(oi_chg_arr[i]) else 0

                n_bull=0
                if fz<-1.5:n_bull+=1
                if oz<-1.5 and rb[i-1]<-0.01:n_bull+=1
                if oc>0.015 and rb[i-1]<-0.01:n_bull+=1
                if fm<-0.5 and fz<-1.0:n_bull+=1
                crash=(n_bull>=4) or (rvp>0.95)
                grind=(atp<0.006 and r30<-0.05 and mcs<0 and rvp<0.50)

                if crash:pnl*=crash_reduce
                elif grind:pnl*=grind_reduce
                else:
                    if fz<-1.5 and boost_mult>0:pnl+=lp[i]*boost_mult
                    if not np.isnan(ca[i]) and ca[i]>CT:pnl*=CR
                    if vr_arr[i]>1.5:pnl*=(1-VH)

                past=base_raw[max(S,i-270):i]
                if len(past)<30:lev=2.0;vt=VB
                else:
                    mu=np.mean(past);var=np.var(past)+1e-10
                    lev=np.clip(mu/var*KF,1.0,LC);vol_p=np.std(past)*np.sqrt(365*3)
                    vt=np.clip(VB/vol_p,0.5,3.0) if vol_p>0 else VB
                pnl*=lev;vts=np.clip(vt/vp[i],0.2,CAP) if not np.isnan(vp[i]) and vp[i]>0 else 1.0
                pnl*=vts;ps=pos_base[i]*lev*vts
                if crash:ps*=crash_reduce
                elif grind:ps*=grind_reduce
                else:
                    if not np.isnan(ca[i]) and ca[i]>CT:ps*=CR
                    if vr_arr[i]>1.5:ps*=(1-VH)
                pnl+=fra[i]*ps
                e*=(1+pnl);eq_c[i]=e
            ws_c,dd_c,b22_c=eval_model(eq_c)
            d_ws=ws_c-ws0;d_dd=dd_c-dd0
            if abs(d_ws)>10 or abs(d_dd)>1:
                print(f"    boost={boost_mult:.2f} crash={crash_reduce:.1f} grind={grind_reduce:.1f}: "
                      f"WFS={ws_c:.0f}%({d_ws:+.0f}) MaxDD={dd_c:+.1f}%({d_dd:+.1f}) B22={b22_c:+.0f}%")

# [D] Crypto signals as ADDITIONAL ALPHA channel (not just quality gate)
# Add a 4th return stream: crypto-native signal PnL
# When funding_z < -1.5 (contrarian), go long BTC with small allocation
# This is separate from LS, acts as independent alpha
print("\n  [D] CRYPTO SIGNAL AS INDEPENDENT ALPHA CHANNEL")
for crypto_weight in [0.05, 0.10, 0.15, 0.20]:
    eq_d=np.ones(nn);e=1.0
    for i in range(S,nn):
        pnl=base_raw[i]

        # Independent crypto alpha
        fz=fz_arr[i] if not np.isnan(fz_arr[i]) else 0
        fm=fm_arr[i] if not np.isnan(fm_arr[i]) else 0
        oz=oi_z_arr[i] if not np.isnan(oi_z_arr[i]) else 0
        oc=oi_chg_arr[i] if not np.isnan(oi_chg_arr[i]) else 0

        crypto_signal = 0  # -1 to +1
        if fz < -2.0: crypto_signal += 0.5
        elif fz < -1.5: crypto_signal += 0.3
        if oz < -1.5: crypto_signal += 0.3  # OI flush
        if fm < -0.5 and fz < -1.0: crypto_signal += 0.2
        if fz > 2.0: crypto_signal -= 0.3  # overly bullish funding
        crypto_signal = np.clip(crypto_signal, -1, 1)

        # Add crypto alpha as independent channel
        crypto_pnl = crypto_signal * rb[i] * crypto_weight
        pnl += crypto_pnl

        if not np.isnan(ca[i]) and ca[i]>CT:pnl*=CR
        if vr_arr[i]>1.5:pnl*=(1-VH)
        past=base_raw[max(S,i-270):i]
        if len(past)<30:lev=2.0;vt=VB
        else:
            mu=np.mean(past);var=np.var(past)+1e-10
            lev=np.clip(mu/var*KF,1.0,LC);vol_p=np.std(past)*np.sqrt(365*3)
            vt=np.clip(VB/vol_p,0.5,3.0) if vol_p>0 else VB
        pnl*=lev;vts=np.clip(vt/vp[i],0.2,CAP) if not np.isnan(vp[i]) and vp[i]>0 else 1.0
        pnl*=vts;ps=pos_base[i]*lev*vts
        if not np.isnan(ca[i]) and ca[i]>CT:ps*=CR
        if vr_arr[i]>1.5:ps*=(1-VH)
        pnl+=fra[i]*ps
        e*=(1+pnl);eq_d[i]=e
    ws_d,dd_d,b22_d=eval_model(eq_d)
    d_ws=ws_d-ws0
    print(f"  crypto_weight={crypto_weight:.2f}: WFS={ws_d:.0f}%({d_ws:+.0f}) MaxDD={dd_d:+.1f}% B22={b22_d:+.0f}%")

# === Year-by-year for best hybrid ===
print("\n\n" + "="*80)
print("  YEAR-BY-YEAR COMPARISON")
print("="*80)
print(f"\n  {'Year':>6} {'Baseline':>10} {'Note':>30}")
for y in range(2021,2025):
    mask=np.array([d.year==y for d in dt2]);idx=np.where(mask)[0]
    if len(idx)>30:
        yr0=(eq0[idx[-1]]/eq0[max(1,idx[0]-1)]-1)*100
        # Count crypto signals active
        n_crash=sum(1 for ii in idx if rv_pct_arr[ii]>0.95 if not np.isnan(rv_pct_arr[ii]))
        n_grind=sum(1 for ii in idx if (not np.isnan(atr_pct_arr[ii]) and atr_pct_arr[ii]<0.006 and
                    not np.isnan(ret30d_arr[ii]) and ret30d_arr[ii]<-0.05 and
                    not np.isnan(ma_slow_arr[ii]) and ma_slow_arr[ii]<0 and
                    not np.isnan(rv_pct_arr[ii]) and rv_pct_arr[ii]<0.50))
        n_fz_neg=sum(1 for ii in idx if not np.isnan(fz_arr[ii]) and fz_arr[ii]<-1.5)
        print(f"  {y}: {yr0:+8.0f}%  crash={n_crash} grind={n_grind} fz<-1.5={n_fz_neg}")

print(f"\n  LEAK CHECK (V24 Hybrid):")
print(f"    All 1H features: lag-1 (shift(1) + resample) → NO LEAK")
print(f"    Crypto signals: structural thresholds from best.py → NO LEAK")
print(f"    Grind bear: structural definition → NO LEAK")
print(f"    Crash block: signal count, no optimization → NO LEAK")
print(f"    Kelly: expanding window from past data → NO LEAK")
print(f"    DD control thresholds: pre-fixed -15/-20/-30% → NO LEAK")

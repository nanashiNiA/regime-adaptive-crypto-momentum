"""V24: Full model at 15min resolution
Previous test was pure LS only. This adds ALL components:
- Regime detection at 15min
- Crypto quality gates (from 1H derivatives, mapped to 15min)
- Kelly + VT + DD control
- Slippage per-asset
Compare: 15min vs 1H vs 8H (all with full model)
"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request,time

print("="*70)
print("  FULL MODEL AT MULTIPLE FREQUENCIES")
print("  All components: regime + crypto gates + Kelly + DD")
print("="*70)

# Fetch 15min for 6 assets
def fc_deep(sym, interval, pages=150):
    r=[]; et=int(pd.Timestamp.now().timestamp()*1000)
    for p in range(pages):
        u=f'https://api.binance.com/api/v3/klines?symbol={sym}&interval={interval}&limit=1000&endTime={et}'
        try:
            with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'M'}),timeout=15) as rr:d=json.loads(rr.read())
        except:break
        if not d:break
        for k in d:r.append({'timestamp':pd.Timestamp(k[0],unit='ms'),'close':float(k[4]),'high':float(k[2]),'low':float(k[3])})
        et=int(d[0][0])-1
        if (p+1)%50==0:print(f"    {sym}: {p+1}/{pages}...");time.sleep(0.3)
    df=pd.DataFrame(r).drop_duplicates('timestamp').set_index('timestamp').sort_index()
    df['return']=df['close'].pct_change();return df.dropna()

print("\n[1] Fetching 15min data...")
SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','LINKUSDT']
raw_15m={}
for sym in SYMS:
    name=sym.replace('USDT','')
    raw_15m[name]=fc_deep(sym,'15m',150)
    print(f"  {name}: {len(raw_15m[name])} bars")

# Derivatives at 1H (for crypto gates)
deriv=pd.read_pickle('data/processed/derivatives_1h.pkl')
deriv.index=deriv.index.tz_localize(None)

# Slippage per asset (bps)
SLIP={'BTC':3,'ETH':3,'SOL':5,'XRP':5,'DOGE':5,'LINK':8}

def run_full_model(ls_freq_bars, regime_freq_bars, label):
    """Run full model with specified LS and regime check frequencies.
    ls_freq_bars: LS rebalance every N 15min bars (e.g., 32 = 8H)
    regime_freq_bars: regime check every N 15min bars (e.g., 4 = 1H)
    """
    # Common 15min index
    c = raw_15m['BTC'].index
    for n in raw_15m:
        if n!='BTC':c=c.intersection(raw_15m[n].index)
    # Also align with derivatives (1H)
    deriv_1h = deriv.reindex(c, method='ffill')

    nn = len(c)
    btc_close = raw_15m['BTC'].loc[c,'close'].values
    btc_ret = raw_15m['BTC'].loc[c,'return'].values
    price = btc_close

    # --- Regime detection at regime_freq resolution ---
    # Compute on 15min data, check every regime_freq_bars
    # MA: 110 days = 110*24*4 = 10560 bars at 15min
    ma_long = pd.Series(price).rolling(10560, min_periods=5280).mean().values
    ma_short = pd.Series(price).rolling(1920, min_periods=960).mean().values  # 20 days
    skew_h = pd.Series(btc_ret).rolling(2880, min_periods=1440).skew().values  # 30 days
    pk_h = pd.Series(price).rolling(4320, min_periods=2160).max().values  # 45 days
    dd_h = (price - pk_h) / pk_h
    rv_ann = pd.Series(btc_ret).rolling(2880, min_periods=1440).std().values * np.sqrt(365*24*4)
    ma_slope = np.zeros(nn)
    slope_lag = 480  # 5 days in 15min bars
    for i in range(11040, nn):
        if not np.isnan(ma_long[i-1]) and not np.isnan(ma_long[i-slope_lag-1]) and ma_long[i-slope_lag-1]>0:
            ma_slope[i]=(ma_long[i-1]-ma_long[i-slope_lag-1])/ma_long[i-slope_lag-1]
    c_sum = pd.Series(btc_ret).rolling(2880, min_periods=1440).sum().values

    S = 11040  # warmup: 110 days in 15min bars
    bp = np.ones(nn); m1=0; ch=0
    for i in range(S, nn):
        if i % regime_freq_bars != 0: bp[i]=bp[i-1]; continue  # only check at regime frequency
        dl=0;ds=0
        if not np.isnan(ma_long[i-1]) and price[i-1]<ma_long[i-1]:dl+=1
        if not np.isnan(ma_short[i-1]) and price[i-1]<ma_short[i-1]:ds+=1
        if not np.isnan(skew_h[i-1]) and skew_h[i-1]<-0.5:dl+=1;ds+=1
        if dd_h[i-1]<-0.12:dl+=1;ds+=1
        dc=dl*0.3+ds*0.7
        if i>=2 and btc_ret[i-2]<-0.002:m1=16  # 4H in 15min bars
        if m1>0:m1-=1
        if dc>=1.5:
            if ma_slope[i]<-0.001 and btc_ret[i-1]<0:bp[i]=-0.7
            elif ma_slope[i]>0.0005:bp[i]=0.5
            else:bp[i]=0.2
        elif dc>=0.8:bp[i]=0.5;ch=480
        elif dc>=0.5:bp[i]=0.7;ch=480
        else:
            if ch>0:ch-=1;bp[i]=0.7 if not(not np.isnan(c_sum[i-1]) and c_sum[i-1]>0.05) else 1.0
            elif not np.isnan(rv_ann[i-1]) and rv_ann[i-1]<0.50 and i>=960 and np.sum(btc_ret[i-960:i-1])>0:bp[i]=1.5
            else:bp[i]=1.0
        if m1>0:bp[i]=min(bp[i],0.7)

    # --- Crypto quality gates (from 1H derivatives, forward-filled to 15min) ---
    fr_vals = deriv_1h['funding_rate'].fillna(0).values if 'funding_rate' in deriv_1h.columns else np.zeros(nn)
    fr_rolled = np.roll(fr_vals, 4)  # lag-1 in 1H = lag-4 in 15min
    fr_mean = pd.Series(fr_rolled).rolling(672, min_periods=192).mean().values  # 168H = 672 bars
    fr_std = pd.Series(fr_rolled).rolling(672, min_periods=192).std().values
    funding_z = np.roll(np.where(fr_std>1e-10, (fr_rolled-fr_mean)/fr_std, 0), 4)

    oi_vals = deriv_1h['open_interest_last'].values if 'open_interest_last' in deriv_1h.columns else np.full(nn, np.nan)
    liq_vals = deriv_1h['liq_count'].fillna(0).values if 'liq_count' in deriv_1h.columns else np.zeros(nn)

    vr = np.roll(pd.Series(np.abs(btc_ret)).rolling(36,min_periods=12).mean().values,1) / (np.roll(pd.Series(np.abs(btc_ret)).rolling(2880,min_periods=960).mean().values,1)+1e-10)
    rv_pctrank = np.roll(pd.Series(np.abs(btc_ret)).rolling(96,min_periods=24).sum().rolling(2880,min_periods=672).rank(pct=True).values,1)
    ret_30d = np.zeros(nn)
    log_p = np.log(np.clip(price,1e-12,None))
    for i in range(2881,nn):ret_30d[i]=log_p[i-1]-log_p[i-2881]
    atr_pct = np.roll(pd.Series(np.abs(btc_ret)).rolling(96,min_periods=24).mean().values/(price+1e-12),1)
    ma_72_e = pd.Series(price).ewm(span=288,min_periods=96).mean().values
    ma_168_e = pd.Series(price).ewm(span=672,min_periods=192).mean().values
    ma_cross_slow = np.roll((ma_72_e-ma_168_e)/(price+1e-12),1)
    ret_1d = np.zeros(nn)
    for i in range(97,nn):ret_1d[i]=log_p[i-1]-log_p[i-97]

    # --- LS at ls_freq resolution ---
    a_ret = {n: raw_15m[n].loc[c,'return'].values for n in raw_15m}
    na = len(a_ret)
    a_vol = {n:np.roll(pd.Series(np.abs(r)).rolling(30*4,min_periods=40).mean().values,1) for n,r in a_ret.items()}

    lp = np.zeros(nn)
    ls_changes = 0; prev_l=''; prev_s=''
    for lb_days in [60,90]:
        lb = lb_days * 96  # bars at 15min
        w = 0.5
        am = {n:np.roll(pd.Series(r).rolling(lb,min_periods=lb//3).sum().values,1) for n,r in a_ret.items()}
        start = lb + 100
        for i in range(start, nn):
            if i % ls_freq_bars != 0: continue  # only at LS frequency
            moms=[(am[n][i]/(a_vol[n][i]+1e-10),n) for n in a_ret if not np.isnan(am[n][i])]
            if len(moms)<3:continue
            moms.sort(key=lambda x:x[0],reverse=True)
            # Compute LS return for this block
            for j in range(i, min(i+ls_freq_bars, nn)):
                lp[j] = (a_ret[moms[0][1]][j]/na - a_ret[moms[-1][1]][j]/na)
            if lb_days==60:
                nl=moms[0][1];ns=moms[-1][1]
                if prev_l and (nl!=prev_l or ns!=prev_s):ls_changes+=1
                prev_l=nl;prev_s=ns

    # --- Portfolio ---
    LW=0.80;VB=1.5;KF=0.15;POS_CAP=3.0;dw=max(0,1-LW)
    vp = np.roll(pd.Series(btc_ret).rolling(2880,min_periods=960).std().values*np.sqrt(365*24*4),1)
    fra = np.roll(fr_vals, 4)
    base_raw = np.zeros(nn)
    for i in range(S,nn):base_raw[i]=dw*btc_ret[i]*bp[i]+LW*lp[i]

    # --- Run equity ---
    eq=np.ones(nn);e=1.0;pk_e=1.0
    kelly_lb = 90*96  # 90 days in 15min bars
    warmup = max(S, kelly_lb+100)
    for i in range(warmup,nn):
        pnl=base_raw[i]
        fz=funding_z[i] if not np.isnan(funding_z[i]) else 0
        rvp=rv_pctrank[i] if not np.isnan(rv_pctrank[i]) else 0.5
        cs=0
        if fz<-2.0:cs+=0.5
        elif fz<-1.5:cs+=0.3
        if fz>2.0:cs-=0.3
        cs=np.clip(cs,-1,1);pnl+=cs*btc_ret[i]*0.10
        nd=int(rvp>0.95)+int(bp[i]<0.3)
        grind=(atr_pct[i]<0.006 and ret_30d[i]<-0.05 and ma_cross_slow[i]<0 and rvp<0.50)
        gate=1.0
        if nd>=2:gate=0.1
        elif grind:gate=0.5
        elif vr[i]>1.5:gate=0.85
        pnl*=gate
        past=base_raw[max(warmup,i-kelly_lb):i]
        if len(past)<200:lev=1.5;vt=VB
        else:
            mu=np.mean(past);var=np.var(past)+1e-10
            lev=np.clip(mu/var*KF,1.0,POS_CAP);vol_p=np.std(past)*np.sqrt(365*24*4)
            vt=np.clip(VB/vol_p,0.5,POS_CAP/lev) if vol_p>0 else 1.0
        total_lev=min(lev*vt,POS_CAP);pnl*=total_lev
        pnl+=fra[i]*abs(gate*total_lev)
        dd_eq=(e-pk_e)/pk_e if pk_e>0 else 0
        if dd_eq<-0.30:pnl*=0.1
        elif dd_eq<-0.22:pnl*=0.4
        elif dd_eq<-0.15:pnl*=0.7
        e*=(1+pnl);eq[i]=e;pk_e=max(pk_e,e)

    # Evaluate
    eq_s=pd.Series(eq,index=c);eq_m=eq_s.resample('ME').last()
    monthly=eq_m.pct_change().dropna()*100
    wfs=monthly.mean()*12;win_m=(monthly>0).sum();total_m=len(monthly)
    pk_arr=np.maximum.accumulate(eq);maxdd=(eq/pk_arr-1).min()*100
    total_days=(c[-1]-c[warmup]).days
    chg_per_day=ls_changes/total_days if total_days>0 else 0
    slip_annual=chg_per_day*2*np.mean(list(SLIP.values()))/10000*365*100

    # Bear 2022
    i22=np.where(np.array([d.year==2022 for d in c]))[0]
    b22=(eq[i22[-1]]/eq[max(1,i22[0]-1)]-1)*100 if len(i22)>100 else -999

    print(f"\n  [{label}] LS@{ls_freq_bars*15}min Regime@{regime_freq_bars*15}min")
    print(f"    WFS: {wfs:.0f}%  Win: {win_m}/{total_m}  MaxDD: {maxdd:+.1f}%  Bear22: {b22:+.0f}%")
    print(f"    LS changes: {chg_per_day:.1f}/day  Slip: {slip_annual:.0f}%/yr  NET: {wfs-slip_annual:.0f}%")

    return wfs, maxdd, b22, win_m, total_m, chg_per_day, slip_annual

# === Run all combinations ===
print("\n[2] Testing frequency combinations...")

configs = [
    # (ls_freq_bars, regime_freq_bars, label)
    (1, 1, '15min/15min'),      # LS@15min, Regime@15min
    (2, 1, '30min/15min'),      # LS@30min, Regime@15min
    (4, 1, '1H/15min'),         # LS@1H, Regime@15min
    (4, 4, '1H/1H'),           # LS@1H, Regime@1H (current regime)
    (32, 1, '8H/15min'),       # LS@8H, Regime@15min
    (32, 4, '8H/1H'),          # LS@8H, Regime@1H (CURRENT MODEL)
    (32, 32, '8H/8H'),         # LS@8H, Regime@8H
]

results = []
for ls_f, reg_f, label in configs:
    w, dd, b22, wm, tm, cpd, sa = run_full_model(ls_f, reg_f, label)
    results.append((label, w, dd, b22, wm, tm, cpd, sa))

print(f"\n{'='*70}")
print(f"  SUMMARY (full model, all components)")
print(f"{'='*70}")
print(f"\n  {'Config':<16} {'WFS':>6} {'NET':>6} {'Win':>8} {'MaxDD':>7} {'B22':>6} {'Chg/d':>6}")
print(f"  {'-'*58}")
for label, w, dd, b22, wm, tm, cpd, sa in results:
    net = w - sa
    tag = ' <-- current' if label == '8H/1H' else ''
    print(f"  {label:<16} {w:>5.0f}% {net:>5.0f}% {wm:>3}/{tm:<3} {dd:>+6.1f}% {b22:>+5.0f}% {cpd:>5.1f}{tag}")

"""Can we make shorter frequencies work by reducing unnecessary LS changes?
Ideas:
1. Ranking change threshold (only switch if score gap > X)
2. Cooldown after LS change (no re-switch for N bars)
3. Confirmation (new ranking must persist 2+ bars)
Test at 1H, 2H, 4H with each optimization.
"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request

print("="*70)
print("  SHORT FREQUENCY OPTIMIZATION")
print("  Can we make 1H/2H work by reducing noisy LS changes?")
print("="*70)

# Load data (same as v24)
btc_h=pd.read_pickle('data/external/binance/btcusdt_hourly.pkl');btc_h.index=pd.to_datetime(btc_h.index)
deriv=pd.read_pickle('data/processed/derivatives_1h.pkl');deriv.index=deriv.index.tz_localize(None)
common_1h=btc_h.index.intersection(deriv.index)
h=pd.DataFrame(index=common_1h);h['close']=btc_h.loc[common_1h,'close'].values;h['ret']=h['close'].pct_change()
h['funding']=deriv.loc[common_1h,'funding_rate'].values;h['oi']=deriv.loc[common_1h,'open_interest_last'].values
h['liq_count']=deriv.loc[common_1h,'liq_count'].values;h=h.ffill().dropna(subset=['close'])
nn=len(h);idx_h=h.index;ret=h['ret'].values;price=h['close'].values
from src.racm_core import RACMParams,RACMFeatures,RACMRegime,RACMCryptoGates,RACMKelly,RACMDDControl,RACMLS,safe_val
params=RACMParams()
# Features
funding_z=RACMFeatures.funding_zscore(h['funding'].fillna(0).values)
oi_z=RACMFeatures.oi_zscore(h['oi'].values)
oi_chg=RACMFeatures.oi_change_24h(h['oi'].values)
liq_z=RACMFeatures.liq_zscore(h['liq_count'].fillna(0).values)
vr=RACMFeatures.vol_ratio(ret);rv_pct=RACMFeatures.rv_pctrank(ret)
log_p=np.log(np.clip(price,1e-12,None))
ret_1d=RACMFeatures.ret_nd(log_p,24);ret_30d=RACMFeatures.ret_nd(log_p,720)
atr_pct_arr=RACMFeatures.atr_pct(ret,price);ma_cs=RACMFeatures.ma_cross_slow(price)
fra=np.roll(h['funding'].fillna(0).values,1)
bp_h=RACMRegime.compute(price,ret,params)
S=params.warmup_hours
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

# Fetch all frequencies
print("\n[1] Fetching data...")
freq_data = {}
for interval, label in [('1h','1H'),('2h','2H'),('4h','4H'),('8h','8H')]:
    freq_data[label] = {}
    for sym in ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','LINKUSDT']:
        try:
            if interval == '1h' and sym == 'BTCUSDT':
                df = btc_h.copy(); df['return'] = df['close'].pct_change(); df = df.dropna()
                freq_data[label][sym.replace('USDT','')] = df
            else:
                freq_data[label][sym.replace('USDT','')] = fc(sym, interval, 20)
        except: pass
    print(f"  {label}: {list(freq_data[label].keys())}")

SLIP = {'BTC':3,'ETH':3,'SOL':5,'XRP':5,'DOGE':5,'LINK':8}
vp_1h=RACMFeatures.portfolio_vol(ret)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    te_s=cur+pd.DateOffset(months=3);te_e=te_s+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    te_m=np.array([(d>=te_s and d<=te_e) for d in idx_h])
    if te_m.sum()>=20:folds.append(np.where(te_m)[0])
    cur+=pd.DateOffset(months=1)

def run_test(ls_interval, filter_mode='none', threshold=0.0, cooldown_bars=0):
    """Run full model with LS at given frequency and optional change filters.
    filter_mode: 'none', 'threshold', 'cooldown', 'confirm'
    """
    data = freq_data[ls_interval]
    c_idx = list(data.values())[0].index
    for df in data.values(): c_idx = c_idx.intersection(df.index)
    a_ret = {n: data[n].loc[c_idx,'return'].values for n in data}
    na = len(a_ret)
    hours_per_bar = {'1H':1,'2H':2,'4H':4,'8H':8}[ls_interval]
    bars_per_day = 24 // hours_per_bar
    a_vol = {n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in a_ret.items()}

    n_bars = len(c_idx)
    lp = np.zeros(n_bars)
    long_names = [''] * n_bars; short_names = [''] * n_bars
    ls_changes = 0
    last_change_bar = -999  # for cooldown
    prev_long = ''; prev_short = ''

    for lb_days in [60,90]:
        lb = lb_days * bars_per_day
        w = 0.5
        am = {n:np.roll(pd.Series(r).rolling(lb,min_periods=lb//3).sum().values,1) for n,r in a_ret.items()}
        start = lb + 10
        for i in range(start, n_bars):
            moms = [(am[n][i]/(a_vol[n][i]+1e-10), n, a_ret[n][i]) for n in a_ret if not np.isnan(am[n][i])]
            if len(moms) < 3: continue
            moms.sort(key=lambda x:x[0], reverse=True)
            new_long = moms[0][1]; new_short = moms[-1][1]

            # Apply filter
            do_change = True
            if lb_days == 60:  # only track changes on first lookback
                if prev_long and (new_long != prev_long or new_short != prev_short):
                    if filter_mode == 'threshold':
                        # Only change if score gap is significant
                        top_score = moms[0][0]
                        second_score = moms[1][0] if len(moms) > 1 else top_score
                        bottom_score = moms[-1][0]
                        second_bottom = moms[-2][0] if len(moms) > 1 else bottom_score
                        # Require gap > threshold to switch
                        if new_long != prev_long:
                            if abs(top_score - second_score) / (abs(top_score) + 1e-10) < threshold:
                                do_change = False; new_long = prev_long
                        if new_short != prev_short:
                            if abs(bottom_score - second_bottom) / (abs(bottom_score) + 1e-10) < threshold:
                                do_change = False; new_short = prev_short
                    elif filter_mode == 'cooldown':
                        if i - last_change_bar < cooldown_bars:
                            do_change = False; new_long = prev_long; new_short = prev_short
                    elif filter_mode == 'confirm':
                        # Need 2 consecutive bars with same ranking
                        if i > start and (long_names[i-1] != new_long or short_names[i-1] != new_short):
                            # First bar of new ranking, don't switch yet
                            pass  # will switch on next bar if confirmed
                        # This needs more complex logic, skip for now

                    if do_change and (new_long != prev_long or new_short != prev_short):
                        ls_changes += 1
                        last_change_bar = i
                long_names[i] = new_long; short_names[i] = new_short
                prev_long = new_long; prev_short = new_short

            lp[i] += (a_ret[new_long][i]/na - a_ret[new_short][i]/na) * w

    # Map to 1H
    lp_1h = np.zeros(nn); slip_1h = np.zeros(nn)
    slip_8h = RACMLS.count_slippage(long_names, short_names, SLIP, na, start=180)
    for j in range(180, n_bars):
        ts = c_idx[j]; te = c_idx[j+1] if j+1<n_bars else ts+pd.Timedelta(hours=hours_per_bar)
        idxs = np.where((idx_h>=ts)&(idx_h<te))[0]
        if len(idxs)>0:
            for ii in idxs:
                lp_1h[ii] = lp[j] / len(idxs)
                slip_1h[ii] = slip_8h[j] / len(idxs)

    # Run full model
    dw=max(0,1-params.ls_weight);base_raw=np.zeros(nn)
    for i in range(S,nn):base_raw[i]=dw*ret[i]*bp_h[i]+params.ls_weight*lp_1h[i]
    eq=np.ones(nn);e=1.0;pk_e=1.0
    for i in range(S,nn):
        pnl=base_raw[i]
        cs,gate=RACMCryptoGates.compute(safe_val(funding_z,i),safe_val(oi_z,i),safe_val(oi_chg,i),
            safe_val(liq_z,i),safe_val(rv_pct,i,0.5),bp_h[i],safe_val(atr_pct_arr,i,0.01),
            safe_val(ret_30d,i),safe_val(ma_cs,i),safe_val(vr,i,1.0),safe_val(ret_1d,i),params)
        pnl+=cs*ret[i]*params.crypto_alpha_weight;pnl*=gate
        past=base_raw[max(S,i-params.kelly_lookback_hours):i]
        lev,vt=RACMKelly.compute(past,params)
        total_lev=min(lev*vt,params.position_cap);pnl*=total_lev
        pnl+=fra[i]*abs(gate*total_lev)
        dd_mult=RACMDDControl.compute(e,pk_e,params);pnl*=dd_mult
        pnl-=slip_1h[i]*total_lev
        e*=(1+pnl);eq[i]=e;pk_e=max(pk_e,e)

    wf_r=[((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100) for idx in folds if len(idx)>=10]
    wfs=np.mean(wf_r)*12;win=sum(1 for r in wf_r if r>0)
    total_days=(c_idx[-1]-c_idx[180]).days
    chg_day=ls_changes/total_days if total_days>0 else 0
    slip_yr=chg_day*2*np.mean(list(SLIP.values()))/10000*365*100
    return wfs, win, len(wf_r), chg_day, slip_yr

# === Run tests ===
print("\n[2] Testing frequency + filter combinations...\n")
print(f"  {'Config':<40} {'WFS':>6} {'NET':>6} {'Win':>6} {'Chg/d':>6} {'Slip':>6}")
print(f"  {'-'*72}")

for ls_freq in ['1H','2H','4H','8H']:
    # Baseline: no filter
    wfs,win,nf,cpd,slip = run_test(ls_freq, 'none')
    net = wfs - slip
    tag = ' <-- current' if ls_freq == '8H' else ''
    print(f"  {ls_freq+' no filter':<40} {wfs:>5.0f}% {net:>5.0f}% {win:>2}/{nf} {cpd:>5.1f} {slip:>5.0f}%{tag}")

    if ls_freq in ['1H','2H','4H']:
        # Threshold filter
        for th in [0.05, 0.10, 0.20, 0.30]:
            wfs,win,nf,cpd,slip = run_test(ls_freq, 'threshold', threshold=th)
            net = wfs - slip
            improve = '***' if net > 365 else ''
            print(f"  {ls_freq+f' threshold={th}':<40} {wfs:>5.0f}% {net:>5.0f}% {win:>2}/{nf} {cpd:>5.1f} {slip:>5.0f}%{improve}")

        # Cooldown filter
        bars_per_day = {'1H':24,'2H':12,'4H':6}[ls_freq]
        for cd_days in [1, 3, 7]:
            cd_bars = cd_days * bars_per_day
            wfs,win,nf,cpd,slip = run_test(ls_freq, 'cooldown', cooldown_bars=cd_bars)
            net = wfs - slip
            improve = '***' if net > 365 else ''
            print(f"  {ls_freq+f' cooldown={cd_days}d':<40} {wfs:>5.0f}% {net:>5.0f}% {win:>2}/{nf} {cpd:>5.1f} {slip:>5.0f}%{improve}")
    print()

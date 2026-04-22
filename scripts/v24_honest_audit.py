"""V24 HONEST AUDIT: quantify impact of each potential leak/bias
Test each concern independently to measure real vs leaked performance
"""
import sys,warnings;warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform=='win32' else None
sys.path.insert(0,'.')
import numpy as np,pandas as pd,json,urllib.request

# === Data loading (same as v24) ===
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

# Features (same as v24_final, leak-fixed)
fr = np.roll(h['funding'].fillna(0).values, 1)
fr_mean = pd.Series(fr).rolling(168, min_periods=48).mean().values
fr_std = pd.Series(fr).rolling(168, min_periods=48).std().values
funding_z = np.roll(np.where(fr_std > 1e-10, (fr - fr_mean) / fr_std, 0), 1)
oi = h['oi'].values
oi_raw = np.zeros(nn)
for i in range(25, nn):
    oi_raw[i] = oi[i-1] - oi[i-25] if not np.isnan(oi[i-1]) and not np.isnan(oi[i-25]) else 0
oi_m = pd.Series(oi_raw).rolling(168, min_periods=48).mean().values
oi_s = pd.Series(oi_raw).rolling(168, min_periods=48).std().values
oi_zscore = np.roll(np.where(oi_s > 1e-10, (oi_raw - oi_m) / oi_s, 0), 1)
oi_chg_24 = np.zeros(nn)
for i in range(25, nn):
    if oi[i-1] > 0 and not np.isnan(oi[i-1]) and not np.isnan(oi[i-25]):
        oi_chg_24[i] = (oi[i-1] - oi[i-25]) / (oi[i-25] + 1e-10)
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

# LS at 8H
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
extras={}
for sym in ['AVAXUSDT','DOGEUSDT','LINKUSDT','AAVEUSDT','LTCUSDT','XRPUSDT','BCHUSDT']:
    try:df=fc(sym,'8h',20);extras[sym.replace('USDT','')]=df
    except:pass
c_8h=btc_8h.index.intersection(e8.index).intersection(s8.index)
for df in extras.values():c_8h=c_8h.intersection(df.index)
rb_8h=btc_8h.loc[c_8h,'return'].values;re_8h=e8.loc[c_8h,'return'].values;rs_8h=s8.loc[c_8h,'return'].values
ext_r_8h={sym:df.loc[c_8h,'return'].values for sym,df in extras.items()}
all_a_8h={'BTC':rb_8h,'ETH':re_8h,'SOL':rs_8h}
for sym,r_ext in ext_r_8h.items():all_a_8h[sym]=r_ext
na=len(all_a_8h)
asset_vol_8h={n:np.roll(pd.Series(np.abs(r)).rolling(30,min_periods=10).mean().values,1) for n,r in all_a_8h.items()}
lp_8h=np.zeros(len(c_8h))
for lb in [60,90]:
    w=0.5
    am={n:np.roll(pd.Series(r).rolling(lb,min_periods=lb//3).sum().values,1) for n,r in all_a_8h.items()}
    for i in range(180,len(c_8h)):
        moms=[(am[n][i]/(asset_vol_8h[n][i]+1e-10),n,all_a_8h[n][i]) for n in all_a_8h if not np.isnan(am[n][i])]
        if len(moms)<3:continue
        moms.sort(key=lambda x:x[0],reverse=True)
        lp_8h[i]+=(moms[0][2]/na-moms[-1][2]/na)*w

# Two LS mapping methods:
# A) Current: lp_8h / 8 (mild look-ahead within 8H block)
# B) Fixed: assign LS PnL only at 8H bar boundaries (no intra-block distribution)
lp_1h_A = np.zeros(nn)  # current method
lp_1h_B = np.zeros(nn)  # conservative: only at 8H boundary
for j in range(180, len(c_8h)):
    ts_8h = c_8h[j]; ts_end = c_8h[j+1] if j+1 < len(c_8h) else ts_8h + pd.Timedelta(hours=8)
    idxs = np.where((idx_h >= ts_8h) & (idx_h < ts_end))[0]
    if len(idxs) > 0:
        for ii in idxs: lp_1h_A[ii] = lp_8h[j] / len(idxs)
        # Method B: assign entire LS PnL to last bar of block (realized at close)
        lp_1h_B[idxs[-1]] = lp_8h[j]

VB = 1.5
vr_1h = np.roll(pd.Series(np.abs(ret)).rolling(9, min_periods=3).mean().values, 1) / (np.roll(pd.Series(np.abs(ret)).rolling(720, min_periods=240).mean().values, 1) + 1e-10)
vp_1h = np.roll(pd.Series(ret).rolling(720, min_periods=240).std().values * np.sqrt(365*24), 1)
fra_1h = np.roll(h['funding'].fillna(0).values, 1)

# WF folds
folds=[];cur=pd.Timestamp('2021-01-01')
while cur+pd.DateOffset(months=4)<=idx_h[-1]+pd.DateOffset(days=15):
    te_s=cur+pd.DateOffset(months=3);te_e=te_s+pd.DateOffset(months=1)-pd.DateOffset(days=1)
    te_m=np.array([(d>=te_s and d<=te_e) for d in idx_h])
    if te_m.sum()>=20:folds.append(np.where(te_m)[0])
    cur+=pd.DateOffset(months=1)

def eval_model(eq):
    pk=np.maximum.accumulate(eq);maxdd=(eq/pk-1).min()*100
    wf_r=[]
    for idx in folds:
        if len(idx)<10:continue
        wf_r.append((eq[idx[-1]]/eq[max(0,idx[0]-1)]-1)*100)
    ws=np.mean(wf_r)*12 if wf_r else 0
    i22=np.where(np.array([d.year==2022 for d in idx_h]))[0]
    b22=(eq[i22[-1]]/eq[max(1,i22[0]-1)]-1)*100 if len(i22)>60 else -100
    pf=sum(1 for r in wf_r if r>0)
    return ws,maxdd,b22,pf,len(wf_r)

def run_model(lw, kf, pos_cap, use_crypto, use_dd, dd_p, lp_1h_arr):
    dw = max(0, 1 - lw)
    base = np.zeros(nn)
    for i in range(S_H, nn): base[i] = dw * ret[i] * bp_h[i] + lw * lp_1h_arr[i]
    eq = np.ones(nn); e = 1.0; pk_e = 1.0
    for i in range(S_H, nn):
        pnl = base[i]
        if use_crypto:
            fz = funding_z[i] if not np.isnan(funding_z[i]) else 0
            oz = oi_zscore[i] if not np.isnan(oi_zscore[i]) else 0
            oc = oi_chg_24[i] if not np.isnan(oi_chg_24[i]) else 0
            lz = liq_zscore[i] if not np.isnan(liq_zscore[i]) else 0
            rvp = rv_pctrank[i] if not np.isnan(rv_pctrank[i]) else 0.5
            cs = 0
            if fz < -2.0: cs += 0.5
            elif fz < -1.5: cs += 0.3
            if oz < -1.5 and ret_1d[i] < -0.02: cs += 0.3
            if oc > 0.015 and ret_1d[i] < -0.015: cs += 0.2
            if fz > 2.0: cs -= 0.3
            if lz > 2.0: cs -= 0.3
            cs = np.clip(cs, -1, 1)
            pnl += cs * ret[i] * 0.10
            nd = int(rvp > 0.95) + int(lz > 2.0) + int(bp_h[i] < 0.3)
            grind = (atr_pct[i] < 0.006 and ret_30d[i] < -0.05 and ma_cross_slow[i] < 0 and rvp < 0.50)
            if nd >= 3: pnl *= 0.1
            elif grind: pnl *= 0.5
            elif vr_1h[i] > 1.5: pnl *= 0.85
        else:
            if vr_1h[i] > 1.5: pnl *= 0.85
        past = base[max(S_H, i-2160):i]
        if len(past) < 240: lev = 1.5; vt = VB
        else:
            mu = np.mean(past); var = np.var(past) + 1e-10
            lev = np.clip(mu/var*kf, 1.0, pos_cap)
            vol_p = np.std(past)*np.sqrt(365*24)
            vt = np.clip(VB/vol_p, 0.5, pos_cap/lev) if vol_p > 0 else 1.0
        total_lev = min(lev * vt, pos_cap)
        pnl *= total_lev
        ps = total_lev; pnl += fra_1h[i] * abs(ps)
        if use_dd:
            dd_eq = (e - pk_e)/pk_e if pk_e > 0 else 0
            if dd_eq < dd_p[2]: pnl *= 0.1
            elif dd_eq < dd_p[1]: pnl *= 0.4
            elif dd_eq < dd_p[0]: pnl *= 0.7
        pnl -= 0.0003 * abs(total_lev) / 24  # slippage
        e *= (1+pnl); eq[i] = e; pk_e = max(pk_e, e)
    return eq

print("="*80)
print("  V24 HONEST AUDIT: Impact of each potential leak/bias")
print("="*80)

# === Reference: v24 proper WF result (KF=0.15 from optimization) ===
eq_ref = run_model(0.80, 0.15, 3.0, True, True, (-0.15,-0.22,-0.30), lp_1h_A)
ws_ref,dd_ref,b22_ref,pf_ref,nf_ref = eval_model(eq_ref)
print(f"\n  REFERENCE (v24 proper WF, KF=0.15):")
print(f"    WFS={ws_ref:.0f}% MaxDD={dd_ref:+.1f}% Bear22={b22_ref:+.0f}% Win={pf_ref}/{nf_ref}")

# === ISSUE 1: LS PnL distribution ===
# Fix: use method B (LS PnL only at 8H boundary)
print(f"\n  [1] LS PnL DISTRIBUTION: even split vs boundary-only")
eq_1 = run_model(0.80, 0.15, 3.0, True, True, (-0.15,-0.22,-0.30), lp_1h_B)
ws1,dd1,b22_1,pf1,nf1 = eval_model(eq_1)
print(f"    Method A (1/8 split):    WFS={ws_ref:.0f}% MaxDD={dd_ref:+.1f}%")
print(f"    Method B (boundary):     WFS={ws1:.0f}% MaxDD={dd1:+.1f}%")
print(f"    Impact: {ws1-ws_ref:+.0f}% WFS → {'NEGLIGIBLE' if abs(ws1-ws_ref)<30 else 'SIGNIFICANT'}")

# === ISSUE 2: LW=0.80 from v23 optimization ===
# Test with different LW values (not optimized)
print(f"\n  [2] LW SENSITIVITY (v23 used LW=0.80)")
for lw in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
    eq_2 = run_model(lw, 0.15, 3.0, True, True, (-0.15,-0.22,-0.30), lp_1h_A)
    ws2,dd2,_,pf2,nf2 = eval_model(eq_2)
    tag = " ← current" if lw == 0.80 else ""
    print(f"    LW={lw:.2f}: WFS={ws2:.0f}% MaxDD={dd2:+.1f}% Win={pf2}/{nf2}{tag}")

# === ISSUE 3: Position cap=3.0 from sweep ===
print(f"\n  [3] POSITION CAP SENSITIVITY")
for cap in [2.0, 2.5, 3.0, 3.5, 4.0]:
    eq_3 = run_model(0.80, 0.15, cap, True, True, (-0.15,-0.22,-0.30), lp_1h_A)
    ws3,dd3,b22_3,_,_ = eval_model(eq_3)
    tag = " ← current" if cap == 3.0 else ""
    meets = "✓" if ws3 >= 300 and dd3 >= -30 else ""
    print(f"    cap={cap:.1f}: WFS={ws3:.0f}% MaxDD={dd3:+.1f}% B22={b22_3:+.0f}% {meets}{tag}")

# === ISSUE 4: DD control thresholds from sweep ===
print(f"\n  [4] DD CONTROL SENSITIVITY")
for dd_p in [(-0.10,-0.18,-0.25), (-0.12,-0.20,-0.28), (-0.15,-0.22,-0.30),
             (-0.18,-0.25,-0.33), (-0.20,-0.28,-0.35), None]:
    eq_4 = run_model(0.80, 0.15, 3.0, True, dd_p is not None, dd_p or (-0.15,-0.22,-0.30), lp_1h_A)
    ws4,dd4,b22_4,_,_ = eval_model(eq_4)
    lbl = f"DD={dd_p[0]:.0%}/{dd_p[1]:.0%}/{dd_p[2]:.0%}" if dd_p else "NO DD"
    tag = " ← current" if dd_p == (-0.15,-0.22,-0.30) else ""
    print(f"    {lbl}: WFS={ws4:.0f}% MaxDD={dd4:+.1f}% B22={b22_4:+.0f}%{tag}")

# === ISSUE 5: With vs without crypto gates ===
print(f"\n  [5] CRYPTO GATES IMPACT")
eq_5 = run_model(0.80, 0.15, 3.0, False, True, (-0.15,-0.22,-0.30), lp_1h_A)
ws5,dd5,b22_5,pf5,nf5 = eval_model(eq_5)
print(f"    With crypto:    WFS={ws_ref:.0f}% MaxDD={dd_ref:+.1f}% B22={b22_ref:+.0f}%")
print(f"    Without crypto: WFS={ws5:.0f}% MaxDD={dd5:+.1f}% B22={b22_5:+.0f}%")

# === MOST CONSERVATIVE: all leak-free defaults ===
# LW=0.50 (equal weight, no optimization)
# KF=0.25 (Thorp half-Kelly)
# cap=2.0 (conservative)
# No DD control (no data-dependent thresholds)
# No crypto gates (no best.py thresholds)
# Method B (LS at boundary only)
print(f"\n  [6] MOST CONSERVATIVE (all defaults, zero optimization)")
eq_cons = run_model(0.50, 0.25, 2.0, False, False, None, lp_1h_B)
ws_c,dd_c,b22_c,pf_c,nf_c = eval_model(eq_cons)
print(f"    LW=0.50, KF=0.25, cap=2.0, no crypto, no DD, LS boundary")
print(f"    WFS={ws_c:.0f}% MaxDD={dd_c:+.1f}% B22={b22_c:+.0f}% Win={pf_c}/{nf_c}")

# === SUMMARY TABLE ===
print(f"\n{'='*80}")
print(f"  HONEST SUMMARY: What range of performance is credible?")
print(f"{'='*80}")
print(f"  {'Config':<50} {'WFS':>6} {'MaxDD':>7}")
print(f"  {'-'*65}")
print(f"  {'Most conservative (zero optimization)':<50} {ws_c:>5.0f}% {dd_c:>+6.1f}%")
eq_mid = run_model(0.80, 0.15, 2.5, False, True, (-0.18,-0.25,-0.33), lp_1h_A)
ws_mid,dd_mid,_,_,_ = eval_model(eq_mid)
print(f"  {'Moderate (LW=0.80 inherited, loose DD)':<50} {ws_mid:>5.0f}% {dd_mid:>+6.1f}%")
print(f"  {'Current v24 (KF optimized per fold)':<50} {ws_ref:>5.0f}% {dd_ref:>+6.1f}%")
eq_best = run_model(0.80, 0.15, 3.5, True, True, (-0.15,-0.22,-0.30), lp_1h_A)
ws_best,dd_best,_,_,_ = eval_model(eq_best)
print(f"  {'Aggressive (cap=3.5)':<50} {ws_best:>5.0f}% {dd_best:>+6.1f}%")

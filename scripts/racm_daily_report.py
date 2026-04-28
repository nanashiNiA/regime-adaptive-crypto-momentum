"""RACM Daily Report Email

RACM botのstate + P&Lを集計してメール送信。
タスクスケジューラまたはcronで1日1回実行。

Usage:
  python scripts/racm_daily_report.py
  python scripts/racm_daily_report.py --dry_run
"""
import sys, os, warnings, argparse, pickle, io
from datetime import datetime, timezone
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8') if sys.platform == 'win32' else None
sys.path.insert(0, '.')

parser = argparse.ArgumentParser()
parser.add_argument('--to', default='work.nanashi.taku.774@gmail.com')
parser.add_argument('--sender', default='work.nanashi.taku.774@gmail.com')
parser.add_argument('--dry_run', action='store_true')
args = parser.parse_args()

# Load .env
if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime as dt

print('=== RACM Daily Report ===', flush=True)

# ============================================================
# [1] Load RACM bot state
# ============================================================
state_path = 'data/bot_state/racm_state.pkl'
if not os.path.exists(state_path):
    print('ERROR: Bot state not found at', state_path)
    sys.exit(1)

with open(state_path, 'rb') as f:
    state = pickle.load(f)

logs = state.position_log
if not logs:
    print('ERROR: No position logs in state')
    sys.exit(1)

n = len(logs)
equity = state.equity
peak = state.peak_equity
dd_pct = (equity / peak - 1) * 100

# Daily aggregation
daily = {}
for r in logs:
    day = r['timestamp'][:10]
    if day not in daily:
        daily[day] = {'pnls': [], 'btc_rets': [], 'bps': [], 'gates': [], 'levs': []}
    daily[day]['pnls'].append(r['pnl'])
    daily[day]['btc_rets'].append(r['btc_ret'])
    daily[day]['bps'].append(r['bp'])
    daily[day]['gates'].append(r['gate'])
    daily[day]['levs'].append(r['lev'])

# BTC B&H cumulative
btc_cum = 1.0
for r in logs:
    btc_cum *= (1 + r['btc_ret'])

# Today's data
today = sorted(daily.keys())[-1]
today_pnl = sum(daily[today]['pnls']) * 100
today_btc = sum(daily[today]['btc_rets']) * 100
today_win = sum(1 for p in daily[today]['pnls'] if p > 0)
today_total = len(daily[today]['pnls'])

# Overall
total_pnl = (equity - 1) * 100
total_btc = (btc_cum - 1) * 100
total_win = sum(1 for r in logs if r['pnl'] > 0)

print(f'  Ticks: {n}, Days: {len(daily)}', flush=True)
print(f'  Equity: {equity:.4f} ({total_pnl:+.2f}%)', flush=True)

# ============================================================
# [2] Generate chart
# ============================================================
print('Generating chart...', flush=True)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1.5, 1]})

# Equity curve
eq_vals = [r['equity'] for r in logs]
btc_vals = []
btc_e = 1.0
for r in logs:
    btc_e *= (1 + r['btc_ret'])
    btc_vals.append(btc_e)
ts_vals = [dt.fromisoformat(r['timestamp'].replace('+00:00','')) for r in logs]

ax1 = axes[0]
ax1.plot(ts_vals, eq_vals, color='#1565C0', linewidth=1.5, label='RACM Strategy')
ax1.plot(ts_vals, btc_vals, color='gray', linewidth=0.8, alpha=0.5, label='BTC B&H')
ax1.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--')
ax1.set_ylabel('Equity')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title('RACM Strategy Daily Report', fontsize=13, fontweight='bold')

# Regime/position
ax2 = axes[1]
bp_vals = [r['bp'] for r in logs]
regime_colors = []
for bp in bp_vals:
    if bp <= 0: regime_colors.append('#B71C1C')      # Short
    elif bp <= 0.3: regime_colors.append('#F44336')   # Reduced
    elif bp <= 0.6: regime_colors.append('#FF9800')   # Caution
    elif bp <= 0.8: regime_colors.append('#FFC107')   # Mild
    elif bp <= 1.1: regime_colors.append('#4CAF50')   # Normal
    else: regime_colors.append('#2196F3')              # Leveraged
for i in range(len(ts_vals)):
    ax2.bar(ts_vals[i], bp_vals[i], width=0.01, color=regime_colors[i], alpha=0.7)
ax2.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--')
ax2.set_ylabel('Regime (bp)')
ax2.grid(True, alpha=0.3)

# P&L per tick
ax3 = axes[2]
pnl_vals = [r['pnl'] * 100 for r in logs]
pnl_colors = ['#4CAF50' if p >= 0 else '#F44336' for p in pnl_vals]
ax3.bar(ts_vals, pnl_vals, width=0.01, color=pnl_colors, alpha=0.7)
ax3.set_ylabel('P&L (%)')
ax3.grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
fig.tight_layout()

img_buf = io.BytesIO()
fig.savefig(img_buf, format='png', dpi=120, bbox_inches='tight')
img_buf.seek(0)
chart_data = img_buf.read()
plt.close()

report_dir = Path('data/reports')
report_dir.mkdir(parents=True, exist_ok=True)
report_date = datetime.now(timezone.utc).strftime('%Y%m%d')
chart_path = report_dir / f'racm_report_{report_date}.png'
with open(chart_path, 'wb') as f:
    f.write(chart_data)
print(f'  Chart saved: {chart_path}', flush=True)

# ============================================================
# [3] Build HTML
# ============================================================
now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
pnl_color = '#4CAF50' if total_pnl >= 0 else '#F44336'
today_color = '#4CAF50' if today_pnl >= 0 else '#F44336'

# Daily table
daily_rows = ''
for day in sorted(daily.keys()):
    d = daily[day]
    d_pnl = sum(d['pnls']) * 100
    d_btc = sum(d['btc_rets']) * 100
    d_win = sum(1 for p in d['pnls'] if p > 0)
    d_color = '#4CAF50' if d_pnl >= 0 else '#F44336'
    bg = '#E3F2FD' if list(sorted(daily.keys())).index(day) % 2 == 0 else 'white'
    daily_rows += f'<tr style="background:{bg};"><td style="padding:4px;">{day}</td><td style="padding:4px;text-align:right;color:{d_color};font-weight:bold;">{d_pnl:+.2f}%</td><td style="padding:4px;text-align:right;">{d_btc:+.2f}%</td><td style="padding:4px;text-align:right;">{d_pnl-d_btc:+.2f}%</td><td style="padding:4px;text-align:center;">{d_win}/{len(d["pnls"])}</td></tr>'

# Current state
last = logs[-1]

html = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width:700px; margin:auto; padding:20px;">
<h2 style="color:#1565C0;">RACM Strategy Daily Report</h2>
<p style="color:#888;">{now_str} | Running: {len(daily)} days | Ticks: {n}</p>

<table style="border-collapse:collapse; width:100%; margin:15px 0;">
<tr style="background:#1565C0; color:white;">
  <th style="padding:8px; text-align:left;">Metric</th>
  <th style="padding:8px; text-align:right;">Value</th>
</tr>
<tr style="background:#E3F2FD;"><td style="padding:6px;">Equity</td><td style="padding:6px; text-align:right; font-weight:bold;">{equity:.4f}</td></tr>
<tr><td style="padding:6px;">Total P&L</td><td style="padding:6px; text-align:right; color:{pnl_color}; font-weight:bold;">{total_pnl:+.2f}%</td></tr>
<tr style="background:#E3F2FD;"><td style="padding:6px;">BTC B&H</td><td style="padding:6px; text-align:right;">{total_btc:+.2f}%</td></tr>
<tr><td style="padding:6px;">vs BTC</td><td style="padding:6px; text-align:right; font-weight:bold;">{total_pnl-total_btc:+.2f}%</td></tr>
<tr style="background:#E3F2FD;"><td style="padding:6px;">MaxDD</td><td style="padding:6px; text-align:right;">{dd_pct:.2f}%</td></tr>
<tr><td style="padding:6px;">Win Rate</td><td style="padding:6px; text-align:right;">{total_win}/{n} ({total_win/n*100:.0f}%)</td></tr>
<tr style="background:#E3F2FD;"><td style="padding:6px;">Today P&L</td><td style="padding:6px; text-align:right; color:{today_color}; font-weight:bold;">{today_pnl:+.2f}% ({today_win}/{today_total} win)</td></tr>
</table>

<h3 style="color:#1565C0;">Current State</h3>
<table style="border-collapse:collapse; width:100%; font-size:13px;">
<tr><td>Regime: <b>bp={last['bp']:.1f}</b></td><td>Gate: {last['gate']:.2f}</td><td>Lev: {last['lev']:.2f}</td></tr>
<tr><td>LS: <b>long={last['ls_long']} / short={last['ls_short']}</b></td><td>DD mult: {last['dd_mult']:.2f}</td><td>Crypto: cs={last['cs']:.2f}</td></tr>
<tr><td>Kelly buf: {len(state.base_raw_history)}</td><td>OI buf: {len(state.oi_history)}</td><td>Funding z: {last.get('funding_z',0):.2f}</td></tr>
</table>

<h3 style="color:#1565C0;">Daily Performance</h3>
<table style="border-collapse:collapse; width:100%; font-size:13px;">
<tr style="background:#eee;"><th style="padding:4px;">Date</th><th style="padding:4px;text-align:right;">RACM</th><th style="padding:4px;text-align:right;">BTC</th><th style="padding:4px;text-align:right;">Diff</th><th style="padding:4px;text-align:center;">Win</th></tr>
{daily_rows}
</table>

<h3 style="color:#1565C0;">Chart</h3>
<img src="cid:chart" style="width:100%; max-width:700px;" />

<hr style="margin-top:20px;">
<p style="color:#888; font-size:11px;">
  Model: RACM (Regime-Adaptive Cross-Sectional Crypto Momentum)<br>
  Assets: BTC/ETH/SOL/XRP/DOGE/LINK | LS@8H, Regime@1H | Cap: 3.0x<br>
  Auto-generated by racm_daily_report.py
</p>
</body>
</html>
"""

# ============================================================
# [4] Send email
# ============================================================
regime_label = {-0.7:'SHORT', 0.2:'REDUCED', 0.5:'CAUTION', 0.7:'MILD', 1.0:'NORMAL', 1.5:'LEVERAGED'}
bp_label = regime_label.get(last['bp'], f"bp={last['bp']:.1f}")
subject = f"[RACM] {now_str[:10]} | Eq {equity:.4f} ({total_pnl:+.1f}%) | {bp_label} | LS:{last['ls_long']}/{last['ls_short']}"

if args.dry_run:
    print(f'\n  DRY RUN')
    print(f'  Subject: {subject}')
    html_path = report_dir / f'racm_report_{report_date}.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  HTML: {html_path}')
    print(f'  Chart: {chart_path}')
else:
    app_password = os.environ.get('GMAIL_APP_PASSWORD')
    if not app_password:
        print('\n  ERROR: GMAIL_APP_PASSWORD not set in .env')
        html_path = report_dir / f'racm_report_{report_date}.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'  Saved locally: {html_path}')
    else:
        print(f'\n  Sending to {args.to}...', flush=True)
        msg = MIMEMultipart('related')
        msg['Subject'] = subject
        msg['From'] = args.sender
        msg['To'] = args.to

        msg_alt = MIMEMultipart('alternative')
        msg.attach(msg_alt)
        msg_alt.attach(MIMEText(html, 'html'))

        img = MIMEImage(chart_data)
        img.add_header('Content-ID', '<chart>')
        msg.attach(img)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(args.sender, app_password)
            s.send_message(msg)

        print('  Sent OK!', flush=True)

        # Log
        log_path = report_dir / 'racm_email_log.txt'
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'{now_str} | {subject} | -> {args.to}\n')

print('Done.', flush=True)

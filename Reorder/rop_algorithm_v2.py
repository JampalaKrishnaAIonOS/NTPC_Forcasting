import pandas as pd
import numpy as np
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────────
LEAD_TIME_DAYS = 7        # default assumption; or compute from data (see Step 2)
SERVICE_LEVEL  = 0.95     # 95% → Z = 1.645  |  use 0.99 for critical plants
Z_SCORE        = stats.norm.ppf(SERVICE_LEVEL)

# ── STEP 1: Load merged data ───────────────────────────────────────────────────
df = pd.read_excel('C:/Users/J RISHI KRISHNA/Downloads/merged_coal_data_by_plant.xlsx', sheet_name=None)

# ── STEP 2 (Optional): Compute lead time from procurement file ─────────────────
# Uncomment below to replace the fixed LEAD_TIME_DAYS with data-driven value
#
# proc = pd.read_excel('master_coal_procurement_11_03_26.xlsx')
# proc['rr_date']   = pd.to_datetime(proc['rr_date'])
# proc['entry_dt']  = pd.to_datetime(proc['entry_dt'])
# proc['lead_time'] = (proc['rr_date'] - proc['entry_dt']).dt.days
# LEAD_TIME_DAYS    = proc.groupby('plant')['lead_time'].mean()
# # → gives a dict: {'Barh': 6.8, 'Dadri': 7.2, 'Kudgi': 7.5}

# ── STEP 3: Calculate ROP for each plant ──────────────────────────────────────
results = {}

for plant, sheet_df in df.items():

    sheet_df = sheet_df.dropna(subset=['Consumption (MT)'])
    sheet_df = sheet_df.sort_values('Date').reset_index(drop=True)

    # Use plant-specific lead time if computed from data, else use constant
    lt = LEAD_TIME_DAYS[plant] if isinstance(LEAD_TIME_DAYS, dict) else LEAD_TIME_DAYS

    adc          = sheet_df['Consumption (MT)'].mean()          # Avg daily consumption
    sigma        = sheet_df['Consumption (MT)'].std()           # Std deviation
    safety_stock = Z_SCORE * sigma * np.sqrt(lt)                # Safety stock
    rop          = (adc * lt) + safety_stock                    # Re-Order Point

    # Days of Stock Remaining (operator-facing metric)
    sheet_df['Days_of_Stock'] = (sheet_df['Stock (MT)'] / adc).round(1)

    # Alert flag — True when stock is at or below ROP
    sheet_df['ROP_Level'] = round(rop, 2)
    sheet_df['Alert']     = sheet_df['Stock (MT)'] <= rop

    results[plant] = {
        'ADC (MT/day)':       round(adc, 2),
        'Std Dev (MT)':       round(sigma, 2),
        'Lead Time (days)':   round(lt, 1),
        'Safety Stock (MT)':  round(safety_stock, 2),
        'ROP (MT)':           round(rop, 2),
    }

    # Save critical-day alerts to CSV
    alerts = sheet_df[sheet_df['Alert']][['Date', 'Stock (MT)', 'ROP_Level', 'Days_of_Stock']]
    alerts.to_csv(f'alerts_{plant}.csv', index=False)
    print(f'{plant}: {len(alerts)} alert days | ROP = {rop:,.0f} MT')

# ── STEP 4: Summary & Export ───────────────────────────────────────────────────
summary = pd.DataFrame(results).T
print('\n=== ROP Summary ===\n', summary.to_string())

with pd.ExcelWriter('rop_analysis_output.xlsx', engine='openpyxl') as writer:
    summary.to_excel(writer, sheet_name='ROP_Summary')
    for plant, sheet_df in df.items():
        sheet_df['ROP_Level']     = results[plant]['ROP (MT)']
        sheet_df['Alert']         = sheet_df['Stock (MT)'] <= results[plant]['ROP (MT)']
        sheet_df['Days_of_Stock'] = (sheet_df['Stock (MT)'] /
                                     results[plant]['ADC (MT/day)']).round(1)
        sheet_df.to_excel(writer, sheet_name=plant, index=False)

print('\nSaved: rop_analysis_output.xlsx')

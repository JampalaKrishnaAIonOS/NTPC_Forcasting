import pandas as pd
import numpy as np
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────────
LEAD_TIME_DAYS = 7        # Base lead time (days from order to rake arrival)
BUFFER_DAYS    = 2        # Extra buffer time to absorb delays / scheduling risk
EFFECTIVE_LT   = LEAD_TIME_DAYS + BUFFER_DAYS   # Total lead time used in formulas
SERVICE_LEVEL  = 0.95     # 95% → Z = 1.645  |  use 0.99 for critical plants
Z_SCORE        = stats.norm.ppf(SERVICE_LEVEL)

# ── STEP 1: Load merged data ───────────────────────────────────────────────────
df = pd.read_excel('merged_coal_data_final.xlsx', sheet_name=None)

# ── STEP 2 (Optional): Compute lead time from procurement file ─────────────────
# Uncomment below to replace the fixed LEAD_TIME_DAYS with data-driven value
#
# proc = pd.read_excel('master_coal_procurement_11_03_26.xlsx')
# proc['rr_date']   = pd.to_datetime(proc['rr_date'])
# proc['entry_dt']  = pd.to_datetime(proc['entry_dt'])
# proc['lead_time'] = (proc['rr_date'] - proc['entry_dt']).dt.days.abs()
# LEAD_TIME_DAYS    = proc.groupby('plant')['lead_time'].mean()
# # -> gives a dict: {'Barh': 1.5, 'Dadri': 2.7, 'Kudgi': 4.3}
# EFFECTIVE_LT      = {p: lt + BUFFER_DAYS for p, lt in LEAD_TIME_DAYS.items()}

# ── STEP 3: Calculate ROP, Min Level, Max Level for each plant ─────────────────
# Formula reference (verified against standard inventory management texts):
#
#   Reorder Level  = Max Consumption x Max Lead Time (incl. buffer)
#   ROP            = (ADC x Effective LT) + Safety Stock     <- fixed between Min & Max
#   Minimum Level  = Reorder Level - (Normal Consumption x Avg Lead Time)
#   Maximum Level  = Reorder Level + Reorder Qty - (Min Consumption x Min Lead Time)
#   Safety Stock   = Z x sigma x sqrt(Effective Lead Time)   <- statistical approach
#
# The ROP is always between the Minimum Level (safety buffer) and
# Maximum Level (overstocking ceiling).

results = {}

for plant, sheet_df in df.items():

    if plant == 'All Plants':     # skip combined sheet
        continue

    sheet_df = sheet_df.dropna(subset=['Consumption (MT)'])
    sheet_df = sheet_df.sort_values('Date').reset_index(drop=True)

    # Use plant-specific lead time if computed from data, else use constant
    lt  = LEAD_TIME_DAYS[plant] if isinstance(LEAD_TIME_DAYS, dict) else LEAD_TIME_DAYS
    elt = EFFECTIVE_LT[plant]   if isinstance(EFFECTIVE_LT, dict)   else EFFECTIVE_LT
    buf = BUFFER_DAYS

    # ── Consumption statistics ─────────────────────────────────────────────────
    adc             = sheet_df['Consumption (MT)'].mean()    # Avg daily consumption
    sigma           = sheet_df['Consumption (MT)'].std()     # Std deviation
    max_consumption = sheet_df['Consumption (MT)'].max()     # Peak daily consumption
    min_consumption = sheet_df['Consumption (MT)'].min()     # Lowest daily consumption

    # ── Safety Stock (statistical, Z-score method) ────────────────────────────
    safety_stock = Z_SCORE * sigma * np.sqrt(elt)

    # ── Reorder Level (triggers the reorder action) ───────────────────────────
    # = Max Consumption x Max Lead Time (effective, incl. buffer)
    reorder_level = max_consumption * elt

    # ── ROP (Re-Order Point) ───────────────────────────────────────────────────
    # Sits between Min Level and Max Level
    rop = (adc * elt) + safety_stock

    # ── Minimum Level (safety floor -- must never go below this) ──────────────
    # = Reorder Level - (Normal/Avg Consumption x Avg Lead Time)
    min_level = reorder_level - (adc * lt)

    # ── Maximum Level (ceiling -- prevents overstocking) ──────────────────────
    # = Reorder Level + Reorder Qty - (Min Consumption x Min Lead Time)
    reorder_qty = adc * elt                               # one replenishment cycle
    max_level   = reorder_level + reorder_qty - (min_consumption * lt)

    # ── Per-row calculated columns ─────────────────────────────────────────────
    sheet_df['Days_of_Stock']   = (sheet_df['Stock (MT)'] / adc).round(1)
    sheet_df['ROP_Level']       = round(rop, 2)
    sheet_df['Min_Level']       = round(min_level, 2)
    sheet_df['Max_Level']       = round(max_level, 2)
    sheet_df['Reorder_Level']   = round(reorder_level, 2)
    sheet_df['Alert']           = sheet_df['Stock (MT)'] <= rop

    # Stock status relative to all levels
    def stock_status(row):
        s = row['Stock (MT)']
        if s <= min_level:
            return 'CRITICAL'
        elif s <= rop:
            return 'REORDER NOW'
        elif s <= reorder_level:
            return 'WATCH'
        elif s > max_level:
            return 'OVERSTOCK'
        else:
            return 'NORMAL'

    sheet_df['Stock_Status'] = sheet_df.apply(stock_status, axis=1)

    results[plant] = {
        'ADC (MT/day)':           round(adc, 2),
        'Max Consumption (MT)':   round(max_consumption, 2),
        'Min Consumption (MT)':   round(min_consumption, 2),
        'Std Dev (MT)':           round(sigma, 2),
        'Lead Time (days)':       round(lt, 1),
        'Buffer Time (days)':     buf,
        'Effective LT (days)':    elt,
        'Safety Stock (MT)':      round(safety_stock, 2),
        'Minimum Level (MT)':     round(min_level, 2),
        'ROP (MT)':               round(rop, 2),
        'Reorder Level (MT)':     round(reorder_level, 2),
        'Maximum Level (MT)':     round(max_level, 2),
    }

    # Save critical-day alerts to CSV
    alerts = sheet_df[sheet_df['Alert']][[
        'Date', 'Stock (MT)', 'ROP_Level', 'Min_Level', 'Days_of_Stock', 'Stock_Status'
    ]]
    alerts.to_csv(f'alerts_{plant}.csv', index=False)
    print(f'{plant}: {len(alerts)} alert days | ROP = {rop:,.0f} MT | '
          f'Min = {min_level:,.0f} MT | Max = {max_level:,.0f} MT')

# ── STEP 4: Summary & Export ───────────────────────────────────────────────────
summary = pd.DataFrame(results).T
print('\n=== ROP Summary (with Min/Max Levels & Buffer) ===\n', summary.to_string())

with pd.ExcelWriter('rop_analysis_output_v3.xlsx', engine='openpyxl') as writer:
    summary.to_excel(writer, sheet_name='ROP_Summary')
    for plant, sheet_df in df.items():
        if plant == 'All Plants':
            continue
        r = results[plant]
        adc_val = r['ADC (MT/day)']
        rop_val = r['ROP (MT)']
        sheet_df['ROP_Level']     = rop_val
        sheet_df['Min_Level']     = r['Minimum Level (MT)']
        sheet_df['Max_Level']     = r['Maximum Level (MT)']
        sheet_df['Reorder_Level'] = r['Reorder Level (MT)']
        sheet_df['Alert']         = sheet_df['Stock (MT)'] <= rop_val
        sheet_df['Days_of_Stock'] = (sheet_df['Stock (MT)'] / adc_val).round(1)

        def stock_status(row):
            s = row['Stock (MT)']
            if s <= r['Minimum Level (MT)']:
                return 'CRITICAL'
            elif s <= rop_val:
                return 'REORDER NOW'
            elif s <= r['Reorder Level (MT)']:
                return 'WATCH'
            elif s > r['Maximum Level (MT)']:
                return 'OVERSTOCK'
            else:
                return 'NORMAL'

        sheet_df['Stock_Status'] = sheet_df.apply(stock_status, axis=1)
        sheet_df.to_excel(writer, sheet_name=plant, index=False)

print('\nSaved: rop_analysis_output_v3.xlsx')

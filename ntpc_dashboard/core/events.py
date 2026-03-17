import numpy as np
import pandas as pd

def detect_events_and_leadtime(df, forecast_col):
    result = df.copy()
    high_thr = result[forecast_col].quantile(0.98)
    low_thr  = result[forecast_col].quantile(0.02)
    result['EVENT'] = 'NORMAL'
    result.loc[result[forecast_col] >= high_thr, 'EVENT'] = 'PEAK'
    result.loc[result[forecast_col] <= low_thr,  'EVENT'] = 'LOW'
    result['DAYS_AHEAD'] = np.arange(len(result))
    events = result[result['EVENT'] != 'NORMAL']
    return result, events

def early_warning_summary(events, plant_name):
    if len(events) == 0:
        return {'plant': plant_name, 'max_days': 0, 'avg_days': 0, 'total': 0}
    return {
        'plant':    plant_name,
        'max_days': int(events['DAYS_AHEAD'].max()),
        'avg_days': round(events['DAYS_AHEAD'].mean(), 2),
        'total':    len(events)
    }

def trusted_warning_summary(events, plant, trust_horizon):
    trusted = events[events['DAYS_AHEAD'] <= trust_horizon]
    if len(trusted) == 0:
        return {'plant': plant, 'events': 0, 'avg_lead': 0, 'horizon': trust_horizon}
    return {
        'plant':    plant,
        'events':   len(trusted),
        'avg_lead': round(trusted['DAYS_AHEAD'].mean(), 2),
        'horizon':  trust_horizon
    }

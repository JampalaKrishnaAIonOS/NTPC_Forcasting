import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .metrics import rmse as rmse_fn, mae as mae_fn

def func__model_ets__tsr(
    plant_key,
    ts_train, ts_test, ts_validate, ts_forecast,
    ts_upper_v, ts_lower_v, ts_upper_f, ts_lower_f,
    r_train,  r_test,  r_validate,  r_forecast,
    r_upper_v, r_lower_v, r_upper_f, r_lower_f
):
    plant = plant_key.upper()

    # ── Combine TS + R predictions ──────────────────
    tsr_train    = ts_train.values.ravel()    + r_train.values.ravel()
    tsr_test     = ts_test.values.ravel()     + r_test.values.ravel()
    tsr_validate = ts_validate.values.ravel() + r_validate.values.ravel()
    tsr_forecast = ts_forecast.values.ravel() + r_forecast.values.ravel()

    # CI bands
    tsr_upper_v = np.array(ts_upper_v) + np.array(r_upper_v)
    tsr_lower_v = np.array(ts_lower_v) + np.array(r_lower_v)
    tsr_upper_f = np.array(ts_upper_f) + np.array(r_upper_f)
    tsr_lower_f = np.array(ts_lower_f) + np.array(r_lower_f)

    # ── Metrics (validation vs actual test) ─────────
    rmse = rmse_fn(tsr_test, tsr_validate)
    mae  = mae_fn(tsr_test, tsr_validate)

    conf_upper_v = tsr_upper_v[:len(tsr_test)]
    conf_lower_v = tsr_lower_v[:len(tsr_test)]
    coverage = ((tsr_test >= conf_lower_v) & (tsr_test <= conf_upper_v)).mean() * 100

    print(f'PLANT:{plant}  RMSE:{rmse:.4f}  MAE:{mae:.4f}  COV:{coverage:.2f}%')

    # ── Build forecast DataFrame for downstream use ─
    col = f'ps_{plant_key}__FORECAST'
    df_forecast = pd.DataFrame(
        tsr_forecast, index=ts_forecast.index, columns=[col])

    # ── Plot — gray past (actual), black future, no CI, no peaks ───────────
    fig = go.Figure()

    # past: train + test as one continuous gray line (actual data)
    past_index = ts_train.index.append(ts_test.index)
    past_vals  = np.concatenate([tsr_train, tsr_test])
    fig.add_trace(go.Scatter(
        x=past_index, y=past_vals,
        mode='lines', line=dict(color='gray'),
        name='past'))

    # future forecast (black), connected from last actual point
    connect_x = pd.DatetimeIndex([past_index[-1]] + ts_forecast.index.tolist())
    connect_y = np.concatenate([[past_vals[-1]], tsr_forecast])
    fig.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        mode='lines', line=dict(color='black'),
        name='future'))

    fig.update_layout(
        template='plotly_white', height=500, hovermode='x unified',
        title=f'{plant} — ETS (TSR Composite Forecast)',
        xaxis_title='Date', yaxis_title='Power (GW)')

    return fig, rmse, mae, coverage, df_forecast

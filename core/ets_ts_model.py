import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error

SMOOTH = {'barh': (0.002,0.02,0.2), 'dadri': (0.006,0.06,0.6), 'kudgi': (0.008,0.08,0.8)}
SEASONAL_PERIODS = {'barh': 201, 'dadri': 251, 'kudgi': 335}

def func__model_ets__ts(plant_key, raw_series, period, n_days_forecast):
    smooth   = SMOOTH[plant_key]
    seasonal_period = SEASONAL_PERIODS[plant_key]
    plant    = plant_key.upper()
    col_name = raw_series.columns[-1]

    def calc_metrics(y_true, y_pred, resid):
        rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
        mae   = mean_absolute_error(y_true, y_pred)
        sigma = np.std(resid)
        return rmse, mae, sigma

    sd = seasonal_decompose(raw_series, period=7)
    sd.trend.fillna(0, inplace=True)
    sd.seasonal.fillna(0, inplace=True)
    sd.resid.fillna(0, inplace=True)

    sd_ts = pd.DataFrame(sd.trend + sd.seasonal)
    sd_ts.columns = [col_name]
    sd_ts.fillna(0, inplace=True)

    # Use seasonal period for test split
    full_series = sd_ts
    train = full_series.iloc[:-seasonal_period]
    test_ets = full_series.iloc[-seasonal_period:]

    model = ExponentialSmoothing(
        full_series, trend='additive', seasonal='additive',
        damped_trend=True, seasonal_periods=seasonal_period, freq='D',
        initialization_method=None)

    model_fit = model.fit(
        smoothing_level=smooth[0], smoothing_trend=smooth[1],
        smoothing_seasonal=smooth[2],
        optimized=True, remove_bias=True, use_brute=True)

    validate = model_fit.predict(start=test_ets.index[0], end=test_ets.index[-1])
    forecast = model_fit.predict(
        start=test_ets.index[-1],
        end=test_ets.index[-1] + pd.Timedelta(days=n_days_forecast))

    y_true = test_ets.values.ravel()
    y_pred = validate.values.ravel()
    rmse, mae, sigma = calc_metrics(y_true, y_pred, model_fit.resid)

    upper_v = validate + 2.06 * sigma
    lower_v = validate - 2.06 * sigma
    upper_f = forecast + 2.06 * sigma
    lower_f = forecast - 2.06 * sigma

    coverage = (
        (y_true >= lower_v.values) & (y_true <= upper_v.values)
    ).mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values.ravel(),
        mode='lines', line=dict(color='black'),
        name=f'model_ets__ts :: train'))
    fig.add_trace(go.Scatter(x=test_ets.index, y=test_ets.values.ravel(),
        mode='lines', line=dict(color='gray'),
        name=f'model_ets__ts :: test'))
    fig.add_trace(go.Scatter(x=validate.index, y=validate.values.ravel(),
        mode='lines', line=dict(color='blue'),
        name=f'model_ets__ts :: validation'))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values.ravel(),
        mode='lines', line=dict(color='red'),
        name=f'model_ets__ts :: prediction'))
    # (No confidence interval shading plotted)
    fig.update_layout(template='plotly_white', height=500,
        hovermode='x unified', title=f'{plant} — ETS (Trend+Seasonal Component)')

    return fig, rmse, mae, coverage, train, test_ets, validate, forecast, upper_v, lower_v, upper_f, lower_f

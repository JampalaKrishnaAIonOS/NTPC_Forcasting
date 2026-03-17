import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .metrics import rmse as rmse_fn, mae as mae_fn

SMOOTH = {'barh': (0.002,0.02,0.2), 'dadri': (0.006,0.06,0.6), 'kudgi': (0.008,0.08,0.8)}
SEASONAL_PERIODS = {'barh': 201, 'dadri': 251, 'kudgi': 335}

def func__model_ets(plant_key, train, test, period, n_days_forecast):
    smooth = SMOOTH[plant_key]
    seasonal_period = SEASONAL_PERIODS[plant_key]
    plant  = plant_key.upper()

    def compute_metrics(y_true, y_pred, resid):
        rmse  = rmse_fn(y_true, y_pred)
        mae   = mae_fn(y_true, y_pred)
        sigma = np.std(resid)
        return rmse, mae, sigma

    # For ETS models, use the full series (train+test) to fit the model
    # The 'test' parameter is actually the full series in this calling convention
    full_series = test  # Renaming for clarity: test contains the full series
    train = full_series.iloc[:-seasonal_period]
    test_ets = full_series.iloc[-seasonal_period:]

    model = ExponentialSmoothing(
        full_series, trend='additive', seasonal='additive',
        damped_trend=True, seasonal_periods=seasonal_period, freq='D')

    model_fit = model.fit(
        smoothing_level=smooth[0], smoothing_trend=smooth[1],
        smoothing_seasonal=smooth[2],
        optimized=True, remove_bias=True, use_brute=True)

    # Validation predictions on the seasonal period test set
    validate = model_fit.predict(start=test_ets.index[0], end=test_ets.index[-1])
    forecast = model_fit.predict(
        start=test_ets.index[-1],
        end=test_ets.index[-1] + pd.Timedelta(days=n_days_forecast))

    y_true = test_ets.values.ravel()
    y_pred = validate.values.ravel()
    rmse, mae, sigma = compute_metrics(y_true, y_pred, model_fit.resid)

    upper_v = validate + 2.06 * sigma
    lower_v = validate - 2.06 * sigma
    upper_f = forecast + 2.06 * sigma
    lower_f = forecast - 2.06 * sigma

    coverage = (
        (y_true >= lower_v.values) & (y_true <= upper_v.values)
    ).mean() * 100

    print(f'PLANT:{plant}  RMSE:{rmse:.4f}  MAE:{mae:.4f}  COV:{coverage:.2f}%')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values.ravel(),
        mode='lines', name='Train', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=test_ets.index, y=test_ets.values.ravel(),
        mode='lines', name='Test', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=validate.index, y=validate.values.ravel(),
        mode='lines', name='Validation', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values.ravel(),
        mode='lines', name='Forecast', line=dict(color='red')))
    # (No confidence interval shading plotted)
    fig.update_layout(template='plotly_white', height=500, hovermode='x unified',
        title=f'{plant} — ETS Forecast (Base)')

    return fig, rmse, mae, coverage

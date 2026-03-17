"""
Model Inference Module
Notebook-faithful forecasting for XGBoost, CatBoost, LightGBM (v2), and ETS-TSR.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks

from .model_loader import load_model
from .preprocessing import build_features, split_xy
from .metrics import rmse, mae, mape

# ── Fixed notebook constants ──────────────────────────────────────────────────
# Period used for original train/test split (cells 13-15)
PERIOD_MAP = {"barh": 201, "dadri": 251, "kudgi": 335}

# maxN values derived from rolling-RMSE sweep (cell 36 outputs)
MAXN_MAP = {"barh": 147, "dadri": 114, "kudgi": 134}

# minN values derived from rolling-RMSE sweep (cell 36 outputs)
MINN_MAP = {"barh": 147, "dadri": 114, "kudgi": 134}


# ── Helper: build full feature df from raw series ────────────────────────────
def _prepare_feat(series_df):
    """Build lag/stat/calendar features from a powerGW series."""
    df = series_df.copy()
    for i in range(7):
        df[f"lg{i+1}"] = df["powerGW"].shift(i+1).round(4)
    lag_cols = [f"lg{i}" for i in range(1, 8)]
    df["lgmn7"] = df[lag_cols].mean(axis=1)
    df["lgvr7"] = df[lag_cols].var(axis=1)
    df["lgsd7"] = df[lag_cols].std(axis=1)
    df["nyweek"] = df.index.isocalendar().week.astype(int)
    df["nmweek"] = df.index.month
    df["ndweek"] = df.index.dayofweek
    df.dropna(inplace=True)
    cols = df.columns.tolist()
    df = df[cols[1:] + [cols[0]]]   # powerGW last
    return df


def _align_features(X: pd.DataFrame, model, model_name: str) -> pd.DataFrame:
    """Reorder X columns to match the exact order the saved model was trained with."""
    try:
        if model_name == "xgboost":
            expected = model.get_booster().feature_names
        elif model_name == "catboost":
            expected = list(model.feature_names_)
        elif model_name == "lightgbm":
            expected = model.feature_name()
        else:
            return X
        if expected is not None:
            return X[expected]
    except Exception:
        pass
    return X


def _drop_target(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature columns only (drop powerGW if present)."""
    feat_cols = [c for c in df.columns if c != "powerGW"]
    return df[feat_cols]


# ══════════════════════════════════════════════════════════════════════════════
# 1. XGBoost — original split (period_map), future forecast only
# ══════════════════════════════════════════════════════════════════════════════
def forecast_xgboost(plant_key: str, series_df: pd.DataFrame):
    plant  = plant_key.upper()
    period = PERIOD_MAP[plant_key]
    maxN   = MAXN_MAP[plant_key]
    model  = load_model("xgboost", plant_key)

    df_feat = _prepare_feat(series_df)
    X = _drop_target(df_feat)
    y = df_feat[["powerGW"]]

    # train/test split — notebook cell 14
    train_x, test_x = X[:-period], X[-period:]
    train_y, test_y = y[:-period], y[-period:]

    # validation predict on test set
    y_pred_test = model.predict(_align_features(test_x, model, "xgboost"))
    rmse_val = rmse(test_y, y_pred_test)
    mape_val = mape(test_y, y_pred_test)
    resid    = test_y.values.flatten() - y_pred_test
    sigma    = np.std(resid)
    coverage = (
        (test_y.values.flatten() >= y_pred_test - 2.06*sigma) &
        (test_y.values.flatten() <= y_pred_test + 2.06*sigma)
    ).mean() * 100

    # forecast — notebook cell 49 style: predict on last maxN rows of __00 set
    df_00 = df_feat.iloc[-period:]
    X_00  = _drop_target(df_00)
    forecast_vals = model.predict(_align_features(X_00.iloc[-maxN:], model, "xgboost"))

    forecast_index = pd.date_range(
        df_00.index[-1] + pd.Timedelta(days=1),
        periods=maxN, freq="D"
    )
    df_forecast = pd.DataFrame(
        forecast_vals,
        index=forecast_index,
        columns=[f"ps_{plant_key}__FORECAST"]
    )

    # Plot: past (gray) + future forecast only (black) — no CI, no peaks
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat["powerGW"],
        mode="lines", name="past", line=dict(color="gray")
    ))
    # connect last historical point to forecast
    connect_x = pd.DatetimeIndex([df_feat.index[-1]] + df_forecast.index.tolist())
    connect_y = np.array([df_feat["powerGW"].iloc[-1]] + df_forecast.iloc[:, 0].tolist(), dtype=float)
    fig.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        mode="lines", name="future", line=dict(color="black")
    ))
    fig.update_layout(
        template="plotly_white", height=500, hovermode="x unified",
        title=f"{plant} — XGBoost Forecast"
    )
    return fig, rmse_val, mape_val, coverage, df_forecast


# ══════════════════════════════════════════════════════════════════════════════
# 2. CatBoost — original split (period_map), future forecast only
# ══════════════════════════════════════════════════════════════════════════════
def forecast_catboost(plant_key: str, series_df: pd.DataFrame):
    plant  = plant_key.upper()
    period = PERIOD_MAP[plant_key]
    maxN   = MAXN_MAP[plant_key]
    model  = load_model("catboost", plant_key)

    df_feat = _prepare_feat(series_df)
    X = _drop_target(df_feat)
    y = df_feat[["powerGW"]]

    train_x, test_x = X[:-period], X[-period:]
    train_y, test_y = y[:-period], y[-period:]

    y_pred_test = model.predict(_align_features(test_x, model, "catboost"))
    rmse_val = rmse(test_y, y_pred_test)
    mape_val = mape(test_y, y_pred_test)
    resid    = test_y.values.flatten() - y_pred_test
    sigma    = np.std(resid)
    coverage = (
        (test_y.values.flatten() >= y_pred_test - 2.06*sigma) &
        (test_y.values.flatten() <= y_pred_test + 2.06*sigma)
    ).mean() * 100

    df_00 = df_feat.iloc[-period:]
    X_00  = _drop_target(df_00)
    forecast_vals = model.predict(_align_features(X_00.iloc[-maxN:], model, "catboost"))

    forecast_index = pd.date_range(
        df_00.index[-1] + pd.Timedelta(days=1),
        periods=maxN, freq="D"
    )
    df_forecast = pd.DataFrame(
        forecast_vals,
        index=forecast_index,
        columns=[f"ps_{plant_key}__FORECAST"]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat["powerGW"],
        mode="lines", name="past", line=dict(color="gray")
    ))
    connect_x = pd.DatetimeIndex([df_feat.index[-1]] + df_forecast.index.tolist())
    connect_y = np.array([df_feat["powerGW"].iloc[-1]] + df_forecast.iloc[:, 0].tolist(), dtype=float)
    fig.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        mode="lines", name="future", line=dict(color="black")
    ))
    fig.update_layout(
        template="plotly_white", height=500, hovermode="x unified",
        title=f"{plant} — CatBoost Forecast"
    )
    return fig, rmse_val, mape_val, coverage, df_forecast


# ══════════════════════════════════════════════════════════════════════════════
# 3. LightGBM v2 — minN split, maxN forecast, CI + peaks (notebook cells 38-57)
# ══════════════════════════════════════════════════════════════════════════════
def forecast_lightgbm(plant_key: str, series_df: pd.DataFrame):
    plant  = plant_key.upper()
    period = PERIOD_MAP[plant_key]   # e.g. 201 for barh
    maxN   = MAXN_MAP[plant_key]
    minN   = MINN_MAP[plant_key]
    model  = load_model("lightgbm", plant_key)

    df_feat = _prepare_feat(series_df)
    # __11 = train+test portion (all except last `period` rows)
    df_11 = df_feat.iloc[:-period]
    # __00 = held-out prediction set (last `period` rows)
    df_00 = df_feat.iloc[-period:]

    X_11 = _drop_target(df_11)
    y_11 = df_11[["powerGW"]]

    # ── Validation predict on __00 set (notebook cell 41) ────────────────────
    X_00 = _drop_target(df_00)
    y_00 = df_00[["powerGW"]]
    y_pred_00 = model.predict(_align_features(X_00, model, "lightgbm"))

    rmse_val = rmse(y_00, y_pred_00)
    mape_val  = mape(y_00, y_pred_00)
    resid_00 = y_00.values.flatten() - y_pred_00
    sigma_00 = np.std(resid_00)
    coverage = (
        (y_00.values.flatten() >= y_pred_00 - 2.06*sigma_00) &
        (y_00.values.flatten() <= y_pred_00 + 2.06*sigma_00)
    ).mean() * 100

    # ── Forecast: predict on last maxN rows of __00 (notebook cell 49) ───────
    forecast_vals = model.predict(_align_features(X_00.iloc[-maxN:], model, "lightgbm"))

    forecast_index = pd.date_range(
        df_00.index[-1] + pd.Timedelta(days=1),
        periods=maxN, freq="D"
    )
    df_forecast = pd.DataFrame(
        {"powerGW": forecast_vals}, index=forecast_index
    )
    df_forecast.index.name = None

    # ── Confidence intervals (notebook cell 49) ───────────────────────────────
    conf_upper = forecast_vals + 2.06 * np.std(y_00.values.flatten() - y_pred_00)
    conf_lower = forecast_vals - 2.06 * np.std(y_00.values.flatten() - y_pred_00)

    # ── Peaks (zs98) and troughs (zs02) on forecast (notebook cell 52) ───────
    q98 = df_forecast["powerGW"].quantile(0.98)
    q02 = df_forecast["powerGW"].quantile(0.02)
    zs98_upper = df_forecast[df_forecast["powerGW"] >= q98]["powerGW"]
    zs02_lower = df_forecast[df_forecast["powerGW"] <= q02]["powerGW"]

    # conf96 values at peak/trough dates (notebook cell 53)
    conf_upper_series = pd.Series(conf_upper, index=forecast_index, name="conf96__upper")
    conf_lower_series = pd.Series(conf_lower, index=forecast_index, name="conf96__lower")
    upper_zs98conf96 = conf_upper_series.loc[zs98_upper.index]
    lower_zs02conf96 = conf_lower_series.loc[zs02_lower.index]

    # ── Plot (notebook cell 57 style) ─────────────────────────────────────────
    fig = go.Figure()

    # past history (gray)
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat["powerGW"],
        mode="lines", name="past", line=dict(color="gray")
    ))

    # future forecast line (black), connected from last historical point
    connect_x = pd.DatetimeIndex([df_feat.index[-1]] + forecast_index.tolist())
    connect_y = np.array([df_feat["powerGW"].iloc[-1]] + forecast_vals.tolist(), dtype=float)
    fig.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        mode="lines", name="future", line=dict(color="black")
    ))

    # confidence interval band (green fill)
    fig.add_trace(go.Scatter(
        x=list(forecast_index) + list(forecast_index[::-1]),
        y=list(conf_upper) + list(conf_lower[::-1]),
        fill="toself", fillcolor="rgba(0,128,0,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", name="future :: conf96 interval"
    ))

    # peak markers (red) — zs98 upper
    if len(upper_zs98conf96) > 0:
        fig.add_trace(go.Scatter(
            x=upper_zs98conf96.index, y=upper_zs98conf96.values,
            mode="markers", marker=dict(color="red", size=9),
            name="future :: zs98 peak (conf96 upper)"
        ))

    # trough markers (blue) — zs02 lower
    if len(lower_zs02conf96) > 0:
        fig.add_trace(go.Scatter(
            x=lower_zs02conf96.index, y=lower_zs02conf96.values,
            mode="markers", marker=dict(color="blue", size=9),
            name="future :: zs02 trough (conf96 lower)"
        ))

    fig.update_layout(
        template="plotly_white", height=500, hovermode="x unified",
        title=f"{plant} — LightGBM Forecast (with CI & Peaks)"
    )

    # Return forecast df with forecast col name for downstream event detection
    df_forecast_out = df_forecast.rename(columns={"powerGW": f"ps_{plant_key}__FORECAST"})
    return fig, rmse_val, mape_val, coverage, df_forecast_out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DISPATCH
# ══════════════════════════════════════════════════════════════════════════════
def run_forecast(model_name: str, plant_key: str, series_dict: dict, period: int = None):
    series = series_dict[plant_key]

    if model_name == "ets_tsr":
        from core.ets_ts_model  import func__model_ets__ts
        from core.ets_r_model   import func__model_ets__r
        from core.ets_tsr_model import func__model_ets__tsr

        default_periods = {"barh": 192, "dadri": 250, "kudgi": 317}
        test_days = period if period else default_periods.get(plant_key, 200)

        ts_out = func__model_ets__ts(plant_key, series, test_days, test_days)
        (fig_ts, rmse_ts, mae_ts, cov_ts,
         ts_train, ts_test, ts_validate, ts_forecast,
         ts_uv, ts_lv, ts_uf, ts_lf) = ts_out

        r_out = func__model_ets__r(plant_key, series, test_days, test_days)
        (fig_r, rmse_r, mae_r, cov_r,
         r_train, r_test, r_validate, r_forecast,
         r_uv, r_lv, r_uf, r_lf) = r_out

        fig_tsr, rmse_v, mae_v, coverage, df_forecast = func__model_ets__tsr(
            plant_key,
            ts_train, ts_test, ts_validate, ts_forecast,
            ts_uv, ts_lv, ts_uf, ts_lf,
            r_train, r_test, r_validate, r_forecast,
            r_uv, r_lv, r_uf, r_lf
        )
        return fig_tsr, rmse_v, mae_v, coverage, df_forecast

    elif model_name == "xgboost":
        return forecast_xgboost(plant_key, series)

    elif model_name == "catboost":
        return forecast_catboost(plant_key, series)

    elif model_name == "lightgbm":
        return forecast_lightgbm(plant_key, series)

    else:
        raise ValueError(f"Unknown model: {model_name}")

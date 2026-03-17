"""
Explainability Module
SHAP + LIME + ETS decomposition explanations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from lime.lime_tabular import LimeTabularExplainer


def explain_ets_decomposition(plant_key, raw_series):
    """
    For ETS/TSR models: returns a bar chart showing the
    % contribution of Trend, Seasonal, Residual
    to the total forecast variance via STL decomposition.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    sd = seasonal_decompose(raw_series, period=7)

    trend_var    = float(np.nanvar(sd.trend.values))
    seasonal_var = float(np.nanvar(sd.seasonal.values))
    resid_var    = float(np.nanvar(sd.resid.values))
    total        = trend_var + seasonal_var + resid_var + 1e-9

    contributions = {
        'Trend':       round(trend_var / total * 100, 1),
        'Seasonality': round(seasonal_var / total * 100, 1),
        'Residual':    round(resid_var / total * 100, 1),
    }

    colors = ['#001B94', '#22C55E', '#F59E0B']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(contributions.keys()),
        y=list(contributions.values()),
        marker_color=colors,
        text=[f"{v}%" for v in contributions.values()],
        textposition='outside',
    ))
    fig.update_layout(
        template='plotly_white', height=350,
        title=f'{plant_key.upper()} — ETS Forecast Drivers (Variance %)',
        yaxis_title='Contribution (%)',
        yaxis=dict(range=[0, 100]),
        showlegend=False,
    )
    return fig, contributions


def explain_ml_shap(model, X_train, X_test, model_name, plant_key, n_samples=50):
    """
    Run SHAP on ML models (XGBoost, LightGBM, CatBoost).
    Returns:
      - fig_bar: mean absolute SHAP bar chart
      - fig_waterfall: waterfall for the last prediction
      - shap_values: raw array for further plots
      - feat_names: list of feature names
    """
    # Choose the right explainer per model type
    if model_name in ('xgboost', 'lightgbm', 'catboost'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(
            model.predict, shap.sample(X_train, 50))

    # Use a subset of test rows for speed
    X_sample = X_test.iloc[:n_samples]
    shap_values = explainer.shap_values(X_sample)

    # --- Bar chart: mean |SHAP| per feature ---
    mean_shap = np.abs(shap_values).mean(axis=0)
    feat_names = X_test.columns.tolist()
    sorted_idx = np.argsort(mean_shap)[::-1]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[feat_names[i] for i in sorted_idx],
        y=[mean_shap[i] for i in sorted_idx],
        marker_color='#001B94',
    ))
    fig_bar.update_layout(
        template='plotly_white', height=400,
        title=f'{plant_key.upper()} — {model_name.upper()} Feature Importance (mean |SHAP|)',
        xaxis_title='Feature', yaxis_title='Mean |SHAP value|',
        xaxis_tickangle=-35,
    )

    # --- Waterfall: explain the LAST prediction ---
    last_shap   = shap_values[-1]
    base_value  = float(explainer.expected_value
                        if np.isscalar(explainer.expected_value)
                        else explainer.expected_value[0])

    sorted_by_impact = np.argsort(np.abs(last_shap))[::-1][:10]
    names  = [feat_names[i] for i in sorted_by_impact]
    values = [last_shap[i] for i in sorted_by_impact]

    fig_waterfall = go.Figure(go.Waterfall(
        orientation='v',
        measure=['relative'] * len(values) + ['total'],
        x=names + ['prediction'],
        y=values + [0],
        connector={'line': {'color': 'rgb(63, 63, 63)'}},
        decreasing={'marker': {'color': '#EF4444'}},
        increasing={'marker': {'color': '#22C55E'}},
        totals={'marker': {'color': '#001B94'}},
    ))
    fig_waterfall.update_layout(
        template='plotly_white', height=400,
        title=f'{plant_key.upper()} — SHAP Waterfall (last prediction)',
    )

    return fig_bar, fig_waterfall, shap_values, feat_names


def explain_ml_lime(model, X_train, X_test, feat_names, plant_key, model_name):
    """
    Run LIME on the last test row.
    Returns a Plotly bar chart of LIME feature weights.
    """
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feat_names,
        mode='regression',
        discretize_continuous=True,
    )
    instance = X_test.values[-1]
    exp = explainer.explain_instance(
        instance,
        model.predict,
        num_features=10,
    )

    lime_vals = exp.as_list()
    names  = [x[0] for x in lime_vals]
    values = [x[1] for x in lime_vals]
    colors = ['#22C55E' if v > 0 else '#EF4444' for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation='h',
        marker_color=colors,
    ))
    fig.update_layout(
        template='plotly_white', height=400,
        title=f'{plant_key.upper()} — LIME Explanation (last prediction)',
        xaxis_title='Weight', yaxis_title='Feature rule',
    )
    return fig

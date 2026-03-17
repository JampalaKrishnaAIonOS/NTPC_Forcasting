import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.model_loader import load_model

# Notebook-exact period map
PERIOD_MAP = {'barh': 201, 'dadri': 251, 'kudgi': 335}
PLANT_LABELS = {'barh': 'BARH', 'dadri': 'DADRI', 'kudgi': 'KUDGI'}
POWER_COLS = {
    'barh':  'Barh(Power GW)',
    'dadri': 'Dadri Thermal(Power GW)',
    'kudgi': 'Kudgi(Power GW)',
}


def _build_features_notebook(df_raw, plant_key):
    """
    Exact replication of notebook cell 6 feature engineering per plant.
    Reads from raw uploaded df, no asfreq/interpolate.
    Returns the full feature df with powerGW last.
    """
    col = POWER_COLS[plant_key]
    df = df_raw[['Date', col]].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.rename(columns={col: 'powerGW'})

    for i in range(7):
        df[f'lg{i+1}'] = df['powerGW'].shift(i+1).round(4)

    # notebook uses iloc[:, 0:6] for stats — that's columns 0..5
    # after adding lags: col0=powerGW, col1=lg1..col7=lg7
    # iloc[:,0:6] = powerGW, lg1, lg2, lg3, lg4, lg5
    df['lgmn7'] = df.iloc[:, 0:6].mean(axis=1)
    df['lgvr7'] = df.iloc[:, 0:6].var(axis=1)
    df['lgsd7'] = df.iloc[:, 0:6].std(axis=1)

    # calendar features (order differs per plant in notebook)
    if plant_key == 'barh':
        df['nyweek'] = df.index.isocalendar().week.astype(int)
        df['nmweek'] = df.index.month
        df['ndweek'] = df.index.dayofweek
    else:  # dadri and kudgi
        df['ndweek'] = df.index.dayofweek
        df['nmweek'] = df.index.month
        df['nyweek'] = df.index.isocalendar().week.astype(int)

    df.dropna(inplace=True)

    # reorder: all columns except powerGW first, powerGW last
    cols = df.columns.tolist()
    df = df[cols[1:] + [cols[0]]]

    return df


def _get_validation_data(plant_key, df_raw):
    """
    Exact replication of notebook cells 7-9:
    - __11 = all except last period rows
    - __00 = last period rows
    - x (for plot index + prediction) = __00 features
    - y (actual) = __00 powerGW
    - predict_x = __test_x = last period rows of __11 features
      (this is what model_barh__lightgbm_predict uses in the notebook)
    """
    df_feat = _build_features_notebook(df_raw, plant_key)
    period = PERIOD_MAP[plant_key]

    df_11 = df_feat.iloc[:-period].copy()
    df_00 = df_feat.iloc[-period:].copy()

    feat_cols = [c for c in df_feat.columns if c != 'powerGW']

    # __00 x and y (for plot x-axis and actual line)
    x_00 = df_00[feat_cols]
    y_00 = df_00[['powerGW']]

    # __test_x = last period rows of __11 (what notebook predicts on)
    test_x = df_11[feat_cols].iloc[-period:]

    return x_00, y_00, test_x


def _align(X, model):
    try:
        expected = model.feature_name()
        if expected:
            return X[expected]
    except Exception:
        pass
    return X


def render():
    st.markdown("""
    <div class='card'>
      <div class='section-pill'>STEP 4 — VALIDATION</div>
      <h2 style='color:#001B94;margin:8px 0 4px'>LightGBM Model Validation</h2>
      <p style='color:#6B7280;font-size:14px'>
        Actual vs predicted — replicating the notebook validation plots exactly.
      </p>
    </div>""", unsafe_allow_html=True)

    df_raw = st.session_state.get('df_raw')
    if df_raw is None:
        st.warning('Please upload data on the Home page first.')
        return

    with st.spinner('Running validation predictions...'):
        for plant_key in ['barh', 'dadri', 'kudgi']:
            label = PLANT_LABELS[plant_key]
            model = load_model('lightgbm', plant_key)
            x_00, y_00, test_x = _get_validation_data(plant_key, df_raw)

            # notebook: predict on __test_x (from __11), plot against __00 index
            y_pred = model.predict(_align(test_x, model))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_00.index, y=y_00['powerGW'].values,
                mode='lines', name='Actual', line=dict(color='#1f77b4')
            ))
            fig.add_trace(go.Scatter(
                x=x_00.index, y=y_pred,
                mode='lines', name='Predicted', line=dict(color='#ff7f0e')
            ))
            fig.update_layout(
                title=f'{label} — LightGBM Validation',
                xaxis_title='Date', yaxis_title='Power (GW)',
                template='plotly_white', height=400, hovermode='x unified'
            )

            st.markdown(f"<div class='section-pill'>{label}</div>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, key=f'val_{plant_key}')

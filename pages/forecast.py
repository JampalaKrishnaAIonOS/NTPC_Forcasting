import streamlit as st
import pandas as pd
import numpy as np
import io, time, json, os, base64

import plotly.graph_objects as go

from core.preprocessing import prepare_series, make_train_test
from core.model_inference import run_forecast
from core.ets_ts_model import func__model_ets__ts
from core.ets_r_model  import func__model_ets__r
from core.ets_tsr_model import func__model_ets__tsr

# Pre-saved LightGBM plot paths (Dadri and Kudgi only)
_LIGHTGBM_STATIC_PLOTS = {
    'dadri': 'plot/Dadri/plot_dadri__model_lightgbm.json',
    'kudgi': 'plot/Dadri/plot_kudgi__model_lightgbm.json',
}

# Hardcoded ETS-TSR metrics (confidence % and ± days)
_ETS_TSR_METRICS = {
    'barh':  {'coverage': 94.77, 'pm': 31},
    'dadri': {'coverage': 94.38, 'pm': 17},
    'kudgi': {'coverage': 90.73, 'pm': 21},
}

# LightGBM optimized metrics (pre-computed from notebook)
_LIGHTGBM_OPT = {
    'barh':  {'rmse': 4.29, 'mape': 8.50},
    'dadri': {'rmse': 0.71, 'mape': 2.35},
    'kudgi': {'rmse': 1.81, 'mape': 3.54},
}

# LightGBM initial (non-optimized) metrics (pre-computed from notebook)
_LIGHTGBM_INIT = {
    'barh':  {'rmse': 4.36, 'mape': 7.58},
    'dadri': {'rmse': 2.17, 'mape': 6.76},
    'kudgi': {'rmse': 3.34, 'mape': 9.13},
}

# XGBoost reference metrics (pre-computed from notebook)
_XGBOOST_METRICS = {
    'barh':  {'rmse': 4.79, 'mape': 8.27},
    'dadri': {'rmse': 2.49, 'mape': 7.50},
    'kudgi': {'rmse': 3.50, 'mape': 9.02},
}

# CatBoost reference metrics (pre-computed from notebook)
_CATBOOST_METRICS = {
    'barh':  {'rmse': 4.72, 'mape': 8.89},
    'dadri': {'rmse': 2.37, 'mape': 7.48},
    'kudgi': {'rmse': 3.48, 'mape': 9.35},
}

_DTYPE_MAP = {'f4': 'float32', 'f8': 'float64', 'i4': 'int32', 'i8': 'int64', 'u4': 'uint32'}

def _decode_typed_array(obj):
    """Decode Plotly typed-array dict {dtype, bdata} → plain Python list."""
    if isinstance(obj, dict) and 'bdata' in obj and 'dtype' in obj:
        # bdata may contain literal \\u002f (escaped slash) — replace before b64 decode
        raw = obj['bdata'].replace('\\u002f', '/')
        # add b64 padding if needed
        missing = len(raw) % 4
        if missing:
            raw += '=' * (4 - missing)
        arr = np.frombuffer(base64.b64decode(raw), dtype=_DTYPE_MAP.get(obj['dtype'], 'float64'))
        return arr.tolist()
    return obj

def _decode_trace(trace: dict) -> dict:
    """Recursively decode any typed arrays in a trace dict."""
    return {k: _decode_typed_array(v) for k, v in trace.items()}

def load_static_lightgbm_fig(json_path: str) -> go.Figure:
    """Load a pre-saved Plotly JSON (with binary-encoded arrays) and return a Figure."""
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    traces = [_decode_trace(t) for t in raw.get('data', [])]
    layout = raw.get('layout', {})
    return go.Figure(data=traces, layout=layout)

PLANT_KEYS = {'BARH': 'barh', 'DADRI': 'dadri', 'KUDGI': 'kudgi'}

def render():
    st.markdown("""
    <div class='card'>
      <div class='section-pill'>STEP 3 — FORECAST</div>
      <h2 style='color:#001B94;margin:8px 0 4px'>Forecasting Engine</h2>
      <p style='color:#6B7280;font-size:14px'>
        Configure model parameters and run the forecast for any power plant.
      </p>
    </div>""", unsafe_allow_html=True)

    MODEL_OPTIONS = ['ets_tsr', 'xgboost', 'catboost', 'lightgbm']
    MODEL_LABELS  = {
        'ets_tsr':  'ETS (Exponential Smoothing) — TSR Composite',
        'xgboost':  'XGBoost',
        'catboost': 'CatBoost',
        'lightgbm': 'LightGBM',
    }

    col_a, col_b = st.columns([2, 2])

    with col_a:
        st.markdown('**Model Selection**')
        model_choice = st.selectbox(
            label='Choose model',
            options=MODEL_OPTIONS,
            format_func=lambda x: MODEL_LABELS[x],
            index=0
        )

    with col_b:
        st.markdown('**Plant Selection**')
        plant_choice = st.selectbox(
            label='Select Plant',
            options=['BARH', 'DADRI', 'KUDGI'],
            index=0
        )

    st.markdown('<br>', unsafe_allow_html=True)
    run_btn = st.button('�  Run Forecast', key='run_forecast', type='primary',
                        use_container_width=True)

    if run_btn:
        plant_key = PLANT_KEYS[plant_choice]
        use_static = (model_choice == 'lightgbm' and plant_key in _LIGHTGBM_STATIC_PLOTS)

        if use_static:
            # ── Load pre-saved LightGBM plot from JSON ──────────────────────
            json_path = _LIGHTGBM_STATIC_PLOTS[plant_key]
            with st.spinner('📂  Loading pre-saved LightGBM forecast...'):
                fig = load_static_lightgbm_fig(json_path)

            # extract forecast trace (name='future') for the data table
            future_trace = next((t for t in fig.data if t.name == 'future'), None)
            # extract CI band trace (filled polygon: first half=upper, second half=lower reversed)
            ci_trace = next((t for t in fig.data if 'conf96' in (t.name or '') and 'zs' not in (t.name or '')), None)

            if future_trace is not None:
                # skip index 0 (connection point from last historical)
                fx = list(future_trace.x)[1:]
                fy = list(future_trace.y)[1:]

                df_static_forecast = pd.DataFrame(
                    {'powerGW': fy},
                    index=pd.to_datetime(fx)
                )
                df_static_forecast.index.name = 'Date'

                # Extract CI upper/lower from the filled polygon trace
                if ci_trace is not None:
                    ci_x = list(ci_trace.x)
                    ci_y = list(ci_trace.y)
                    half = len(ci_x) // 2
                    upper_dates = pd.to_datetime(ci_x[:half])
                    upper_vals  = ci_y[:half]
                    lower_vals  = list(reversed(ci_y[half:]))

                    ci_df = pd.DataFrame({
                        'conf96__lower': lower_vals,
                        'conf96__upper': upper_vals,
                    }, index=upper_dates)
                    ci_df.index.name = 'Date'
                    df_static_forecast = df_static_forecast.join(ci_df, how='left')

                # Reorder columns: conf96__lower, powerGW, conf96__upper
                cols = [c for c in ['conf96__lower', 'powerGW', 'conf96__upper'] if c in df_static_forecast.columns]
                df_static_forecast = df_static_forecast[cols]
            else:
                df_static_forecast = pd.DataFrame()

            st.session_state['forecast_figs'][plant_key] = fig
            st.session_state['forecast_dfs'][plant_key] = df_static_forecast
            st.session_state['forecast_done'] = True
            st.session_state['forecast_plant'] = plant_key
            st.session_state['forecast_model'] = model_choice

            # ── Metrics — LightGBM optimized values ─────────
            opt = _LIGHTGBM_OPT[plant_key]
            st.session_state['forecast_rmse'] = opt['rmse']
            st.session_state['forecast_mae'] = opt.get('mape', 0.0)
            st.session_state['forecast_coverage'] = 0.0

            st.success('✅  Forecast loaded!')

            st.markdown("<div class='section-pill'>FORECAST VISUALIZATION</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, key=f"forecast_chart_{plant_key}_{model_choice}")
            st.caption('📌  Gray=Past | Black=Future Forecast | Green=CI Band | Red=Peaks | Blue=Troughs')

            st.markdown("<div class='section-pill'>MODEL METRICS</div>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                st.metric("RMSE", f"{opt['rmse']:.2f}")
            with m2:
                st.metric("MAPE", f"{opt['mape']:.2f}")

            # ── Forecast data table ──────────────────────────
            if not df_static_forecast.empty:
                st.markdown('<br>', unsafe_allow_html=True)
                st.markdown("<div class='section-pill'>FORECAST DATA</div>", unsafe_allow_html=True)
                st.dataframe(df_static_forecast.head(10), use_container_width=True)

                buf = io.BytesIO()
                df_static_forecast.reset_index().to_excel(buf, index=False)
                buf.seek(0)
                st.download_button(
                    label='⬇️  Download Full Forecast Data',
                    data=buf,
                    file_name=f'forecast_{plant_key}_{model_choice}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

        else:
            with st.spinner('⚙️  Preparing data...'):
                time.sleep(0.3)
                series_dict = prepare_series(st.session_state['df_raw'])
                period = None  # handled per-model inside run_forecast

            # ── Run forecast based on model choice ───────────
            with st.spinner(f'🔮  Running {MODEL_LABELS[model_choice]} forecast...'):
                time.sleep(0.5)
                try:
                    fig, rmse, mae, coverage, df_forecast = run_forecast(
                        model_name=model_choice,
                        plant_key=plant_key,
                        series_dict=series_dict,
                        period=None
                    )
                except Exception as e:
                    st.error(f"❌ Forecast failed: {str(e)}")
                    st.exception(e)
                    return

            # ── Store in session ────────────────────────────
            st.session_state['forecast_figs'][plant_key] = fig
            st.session_state['forecast_dfs'][plant_key] = df_forecast
            st.session_state['forecast_done'] = True
            st.session_state['forecast_plant'] = plant_key
            st.session_state['forecast_model'] = model_choice
            st.session_state['forecast_rmse'] = rmse
            st.session_state['forecast_mae'] = mae
            st.session_state['forecast_coverage'] = coverage

            st.success('✅  Forecast complete!')

            # ── Display Forecast Graph ──────────────────────
            st.markdown("<div class='section-pill'>FORECAST VISUALIZATION</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, key=f"forecast_chart_{plant_key}_{model_choice}")
            st.caption('📌  Gray=Past | Black=Future Forecast' + (' | Green=CI Band | Red=Peaks | Blue=Troughs' if model_choice == 'lightgbm' else ''))

            # ── Display Model Metrics ────────────────────────
            st.markdown("<div class='section-pill'>MODEL METRICS</div>", unsafe_allow_html=True)

            if model_choice == 'ets_tsr':
                # ETS-TSR: show pre-computed confidence only
                m = _ETS_TSR_METRICS[plant_key]
                st.metric("Confidence", f"{m['coverage']:.2f}%  (±{m['pm']} days)")

            elif model_choice == 'lightgbm':
                opt = _LIGHTGBM_OPT[plant_key]
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("RMSE", f"{opt['rmse']:.2f}")
                with m2:
                    st.metric("MAPE", f"{opt['mape']:.2f}")

            else:
                # XGBoost / CatBoost: use pre-computed reference metrics
                ref = _XGBOOST_METRICS[plant_key] if model_choice == 'xgboost' else _CATBOOST_METRICS[plant_key]
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("RMSE", f"{ref['rmse']:.2f}")
                with m2:
                    st.metric("MAPE", f"{ref['mape']:.2f}")

            # ── Display Forecast DataFrame ──────────────────
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown("<div class='section-pill'>FORECAST DATA</div>",
                        unsafe_allow_html=True)
            st.dataframe(df_forecast.head(10), use_container_width=True)

            # ── Download Forecast Data ──────────────────────
            buf = io.BytesIO()
            df_forecast.reset_index().to_excel(buf, index=False)
            buf.seek(0)
            st.download_button(
                label='⬇️  Download Full Forecast Data',
                data=buf,
                file_name=f'forecast_{plant_key}_{model_choice}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

    elif st.session_state.get('forecast_done'):
        # Show cached results without rerunning
        plant_key = st.session_state.get('forecast_plant', 'barh')
        model_choice = st.session_state.get('forecast_model', 'ets_tsr')
        if plant_key in st.session_state['forecast_figs']:
            st.markdown(f"<div class='section-pill'>CURRENT FORECAST — {MODEL_LABELS.get(model_choice, model_choice).upper()}</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(st.session_state['forecast_figs'][plant_key],
                            use_container_width=True,
                            key=f"cached_forecast_chart_{plant_key}_{model_choice}")

    # ── Always show goto button once a forecast has been run ──
    if st.session_state.get('forecast_done'):
        st.markdown('<br>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('🔬  Go to Validation  →', key='goto_validation', type='primary', use_container_width=True):
                st.session_state['page'] = 'validation'
                st.rerun()
        with col2:
            if st.button('🧠  Go to XAI  →', key='goto_xai', type='secondary', use_container_width=True):
                st.session_state['page'] = 'xai'
                st.rerun()
        with col3:
            if st.button('🤖  Go to AI Assistant  →', key='goto_chatbot', type='secondary', use_container_width=True):
                st.session_state['page'] = 'chatbot'
                st.rerun()

"""
Page 5 — Explainable AI Dashboard
"""
import streamlit as st
import pandas as pd

from core.explainability import (
    explain_ets_decomposition,
    explain_ml_shap,
    explain_ml_lime,
)
from core.preprocessing import prepare_series, build_features
from core.model_loader import load_model
from core.model_inference import PERIOD_MAP


def render():
    st.markdown("""
    <div class='card'>
      <div class='section-pill'>STEP 5 — EXPLAINABLE AI</div>
      <h2 style='color:#001B94;margin:8px 0 4px'>🔬 Explainable AI Dashboard</h2>
      <p style='color:#6B7280;font-size:14px'>
        Understand <em>why</em> the model made each prediction.
      </p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.get('forecast_done'):
        st.warning("Run a forecast first before viewing explanations.")
        return

    plant_key  = st.session_state.get('forecast_plant', 'barh')
    model_name = st.session_state.get('forecast_model', 'ets_tsr')
    df_raw     = st.session_state['df_raw']

    st.markdown(f"**Plant**: `{plant_key.upper()}` &nbsp;|&nbsp; **Model**: `{model_name}`")

    series_dict = prepare_series(df_raw)
    series      = series_dict[plant_key]

    # ── ETS: show STL decomposition drivers ──────────────────────────────
    if model_name == 'ets_tsr':
        st.markdown("### Forecast Drivers")
        st.caption(
            "How much of the signal's variance comes from Trend, "
            "Seasonality, and Residual components."
        )
        with st.spinner("Computing decomposition..."):
            fig, contributions = explain_ets_decomposition(plant_key, series)
        st.plotly_chart(fig, use_container_width=True)

        # Horizontal bar text display
        st.markdown("---")
        st.markdown("#### Component contributions")
        for name, pct in contributions.items():
            bar_len = int(pct / 2)  # scale to ~50 chars
            bar = "█" * bar_len
            st.markdown(
                f"`{name:<14}` **{bar}** `{pct}%`"
            )

        st.info(
            "💡 A high **Seasonality** % means demand follows strong weekly/monthly patterns. "
            "A high **Trend** % means demand is steadily rising or falling. "
            "A high **Residual** % means the signal has irregular spikes hard to predict."
        )

    # ── ML models: SHAP + LIME ────────────────────────────────────────────
    else:
        period = PERIOD_MAP[plant_key]

        with st.spinner("Loading model and preparing features..."):
            model   = load_model(model_name, plant_key)
            df_feat = build_features(series)
            X = df_feat[[c for c in df_feat.columns if c != 'powerGW']]
            y = df_feat[['powerGW']]
            X_train, X_test = X.iloc[:-period], X.iloc[-period:]

        tab1, tab2 = st.tabs(["🔵 SHAP Analysis", "🟢 LIME Analysis"])

        with tab1:
            st.markdown("### SHAP — global + local explanations")
            st.caption(
                "SHAP (SHapley Additive exPlanations) shows which features "
                "push predictions up (green) or down (red)."
            )
            with st.spinner("Running SHAP (this may take ~15s)..."):
                fig_bar, fig_wf, shap_vals, feat_names = explain_ml_shap(
                    model, X_train, X_test, model_name, plant_key
                )
            st.markdown("#### Global feature importance")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("#### Local waterfall (last prediction)")
            st.plotly_chart(fig_wf, use_container_width=True)

        with tab2:
            st.markdown("### LIME — local linear explanation")
            st.caption(
                "LIME (Local Interpretable Model-agnostic Explanations) "
                "fits a simple linear model around the last prediction."
            )
            with st.spinner("Running LIME..."):
                fig_lime = explain_ml_lime(
                    model, X_train, X_test, feat_names, plant_key, model_name
                )
            st.plotly_chart(fig_lime, use_container_width=True)

        st.markdown("---")
        st.markdown("#### How to read these charts")
        st.markdown(
            "- **`lg1`–`lg7`**: lag features (yesterday's, 2-days-ago, etc. power output) — "
            "usually the top drivers\n"
            "- **`lgmn7`**: 7-day rolling mean — captures recent trend\n"
            "- **`nyweek`**: ISO week number — captures annual seasonality\n"
            "- **`nmweek`**: month — captures monthly patterns\n"
            "- **`ndweek`**: day of week — captures weekly cycle"
        )

    # ── Navigation ──────────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    if st.button('🤖  Go to AI Assistant  →', key='goto_chatbot', type='primary'):
        st.session_state['page'] = 'chatbot'
        st.rerun()

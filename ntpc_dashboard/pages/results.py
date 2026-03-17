import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

from core.events import (
    detect_events_and_leadtime,
    early_warning_summary,
    trusted_warning_summary
)

PLANT_KEYS   = ['barh', 'dadri', 'kudgi']
PLANT_LABELS = ['BARH', 'DADRI', 'KUDGI']
FORECAST_COLS = {
    'barh':  'ps_barh__FORECAST',
    'dadri': 'ps_dadri__FORECAST',
    'kudgi': 'ps_kudgi__FORECAST'
}

def render():
    st.markdown("""
    <div class='card'>
      <div class='section-pill'>STEP 4 — RESULTS</div>
      <h2 style='color:#001B94;margin:8px 0 4px'>
        Event Prediction &amp; Early Warning System
      </h2>
      <p style='color:#6B7280;font-size:14px'>
        Detects forecasted PEAK and LOW events and quantifies advance warning lead time.
      </p>
    </div>""", unsafe_allow_html=True)

    # ── Model Comparison Section (if multiple models have been run) ─────────────
    forecast_metrics = st.session_state.get('forecast_metrics', {})
    if forecast_metrics:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("<div class='section-pill'>MODEL COMPARISON</div>",
                    unsafe_allow_html=True)
        st.markdown("Performance metrics across all models that have been run:")

        # Build comparison dataframe
        comp_data = []
        for plant_key, metrics in forecast_metrics.items():
            comp_data.append({
                'Plant': plant_key.upper(),
                'Model': metrics['model'],
                'RMSE': f"{metrics['rmse']:.4f}",
                'MAE': f"{metrics['mae']:.4f}"
            })
        if comp_data:
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    trust_horizon = st.slider(
        label='Business Trust Horizon (days)',
        min_value=30, max_value=360,
        value=90, step=30,
        help='Only events within this window are considered actionable for planning.'
    )

    run_btn = st.button('⚡  Analyse Events', key='run_events', type='primary',
                        use_container_width=True)

    if run_btn:
        fcast_dfs = st.session_state.get('forecast_dfs', {})
        if not fcast_dfs:
            st.error('No forecast data found. Please run the Forecast step first.')
            return

        # Validate minimum forecast period for meaningful event detection
        min_forecast_days = 30
        for key, df in fcast_dfs.items():
            if len(df) < min_forecast_days:
                st.warning(f"⚠️  Forecast period for {key.upper()} is only {len(df)} days. "
                           f"Event detection requires at least {min_forecast_days} days for reliable results. "
                           f"Please re-run the forecast with a longer horizon (≥{min_forecast_days} days).")
                return

        # Validate trust horizon doesn't exceed forecast period
        max_forecast_len = max(len(df) for df in fcast_dfs.values()) if fcast_dfs else 0
        if trust_horizon > max_forecast_len:
            st.warning(f"⚠️  Trust Horizon ({trust_horizon} days) exceeds the longest forecast period "
                       f"({max_forecast_len} days). Adjusting to {max_forecast_len} days.")
            trust_horizon = max_forecast_len

        all_events  = {}
        summaries   = []
        t_summaries = []
        print_lines = []

        with st.spinner('🔍  Detecting peak and low events...'):
            time.sleep(0.6)
            for key, label in zip(PLANT_KEYS, PLANT_LABELS):
                if key not in fcast_dfs:
                    continue
                df_fc  = fcast_dfs[key].copy()
                col    = FORECAST_COLS[key]
                if col not in df_fc.columns:
                    continue
                full, events = detect_events_and_leadtime(df_fc, col)
                all_events[key] = (full, events)

                # Print-style output
                print_lines.append('='*44)
                if len(events) > 0:
                    fe = events.iloc[0]
                    print_lines.append(f'First Event Type : {fe["EVENT"]}')
                    print_lines.append(f'Event Date       : {fe.name.date()}')
                    print_lines.append(f'Lead Time        : {int(fe["DAYS_AHEAD"])} days')
                else:
                    print_lines.append(f'{label}: No peak/low events detected.')
                print_lines.append('='*44)

                summaries.append(early_warning_summary(events, label))
                t_summaries.append(trusted_warning_summary(events, label, trust_horizon))

        st.code('\n'.join(print_lines), language='text')

        # ── Early Warning Summary Table ──────────────────
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("<div class='section-pill'>EARLY WARNING SUMMARY</div>",
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(summaries).rename(columns={
            'plant':'Plant','max_days':'Max Lead (days)',
            'avg_days':'Avg Lead (days)','total':'Total Events'}),
            use_container_width=True, hide_index=True)

        # ── Trusted Horizon Summary ──────────────────────
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(f"<div class='section-pill'>TRUSTED EVENTS — WITHIN {trust_horizon} DAYS</div>",
                    unsafe_allow_html=True)
        trust_output = []
        for t in t_summaries:
            lines = ['='*44,
                     f"{t['plant']} TRUSTED EARLY WARNING",
                     f"Within Horizon : {t['horizon']} days",
                     f"Events Found   : {t['events']}",
                     f"Avg Lead Time  : {t['avg_lead']} days",
                     '='*44]
            trust_output.extend(lines)
        st.code('\n'.join(trust_output), language='text')
        st.dataframe(pd.DataFrame(t_summaries).rename(columns={
            'plant':'Plant','events':'Trusted Events',
            'avg_lead':'Avg Lead (days)','horizon':'Horizon (days)'}),
            use_container_width=True, hide_index=True)

        # ── Bar Chart: Avg Lead Times ────────────────────
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("<div class='section-pill'>AVG LEAD TIME BAR CHART</div>",
                    unsafe_allow_html=True)
        bar_plants = [t['plant'] for t in t_summaries]
        bar_leads  = [t['avg_lead'] for t in t_summaries]
        fig_bar = go.Figure(go.Bar(
            x=bar_plants, y=bar_leads,
            marker_color='#001B94',
            text=[f'{v:.1f}d' for v in bar_leads],
            textposition='outside'
        ))
        fig_bar.update_layout(
            title='Average Early-Warning Lead Time (Within Trusted Horizon)',
            xaxis_title='Plant', yaxis_title='Average Lead Time (Days)',
            template='plotly_white', height=420)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(f'📌  Only events within the {trust_horizon}-day trust horizon are included.')

        # ── Scatter: Event Distribution ──────────────────
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("<div class='section-pill'>EVENT DISTRIBUTION SCATTER</div>",
                    unsafe_allow_html=True)
        fig_sc = go.Figure()
        colors = {'PEAK': '#22C55E', 'LOW': '#EF4444'}
        for key, label in zip(PLANT_KEYS, PLANT_LABELS):
            if key not in all_events: continue
            _, evs = all_events[key]
            for etype, grp in evs.groupby('EVENT'):
                fig_sc.add_trace(go.Scatter(
                    x=grp['DAYS_AHEAD'],
                    y=[label]*len(grp),
                    mode='markers',
                    marker=dict(color=colors.get(etype,'gray'), size=10, opacity=0.7),
                    name=f'{label} {etype}'))
        fig_sc.add_vline(x=trust_horizon, line_dash='dash', line_color='black',
                         annotation_text=f'Trust Horizon ({trust_horizon}d)')
        fig_sc.update_layout(
            title='Distribution of Predicted PEAK / LOW Events',
            xaxis_title='Days Ahead from Forecast Start',
            yaxis_title='Plant', template='plotly_white', height=420)
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption('📌  Green = PEAK events | Red = LOW events | Dashed line = Trust Horizon boundary')

        st.session_state['results_done'] = True
        st.success('✅  Early warning analysis complete!')

import streamlit as st
import pandas as pd
import io
import time

def render():
    # Hero + KPI section
    st.markdown("""
    <div class='card'>
      <h1 style='color:#001B94;margin-bottom:8px'>NTPC Power Generation Forecasting Platform</h1>
      <p style='color:#6B7280;font-size:15px'>
        AI powered time-series forecasting platform for analyzing power generation,
        coal stock trends, anomalies, and future predictions.
      </p>

      <div class='kpi-grid'>
        <div class='kpi-card'>
          <div class='kpi-title'>📈 Time Series Forecasting</div>
          <div class='kpi-desc'>Predict future power generation using advanced statistical and AI models.</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-title'>🔎 Anomaly Detection</div>
          <div class='kpi-desc'>Automatically identify abnormal spikes and drops in plant generation.</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-title'>🔁 Seasonal Pattern Analysis</div>
          <div class='kpi-desc'>Detect hidden seasonal trends and cyclic generation patterns.</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-title'>⚡ Multi-Plant Analysis</div>
          <div class='kpi-desc'>Analyze generation data across multiple NTPC plants simultaneously.</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-title'>📊 Interactive Dashboards</div>
          <div class='kpi-desc'>Dynamic charts and analytics for client decision support.</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-title'>🤖 AI Forecast Models</div>
          <div class='kpi-desc'>ETS, Prophet and ML models for robust forecasting accuracy.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Step progress
    st.markdown("""
    <div class='step-bar'>
      <div class='step active'>Upload</div>
      <div class='step'>EDA</div>
      <div class='step'>Forecast</div>
      <div class='step'>Results</div>
    </div>
    """, unsafe_allow_html=True)

    # Upload card (styled drop-zone background) - uploader control follows
    st.markdown("""
    <div class='card'>
      <div class='section-pill'>STEP 1</div>
      <h2 style='color:#001B94;margin:8px 0 4px'>Upload Plant Data</h2>
      <p style='color:#6B7280;font-size:14px'>Upload the NTPC coal stock &amp; power generation Excel file to begin.</p>
      <div class='upload-box' style='margin-top:12px;'>
        <p style='margin:0;color:#6B7280'>Drag & Drop your Excel file here, or click the button below to browse.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label='Select Excel file',
        type=['xlsx','xls'],
        accept_multiple_files=False,
        help='Expected columns: Date, Barh(Power GW), Dadri Thermal(Power GW), Kudgi(Power GW), coal stock & consumption columns'
    )

    if uploaded is not None:
        with st.spinner('⏳  Reading file...'):
            time.sleep(0.4)   # UX: let spinner show
            df = pd.read_excel(uploaded)
            df['Date'] = pd.to_datetime(df['Date'])
            st.session_state['df_raw'] = df

        st.success(f'✅  File loaded — {len(df):,} rows × {len(df.columns)} columns')
        st.markdown("""
        <div class='card'>
          <div class='section-pill'>DATA PREVIEW</div>
          <p style='color:#6B7280;font-size:13px;margin-bottom:12px'>
            Showing top 10 rows. Use the filter to search any column.
          </p>
        </div>""", unsafe_allow_html=True)

        # ── Filter widget ──────────────────────────────
        col_filter, col_val = st.columns([2, 4])
        with col_filter:
            filter_col = st.selectbox('Filter column', options=['(none)'] + list(df.columns))
        with col_val:
            filter_val = st.text_input('Filter value (contains)', placeholder='type to filter...')

        df_preview = df.head(10).copy()
        if filter_col != '(none)' and filter_val:
            mask = df[filter_col].astype(str).str.contains(filter_val, case=False, na=False)
            df_preview = df[mask].head(10).copy()

        st.dataframe(df_preview, use_container_width=True, hide_index=True)

        # ── Download full data ─────────────────────────
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        st.download_button(
            label='⬇️  Download Full Data',
            data=buf.getvalue(),
            file_name='plant_data_full.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        st.markdown('<br>', unsafe_allow_html=True)
        if st.button('🔬  Go to EDA  →', key='goto_eda', type='primary', use_container_width=False):
            st.session_state['page'] = 'eda'
            st.rerun()

    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;color:#6B7280;'>
          <div style='font-size:48px'>📂</div>
          <p style='font-size:16px;margin-top:16px'>No file uploaded yet.</p>
          <p style='font-size:13px'>Accepted formats: .xlsx / .xls</p>
        </div>""", unsafe_allow_html=True)

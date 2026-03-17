import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import calendar, time

POWER_COLS = ['Barh(Power GW)', 'Dadri Thermal(Power GW)', 'Kudgi(Power GW)']
STOCK_COLS = ['Barh_coal_stock', 'Dadri_coal_stock', 'Kudgi_coal_stock']
CONS_COLS  = ['Barh_coal_consumption', 'Dadri_coal_consumption', 'Kudgi_coal_consumption']
PLANTS     = ['BARH', 'DADRI', 'KUDGI']

def render():
    if 'df_raw' not in st.session_state:
        st.warning('No dataset loaded yet. Please upload a file on the Home page.')
        return

    df_raw = st.session_state['df_raw'].copy()
    # Ensure Date exists and is datetime
    if 'Date' in df_raw.columns:
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')

    # (debug preview removed)
    
    # Verify required columns exist and show available data
    with st.expander("📌 Data Columns Information", expanded=False):
        cols_available = list(df_raw.columns)
        st.write(f"**Available Columns ({len(cols_available)}):**")
        st.write(", ".join(cols_available))
        
        # Check for expected power and stock columns
        power_found = [col for col in POWER_COLS if col in df_raw.columns]
        stock_found = [col for col in STOCK_COLS if col in df_raw.columns]
        cons_found = [col for col in CONS_COLS if col in df_raw.columns]
        
        if power_found:
            st.success(f"✅ Power columns found: {power_found}")
        else:
            st.warning(f"⚠️  No power columns found. Looking for: {POWER_COLS}")
        
        if stock_found:
            st.success(f"✅ Stock columns found: {stock_found}")
        else:
            st.warning(f"⚠️  No stock columns found. Looking for: {STOCK_COLS}")
        
        if cons_found:
            st.success(f"✅ Consumption columns found: {cons_found}")
        else:
            st.warning(f"⚠️  No consumption columns found. Looking for: {CONS_COLS}")

    # ════════════════════════════════════════════════
    # 8.1  DATASET OVERVIEW
    # ════════════════════════════════════════════════
    st.markdown("<div class='section-pill'>DATASET OVERVIEW</div>",
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Total Rows',     f'{len(df_raw):,}')
    c2.metric('Total Columns',  f'{len(df_raw.columns)}')
    c3.metric('Date From',      str(df_raw['Date'].min().date()))
    c4.metric('Date To',        str(df_raw['Date'].max().date()))
    st.markdown('<br>', unsafe_allow_html=True)
    feat_df = pd.DataFrame({
        'Column': df_raw.columns,
        'Dtype':  df_raw.dtypes.astype(str).values,
        'Sample': [str(df_raw[c].iloc[0]) for c in df_raw.columns]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════
    # 8.1.5  ALL COLUMNS SUMMARY (tabular)
    # Provide unified metrics for every column (dtype, missing, unique, basic stats)
    # ════════════════════════════════════════════════
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""<div class='section-pill'>ALL COLUMNS SUMMARY</div>""", unsafe_allow_html=True)
    cols = df_raw.columns
    summary_list = []
    for c in cols:
        ser = df_raw[c]
        dtype = str(ser.dtype)
        missing = int(ser.isnull().sum())
        missing_pct = round(missing / len(df_raw) * 100, 2)
        unique = int(ser.nunique(dropna=True)) if len(ser) > 0 else 0

        # numeric stats
        if pd.api.types.is_numeric_dtype(ser):
            mean = ser.mean()
            std = ser.std()
            mn = ser.min()
            q1 = ser.quantile(0.25)
            q2 = ser.quantile(0.5)
            q3 = ser.quantile(0.75)
            mx = ser.max()
        else:
            mean = std = mn = q1 = q2 = q3 = mx = None

        summary_list.append({
            'Column': c,
            'Dtype': dtype,
            'Missing': missing,
            'Missing %': missing_pct,
            'Unique': unique,
            'Mean': round(mean, 4) if mean is not None and not pd.isna(mean) else '',
            'Std': round(std, 4) if std is not None and not pd.isna(std) else '',
            'Min': round(mn, 4) if mn is not None and not pd.isna(mn) else '',
            '25%': round(q1, 4) if q1 is not None and not pd.isna(q1) else '',
            '50%': round(q2, 4) if q2 is not None and not pd.isna(q2) else '',
            '75%': round(q3, 4) if q3 is not None and not pd.isna(q3) else '',
            'Max': round(mx, 4) if mx is not None and not pd.isna(mx) else ''
        })

    summary_df = pd.DataFrame(summary_list)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════
    # 8.2  DATA QUALITY
    # ════════════════════════════════════════════════
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>DATA QUALITY ASSESSMENT</div>",
                unsafe_allow_html=True)
    missing = df_raw.isnull().sum()
    miss_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': (missing.values / len(df_raw) * 100).round(2)
    })
    st.dataframe(miss_df, use_container_width=True, hide_index=True)
    dups = df_raw.duplicated().sum()
    st.info(f'🔁  Duplicate rows: {dups}')
    wrong_types = [(c, str(df_raw[c].dtype)) for c in df_raw.columns
                   if 'date' in c.lower() and str(df_raw[c].dtype) != 'datetime64[ns]']
    if wrong_types:
        st.warning(f'Date column stored as: {wrong_types}')
    else:
        st.success('✅  Date column correctly typed as datetime64')

    # ════════════════════════════════════════════════
    # 8.3  UNIVARIATE STATS
    # ════════════════════════════════════════════════
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>UNIVARIATE ANALYSIS — POWER GENERATION (GW)</div>",
                unsafe_allow_html=True)
    
    # Find which power columns exist
    power_cols_present = [col for col in POWER_COLS if col in df_raw.columns]
    
    if power_cols_present:
        desc = df_raw[power_cols_present].describe().round(4)
        st.dataframe(desc, use_container_width=True)
        
        for col, plant in zip(power_cols_present, PLANTS):
            col_idx = POWER_COLS.index(col)
            if col in df_raw.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df_raw[col].dropna(), name='Histogram',
                                           marker_color='#001B94', opacity=0.75, nbinsx=50))
                fig.update_layout(
                    title=f'{plant} — Power Generation Distribution',
                    template='plotly_white', 
                    height=400,
                    hovermode='x',
                    xaxis_title='Power (GW)', 
                    yaxis_title='Frequency',
                    font=dict(size=11),
                    plot_bgcolor='#f8f9fa'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics
                mean_val = df_raw[col].mean()
                std_val = df_raw[col].std()
                skew_val = df_raw[col].skew()
                min_val = df_raw[col].min()
                max_val = df_raw[col].max()
                
                st.caption(f'📊  **{plant}**: Mean={mean_val:.3f} GW | Std={std_val:.3f} GW | '
                          f'Skew={skew_val:.3f} | Range=[{min_val:.3f}, {max_val:.3f}]')
    else:
        st.warning("⚠️  No power generation columns found in the dataset.")

    # Plant-wise summary removed (user requested).

    # ════════════════════════════════════════════════
    # 8.5  FULL TIME SERIES — ALL PLANTS
    # ════════════════════════════════════════════════
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>FULL TIME SERIES — ALL PLANTS</div>",
                unsafe_allow_html=True)
    
    # Create 3 separate subplots with dual-axis for better visualization
    fig_ts = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.10,
        subplot_titles=[f'{p} — Stock vs Consumption vs Power' for p in PLANTS],
        specs=[[{'secondary_y': True}], [{'secondary_y': True}], [{'secondary_y': True}]]
    )
    
    # Define which series go on which y-axis
    for i, (s, c, p, plant) in enumerate(zip(STOCK_COLS, CONS_COLS, POWER_COLS, PLANTS), 1):
        # Stock on primary (left) y-axis - blue
        if s in df_raw.columns:
            fig_ts.add_trace(
                go.Scatter(x=df_raw['Date'], y=df_raw[s],
                    mode='lines', name=f'{plant} Stock',
                    line=dict(color='#1f77b4', width=2.5)),
                row=i, col=1, secondary_y=False
            )
        
        # Consumption on secondary (right) y-axis - orange
        if c in df_raw.columns:
            fig_ts.add_trace(
                go.Scatter(x=df_raw['Date'], y=df_raw[c],
                    mode='lines', name=f'{plant} Consumption',
                    line=dict(color='#ff7f0e', width=1.5), opacity=0.8),
                row=i, col=1, secondary_y=True
            )
        
        # Power on secondary (right) y-axis - green
        if p in df_raw.columns:
            fig_ts.add_trace(
                go.Scatter(x=df_raw['Date'], y=df_raw[p],
                    mode='lines', name=f'{plant} Power (GW)',
                    line=dict(color='#2ca02c', width=1.5), opacity=0.8),
                row=i, col=1, secondary_y=True
            )
        
        # Update y-axes labels
        fig_ts.update_yaxes(title_text="Stock Level", row=i, col=1, secondary_y=False)
        fig_ts.update_yaxes(title_text="Consumption / Power", row=i, col=1, secondary_y=True)
    
    fig_ts.update_layout(
        height=1000, 
        template='plotly_white', 
        hovermode='x unified',
        title_text='Plant-wise Coal Stock, Consumption & Power Generation (2023–2025)',
        title_font_size=18,
        title_x=0.5,
        font=dict(size=11),
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top'
        )
    )
    fig_ts.update_xaxes(title_text="Date", row=3, col=1)
    st.plotly_chart(fig_ts, use_container_width=True)
    st.caption('📌  Each subplot shows Stock (blue, left axis), Consumption (orange, right axis), and Power Generation (green, right axis) for a single plant over the full date range.')

    # ════════════════════════════════════════════════
    # 8.6  MONTHLY AGGREGATED
    # ════════════════════════════════════════════════
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>MONTHLY AGGREGATED TRENDS</div>",
                unsafe_allow_html=True)
    
    df_m = df_raw.copy()
    df_m['Date'] = pd.to_datetime(df_m['Date'])
    df_m.set_index('Date', inplace=True)
    
    # Aggregate available columns
    agg_dict = {col: 'mean' for col in STOCK_COLS + CONS_COLS + POWER_COLS
                if col in df_m.columns}
    
    if agg_dict:
        # Use pandas month-end frequency 'M' ("ME" is not a valid alias in some pandas versions)
        df_monthly = df_m.resample('M').agg(agg_dict).reset_index()
        
        # Create monthly charts for available plants
        for s, c, p, plant in zip(STOCK_COLS, CONS_COLS, POWER_COLS, PLANTS):
            # Check if this plant has data
            plant_has_data = any(col in df_monthly.columns for col in [s, c, p])
            
            if plant_has_data:
                from plotly.subplots import make_subplots as _ms
                fig_m = _ms(specs=[[{'secondary_y': True}]])
                
                # Add Stock on primary y-axis (left)
                if s in df_monthly.columns:
                    fig_m.add_trace(
                        go.Scatter(x=df_monthly['Date'], y=df_monthly[s],
                            mode='lines+markers', name='Stock', 
                            line=dict(width=3, color='#1f77b4'),
                            marker=dict(size=6)),
                        secondary_y=False
                    )
                
                # Add Consumption on secondary y-axis (right)
                if c in df_monthly.columns:
                    fig_m.add_trace(
                        go.Scatter(x=df_monthly['Date'], y=df_monthly[c],
                            mode='lines+markers', name='Consumption', 
                            line=dict(width=2, color='#ff7f0e'),
                            marker=dict(size=5)),
                        secondary_y=True
                    )
                
                # Add Power on secondary y-axis (right)
                if p in df_monthly.columns:
                    fig_m.add_trace(
                        go.Scatter(x=df_monthly['Date'], y=df_monthly[p],
                            mode='lines+markers', name='Power (GW)', 
                            line=dict(width=2, color='#2ca02c'),
                            marker=dict(size=5)),
                        secondary_y=True
                    )
                
                fig_m.update_layout(
                    title=f'{plant} — Monthly Averages',
                    template='plotly_white', 
                    height=500, 
                    hovermode='x unified',
                    font=dict(size=11),
                    title_font_size=14
                )
                fig_m.update_xaxes(title_text='Month', tickformat='%b-%Y')
                fig_m.update_yaxes(title_text='Stock Level', secondary_y=False)
                fig_m.update_yaxes(title_text='Consumption / Power', secondary_y=True)
                
                st.plotly_chart(fig_m, use_container_width=True)
                st.caption(f'📌  {plant}: Monthly mean — seasonality and operational cycles are visible.')
    else:
        st.warning("⚠️  No aggregatable columns found for monthly analysis.")

    # ════════════════════════════════════════════════
    # 8.7  PEAKS & TROUGHS
    # ════════════════════════════════════════════════
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>HISTORICAL PEAKS & TROUGHS ANALYSIS</div>",
                unsafe_allow_html=True)

    df_pk = df_raw.copy()
    df_pk['Date'] = pd.to_datetime(df_pk['Date'])
    df_pk.set_index('Date', inplace=True)
    df_pk = df_pk.asfreq('D').interpolate()

    stock_cols_present = [s for s in STOCK_COLS if s in df_pk.columns]
    
    if stock_cols_present:
        fig_pk = make_subplots(
            rows=len(stock_cols_present), cols=1,
            shared_xaxes=True, 
            vertical_spacing=0.10,
            subplot_titles=[f'Peaks & Troughs: {s}' for s in stock_cols_present]
        )

        print_output = []
        for row_i, (s_col, plant) in enumerate(zip(stock_cols_present, PLANTS), 1):
            arr   = df_pk[s_col].values
            idx   = df_pk.index
            s_std = df_pk[s_col].std()
            
            # Detect peaks and troughs with better parameters
            peaks,   _ = find_peaks(arr,  distance=14, prominence=s_std*0.2)
            troughs, _ = find_peaks(-arr, distance=14, prominence=s_std*0.2)

            lines = [
                f"{'='*60}",
                f"{s_col.upper()}{' — ' + plant if plant else ''}",
                f"Historical Peaks   : {len(peaks)}",
                f"Historical Troughs : {len(troughs)}",
                ""
            ]
            
            if len(peaks) > 0:
                lines.append("Top 5 Peaks:")
                top_peaks = sorted(peaks, key=lambda x: arr[x], reverse=True)[:5]
                for i in top_peaks:
                    lines.append(f"  📈  {idx[i].date()}  →  {arr[i]:>10,.0f}")
            
            lines.append("")
            
            if len(troughs) > 0:
                lines.append("Top 5 Troughs:")
                top_troughs = sorted(troughs, key=lambda x: arr[x])[:5]
                for i in top_troughs:
                    lines.append(f"  📉  {idx[i].date()}  →  {arr[i]:>10,.0f}")
            
            if len(peaks) > 0:
                peak_months   = pd.Series(idx[peaks].month).value_counts()
                lines.append(f"Peak-heavy months   : {[calendar.month_abbr[m] for m in peak_months.head(3).index.tolist()]}")
            
            if len(troughs) > 0:
                trough_months = pd.Series(idx[troughs].month).value_counts()
                lines.append(f"Trough-heavy months : {[calendar.month_abbr[m] for m in trough_months.head(3).index.tolist()]}")
            
            print_output.extend(lines)

            # Add traces to plot
            fig_pk.add_trace(
                go.Scatter(x=idx, y=arr, 
                    mode='lines',
                    line=dict(color='#1f77b4', width=1.5), 
                    name=f'{plant} Stock'),
                row=row_i, col=1
            )
            
            if len(peaks) > 0:
                fig_pk.add_trace(
                    go.Scatter(x=idx[peaks], y=arr[peaks],
                        mode='markers', 
                        marker=dict(color='#2ca02c', size=8, symbol='triangle-up'),
                        name=f'Peaks ({len(peaks)})'),
                    row=row_i, col=1
                )
            
            if len(troughs) > 0:
                fig_pk.add_trace(
                    go.Scatter(x=idx[troughs], y=arr[troughs],
                        mode='markers', 
                        marker=dict(color='#d62728', size=8, symbol='triangle-down'),
                        name=f'Troughs ({len(troughs)})'),
                    row=row_i, col=1
                )

        fig_pk.update_layout(
            height=250 * len(stock_cols_present), 
            template='plotly_white',
            hovermode='x unified',
            title_text='Peaks & Troughs Analysis — Coal Power Plants',
            title_font_size=16,
            font=dict(size=11),
            showlegend=True
        )
        fig_pk.update_xaxes(title_text="Date", row=len(stock_cols_present), col=1)
        
        st.plotly_chart(fig_pk, use_container_width=True)
        
        if print_output:
            st.code('\n'.join(print_output), language='text')
        
        st.caption('📌  Green triangles = historical demand peaks | Red triangles = troughs. Peak-heavy months indicate seasonal coal demand cycles.')
    else:
        st.warning("⚠️  No stock columns found for peaks & troughs analysis.")

    # ════════════════════════════════════════════════
    # GO TO FORECAST BUTTON
    # ════════════════════════════════════════════════
    st.markdown('<br><br>', unsafe_allow_html=True)
    if st.button('📈  Go to Forecast  →', key='goto_forecast', type='primary'):
        st.session_state['eda_done'] = True
        st.session_state['page']     = 'forecast'
        st.rerun()

    # ════════════════════════════════════════════════
    # TEMPERATURE DATASET ANALYSIS
    # ════════════════════════════════════════════════
    st.markdown('<br>', unsafe_allow_html=True)
    
    # Button to trigger temperature analysis
    if st.button('🌡️  Analyze Temperature Impact on Power Generation', key='temp_analysis', type='secondary'):
        st.session_state['show_temp_analysis'] = True
    
    # Show temperature analysis if button was clicked
    if st.session_state.get('show_temp_analysis', False):
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("<div class='section-pill'>EXTERNAL TEMPERATURE DATASET ANALYSIS</div>",
                    unsafe_allow_html=True)
        
        # Description
        st.markdown("""
        <div style='background:#f0f4ff; padding:12px; border-radius:8px; margin-bottom:16px;'>
          <strong>Analysis of how temperature and seasonal variables affect power generation</strong><br>
          This section uses an external dataset to explore correlations between weather features and power output 
          across the three plants before building forecasting models.
        </div>
        """, unsafe_allow_html=True)
        
        # Load temperature dataset
        temp_file_path = r"C:\Users\J RISHI KRISHNA\Downloads\Streamlit_Forcast\ntpc_dashboard\assets\power_prediction__var_temp 1.xlsx"
        
        try:
            with st.spinner('⏳ Loading temperature dataset...'):
                xldf__pg_temp = pd.read_excel(temp_file_path, sheet_name=["Barh", "Dadri", "Kudgi"])
            
            st.success(f'✅ Temperature dataset loaded successfully!')
            
            # Show preview of each sheet
            st.markdown("#### 📊 Dataset Preview")
            tab1, tab2, tab3 = st.tabs(["Barh", "Dadri", "Kudgi"])
            
            for tab, plant in zip([tab1, tab2, tab3], ["Barh", "Dadri", "Kudgi"]):
                with tab:
                    df_plant = xldf__pg_temp[plant]
                    st.write(f"**{plant}** — {len(df_plant)} rows × {len(df_plant.columns)} columns")
                    st.dataframe(df_plant.head(10), use_container_width=True, hide_index=True)
            
            # Season encoding
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown("#### 🔢 Season Encoding")
            st.markdown("""
            Converting categorical seasons to numeric values for correlation analysis:
            - Pre-Monsoon → 1
            - Monsoon → 2
            - Post-Monsoon → 3
            - Winter → 4
            """)
            
            season_map = {"Pre-Monsoon": 1, "Monsoon": 2, "Post-Monsoon": 3, "Winter": 4}
            for plant in ["Barh", "Dadri", "Kudgi"]:
                if "Season_Classification" in xldf__pg_temp[plant].columns:
                    xldf__pg_temp[plant]["Season_Classification"] = xldf__pg_temp[plant]["Season_Classification"].map(season_map)
            
            # Correlation Heatmaps
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown("#### 🔥 Correlation Heatmaps")
            st.markdown("Correlation between all variables including temperature, season, and power generation.")
            
            # Create correlation heatmaps and collect insights
            insights = {}
            for plant in ["Barh", "Dadri", "Kudgi"]:
                df_corr = xldf__pg_temp[plant].corr()
                
                # Find power column and its correlations
                power_cols = [col for col in df_corr.columns if 'power' in col.lower()]
                if power_cols:
                    power_col = power_cols[0]
                    power_corr = df_corr[power_col].drop(power_col).abs().sort_values(ascending=False)
                    top_corr = power_corr.head(3)
                    insights[plant] = {
                        'power_col': power_col,
                        'top_corr': top_corr,
                        'has_strong': any(power_corr > 0.7),
                        'has_moderate': any((power_corr > 0.3) & (power_corr <= 0.7))
                    }
                
                # Convert to Plotly heatmap
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=df_corr.values,
                    x=df_corr.columns,
                    y=df_corr.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=df_corr.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    colorbar=dict(title="Correlation")
                ))
                fig_heatmap.update_layout(
                    title=f'{plant} — Correlation Matrix',
                    template='plotly_white',
                    height=500,
                    width=700,
                    title_x=0.5,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Insights Summary Box
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown("#### 📋 Analysis Summary & Insights")
            
            # Build markdown summary
            summary_md = "**Key Findings from Temperature Dataset Analysis:**\n\n"
            
            for plant in ["Barh", "Dadri", "Kudgi"]:
                if plant not in insights:
                    summary_md += f"\n• **{plant}**: No power column found for correlation analysis.\n"
                    continue
                    
                info = insights[plant]
                power_col = info['power_col']
                top_corr = info['top_corr']
                
                summary_md += f"\n• **{plant}** (Power Column: `{power_col}`):\n"
                
                if len(top_corr) == 0:
                    summary_md += "  - No significant correlations found.\n"
                else:
                    for var, corr_val in top_corr.items():
                        corr_type = "strong" if corr_val > 0.7 else "moderate" if corr_val > 0.3 else "weak"
                        summary_md += f"  - **{var}**: {corr_val:.3f} ({corr_type} positive)\n"
            
            summary_md += "\n**Interpretation Guidelines:**\n"
            summary_md += "• **|r| > 0.7**: Strong correlation - likely influential on power generation\n"
            summary_md += "• **0.3 < |r| ≤ 0.7**: Moderate correlation - may have some influence\n"
            summary_md += "• **|r| ≤ 0.3**: Weak correlation - minimal linear influence\n"
            summary_md += "\n**Note:** Correlation does not imply causation. These relationships should be validated with domain expertise and further analysis."
            
            # Display in a styled container
            st.markdown(summary_md)
            
            st.success('✅ Temperature analysis complete!')
            
        except FileNotFoundError:
            st.error(f"❌ Temperature dataset not found at: `{temp_file_path}`")
            st.info("Please ensure the file `power_prediction__var_temp 1.xlsx` exists in the assets folder.")
        except Exception as e:
            st.error(f"❌ Error loading temperature dataset: {str(e)}")
            st.exception(e)

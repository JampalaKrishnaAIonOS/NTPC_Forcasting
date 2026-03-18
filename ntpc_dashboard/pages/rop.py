import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from groq import Groq
import json

# --- Path to the merged data file (already exists, no upload needed) ---
DATA_PATH = r'Reorder\merged_coal_data_final.xlsx'

# --- ROP Configuration (matches rop_algorithm_v3.py exactly) ---
LEAD_TIME_DAYS = 7
BUFFER_DAYS    = 2
EFFECTIVE_LT   = LEAD_TIME_DAYS + BUFFER_DAYS   # = 9
SERVICE_LEVEL  = 0.95
Z_SCORE        = stats.norm.ppf(SERVICE_LEVEL)   # ~ 1.645
PLANTS         = ['Barh', 'Dadri', 'Kudgi']

@st.cache_data
def load_data():
    """Load merged coal data from fixed path. Returns dict of {plant: DataFrame}."""
    try:
        sheets = pd.read_excel(DATA_PATH, sheet_name=None)
        plant_dfs = {}
        for plant in PLANTS:
            if plant in sheets:
                df = sheets[plant].copy()
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
                df = df.sort_values('Date').reset_index(drop=True)
                df = df.dropna(subset=['Consumption (MT)'])
                plant_dfs[plant] = df
        return plant_dfs
    except FileNotFoundError:
        return None

def compute_rop(df, lt=LEAD_TIME_DAYS, elt=EFFECTIVE_LT):
    """
    Compute all inventory control levels for one plant's DataFrame.
    Returns a dict of scalar metrics and the DataFrame with new columns added.
    
    Formulas (verified against standard inventory management texts):
      Safety Stock   = Z x sigma x sqrt(Effective LT)
      Reorder Level  = Max Consumption x Effective LT
      ROP            = (ADC x Effective LT) + Safety Stock   [between Min and Max]
      Minimum Level  = Reorder Level - (ADC x Avg LT)
      Maximum Level  = Reorder Level + Reorder Qty - (Min Consumption x Avg LT)
    """
    adc             = df['Consumption (MT)'].mean()
    sigma           = df['Consumption (MT)'].std()
    max_consumption = df['Consumption (MT)'].max()
    min_consumption = df['Consumption (MT)'].min()

    safety_stock  = Z_SCORE * sigma * np.sqrt(elt)
    reorder_level = max_consumption * elt
    rop           = (adc * elt) + safety_stock
    min_level     = reorder_level - (adc * lt)
    reorder_qty   = adc * elt
    max_level     = reorder_level + reorder_qty - (min_consumption * lt)

    df = df.copy()
    df['Days_of_Stock'] = (df['Stock (MT)'] / adc).round(1)
    df['ROP_Level']     = round(rop, 2)
    df['Min_Level']     = round(min_level, 2)
    df['Max_Level']     = round(max_level, 2)
    df['Reorder_Level'] = round(reorder_level, 2)
    df['Alert']         = df['Stock (MT)'] <= rop

    def _status(row):
        s = row['Stock (MT)']
        if s <= min_level:      return 'CRITICAL'
        elif s <= rop:          return 'REORDER NOW'
        elif s <= reorder_level:return 'WATCH'
        elif s > max_level:     return 'OVERSTOCK'
        else:                   return 'NORMAL'

    df['Stock_Status'] = df.apply(_status, axis=1)

    return {
        'adc':             round(adc, 2),
        'sigma':           round(sigma, 2),
        'max_consumption': round(max_consumption, 2),
        'min_consumption': round(min_consumption, 2),
        'safety_stock':    round(safety_stock, 2),
        'reorder_level':   round(reorder_level, 2),
        'rop':             round(rop, 2),
        'min_level':       round(min_level, 2),
        'max_level':       round(max_level, 2),
        'reorder_qty':     round(reorder_qty, 2),
        'df':              df,
    }

def get_ai_explanation(plant_name, metrics, alert_count, total_rows):
    """
    Call the Groq AI API (model: Kimi-K2-Instruct-0905) to generate a plain-English
    explanation of the ROP analysis for the given plant.
    Uses the GROQ_API_KEY from the environment (loaded from .env).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not found in environment. Please check your .env file."

    prompt = f"""
You are a coal inventory management expert for NTPC power plants in India.

Explain the following Re-Order Point (ROP) analysis results for the {plant_name} plant
in clear, non-technical language for a procurement manager. Be concise (3-4 paragraphs).
Focus on: what the numbers mean operationally, what actions to take, and any risks.

Plant: {plant_name}
Average Daily Consumption: {metrics['adc']:,.0f} MT/day
Std Deviation: {metrics['sigma']:,.0f} MT
Lead Time: {LEAD_TIME_DAYS} days + {BUFFER_DAYS} days buffer = {EFFECTIVE_LT} days effective
Safety Stock: {metrics['safety_stock']:,.0f} MT
Minimum Level (never go below): {metrics['min_level']:,.0f} MT
Re-Order Point (place order when stock hits this): {metrics['rop']:,.0f} MT
Reorder Level (upper watch zone trigger): {metrics['reorder_level']:,.0f} MT
Maximum Level (overstocking ceiling): {metrics['max_level']:,.0f} MT
Alert Days (stock was at or below ROP): {alert_count} out of {total_rows} days ({100*alert_count/total_rows:.1f}%)

Use simple language. Include:
1. What the ROP number means in practice for this plant
2. What the safety stock buffer protects against
3. Whether the alert frequency ({100*alert_count/total_rows:.1f}%) is concerning
4. One specific procurement recommendation
"""
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI explanation could not be generated: {str(e)}"

def make_stock_levels_chart(df, metrics, plant_name):
    """
    Line chart: Stock (MT) over time with horizontal reference lines for
    Min Level, ROP, Reorder Level, Max Level. Alert days shown as red dots.
    """
    fig = go.Figure()

    # Stock line
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Stock (MT)'],
        mode='lines', name='Stock (MT)',
        line=dict(color='#001B94', width=2)
    ))

    # Alert dots (red) - days where stock <= ROP
    alerts = df[df['Alert']]
    if not alerts.empty:
        fig.add_trace(go.Scatter(
            x=alerts['Date'], y=alerts['Stock (MT)'],
            mode='markers', name='Alert (Stock <= ROP)',
            marker=dict(color='#EF4444', size=6, symbol='circle')
        ))

    # Horizontal level lines
    levels = [
        (metrics['min_level'],     'Min Level',     '#C00000', 'dash'),
        (metrics['rop'],           'ROP',           '#FF6B00', 'dashdot'),
        (metrics['reorder_level'], 'Reorder Level', '#F59E0B', 'dot'),
        (metrics['max_level'],     'Max Level',     '#22C55E', 'dash'),
    ]
    for val, label, color, dash in levels:
        fig.add_hline(
            y=val, line_dash=dash, line_color=color, line_width=1.5,
            annotation_text=f'{label}: {val:,.0f} MT',
            annotation_position='right',
            annotation_font=dict(size=10, color=color)
        )

    fig.update_layout(
        title=f'{plant_name} - Stock Level vs Inventory Control Bands',
        xaxis_title='Date', yaxis_title='Stock (MT)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=420,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#F0F0F0')
    fig.update_yaxes(showgrid=True, gridcolor='#F0F0F0')
    return fig

def make_consumption_chart(df, plant_name):
    """
    Daily consumption bar chart with a rolling 30-day average overlay.
    Highlights max and min consumption days.
    """
    df = df.copy()
    df['Rolling_Avg'] = df['Consumption (MT)'].rolling(30, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Consumption (MT)'],
        name='Daily Consumption',
        marker_color='#93C5FD', opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Rolling_Avg'],
        mode='lines', name='30-Day Rolling Avg',
        line=dict(color='#001B94', width=2)
    ))
    fig.update_layout(
        title=f'{plant_name} - Daily Consumption & 30-Day Average',
        xaxis_title='Date', yaxis_title='Consumption (MT)',
        height=360,
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    return fig

def make_days_of_stock_chart(df, plant_name):
    """
    Days of Stock Remaining over time. Color zones:
    > 14 days = green, 7-14 = orange, <= 7 = red.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Days_of_Stock'],
        mode='lines', name='Days of Stock',
        line=dict(color='#001B94', width=1.5),
        fill='tozeroy', fillcolor='rgba(0,27,148,0.08)'
    ))
    # Danger zone shading
    fig.add_hrect(y0=0, y1=7,   fillcolor='rgba(239,68,68,0.10)',  line_width=0, annotation_text='Critical (<=7d)')
    fig.add_hrect(y0=7, y1=14,  fillcolor='rgba(245,158,11,0.10)', line_width=0, annotation_text='Watch (7-14d)')
    fig.add_hrect(y0=14, y1=df['Days_of_Stock'].max()*1.05,
                  fillcolor='rgba(34,197,94,0.06)', line_width=0, annotation_text='Safe (>14d)')
    fig.update_layout(
        title=f'{plant_name} - Days of Stock Remaining',
        xaxis_title='Date', yaxis_title='Days',
        height=360,
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Arial', size=12)
    )
    return fig

def make_all_plants_comparison_chart(all_metrics):
    """
    Grouped bar chart comparing Min Level, ROP, Reorder Level, Max Level
    across all three plants side by side.
    """
    plants = list(all_metrics.keys())
    categories = ['Min Level', 'ROP', 'Reorder Level', 'Max Level']
    colors     = ['#C00000',  '#FF6B00', '#F59E0B',      '#22C55E']
    keys       = ['min_level','rop',     'reorder_level','max_level']

    fig = go.Figure()
    for cat, color, key in zip(categories, colors, keys):
        fig.add_trace(go.Bar(
            name=cat,
            x=plants,
            y=[all_metrics[p][key] for p in plants],
            marker_color=color,
            text=[f"{all_metrics[p][key]:,.0f}" for p in plants],
            textposition='outside'
        ))

    fig.update_layout(
        barmode='group',
        title='All Plants - Inventory Level Comparison (MT)',
        xaxis_title='Plant', yaxis_title='Stock Level (MT)',
        height=420,
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    return fig

def make_status_distribution_chart(df, plant_name):
    """
    Donut/pie chart showing % of days in each Stock_Status category.
    """
    status_counts = df['Stock_Status'].value_counts()
    color_map = {
        'CRITICAL':    '#C00000',
        'REORDER NOW': '#EF4444',
        'WATCH':       '#F59E0B',
        'NORMAL':      '#22C55E',
        'OVERSTOCK':   '#6B7280',
    }
    colors = [color_map.get(s, '#999999') for s in status_counts.index]
    fig = go.Figure(go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.45,
        marker=dict(colors=colors),
        textinfo='label+percent'
    ))
    fig.update_layout(
        title=f'{plant_name} - Stock Status Distribution (% of Days)',
        height=360,
        font=dict(family='Arial', size=12)
    )
    return fig

def render():
    """Main render function called by app.py router."""

    # --- Page header ---
    st.markdown("""
        <div style='background:linear-gradient(90deg,#001B94,#2E75B6);
                    padding:20px 28px;border-radius:8px;margin-bottom:20px'>
            <h2 style='color:white;margin:0'>Re-Order Point (ROP) Analysis</h2>
            <p style='color:#ADD8E6;margin:4px 0 0'>
                Coal inventory control - Barh - Dadri - Kudgi
                &nbsp;|&nbsp; Live from merged_coal_data_final.xlsx
            </p>
        </div>
    """, unsafe_allow_html=True)

    # --- Load data ---
    with st.spinner('Loading merged coal data...'):
        plant_dfs = load_data()

    if plant_dfs is None:
        st.error(
            "**File not found:** `merged_coal_data_final.xlsx`\\n\\n"
            "Make sure the file is in the same folder as `app.py`.\\n"
            "It should have sheets named: Barh, Dadri, Kudgi."
        )
        return

    if not plant_dfs:
        st.error("No valid plant sheets found. Expected: Barh, Dadri, Kudgi.")
        return

    # --- Compute ROP for all plants ---
    with st.spinner('Computing ROP levels for all plants...'):
        all_metrics = {}
        for plant in plant_dfs:
            all_metrics[plant] = compute_rop(plant_dfs[plant])

    # --- Config sidebar (parameters) ---
    with st.expander('Settings ROP Configuration Parameters', expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Lead Time (days)', LEAD_TIME_DAYS)
            st.caption('Base days from order to rake arrival')
        with col2:
            st.metric('Buffer Days', BUFFER_DAYS)
            st.caption('Extra days to absorb delays')
        with col3:
            st.metric('Effective Lead Time', EFFECTIVE_LT)
            st.caption(f'= {LEAD_TIME_DAYS} + {BUFFER_DAYS} days total')
        st.info(f'Service Level: {SERVICE_LEVEL*100:.0f}%  |  Z-Score: {Z_SCORE:.3f}  '
                f'(To change these, edit `pages/rop.py` configuration block at the top)')

    st.markdown('---')

    # -------------------------------
    # TAB LAYOUT: Overview | Barh | Dadri | Kudgi | Data Explorer
    # -------------------------------
    tab_overview, tab_barh, tab_dadri, tab_kudgi, tab_data = st.tabs([
        'All Plants Overview',
        'Barh',
        'Dadri',
        'Kudgi',
        'Data Explorer',
    ])

    # ---------------------------------
    # TAB 1: ALL PLANTS OVERVIEW
    # ---------------------------------
    with tab_overview:
        st.subheader('ROP Summary - All Three Plants')

        # KPI cards row (one column per plant)
        cols = st.columns(len(all_metrics))
        for col, (plant, m) in zip(cols, all_metrics.items()):
            df_p = m['df']
            alert_pct = 100 * df_p['Alert'].sum() / len(df_p)
            status_color = '#C00000' if alert_pct > 20 else '#F59E0B' if alert_pct > 10 else '#22C55E'
            col.markdown(f"""
                <div style='border:1px solid #E5E7EB;border-radius:8px;padding:16px;
                            background:#F8FAFC;margin-bottom:8px'>
                    <h4 style='color:#001B94;margin:0 0 8px'>{plant}</h4>
                    <p style='margin:2px 0'><b>ADC:</b> {m['adc']:,.0f} MT/day</p>
                    <p style='margin:2px 0'><b>Safety Stock:</b> {m['safety_stock']:,.0f} MT</p>
                    <p style='margin:2px 0'><b>Min Level:</b>
                        <span style='color:#C00000;font-weight:bold'>{m['min_level']:,.0f} MT</span></p>
                    <p style='margin:2px 0'><b>ROP:</b>
                        <span style='color:#FF6B00;font-weight:bold'>{m['rop']:,.0f} MT</span></p>
                    <p style='margin:2px 0'><b>Reorder Level:</b> {m['reorder_level']:,.0f} MT</p>
                    <p style='margin:2px 0'><b>Max Level:</b>
                        <span style='color:#22C55E;font-weight:bold'>{m['max_level']:,.0f} MT</span></p>
                    <p style='margin:4px 0 0'><b>Alert Days:</b>
                        <span style='color:{status_color};font-weight:bold'>
                        {df_p['Alert'].sum()} / {len(df_p)} ({alert_pct:.1f}%)</span></p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # Comparison bar chart
        st.plotly_chart(
            make_all_plants_comparison_chart(all_metrics),
            use_container_width=True
        )

        # Summary table
        st.subheader('Full ROP Metrics Table (Top 10 rows per plant)')
        summary_rows = []
        for plant, m in all_metrics.items():
            summary_rows.append({
                'Plant':             plant,
                'ADC (MT/day)':      f"{m['adc']:,.1f}",
                'Std Dev (MT)':      f"{m['sigma']:,.1f}",
                'Safety Stock (MT)': f"{m['safety_stock']:,.0f}",
                'Min Level (MT)':    f"{m['min_level']:,.0f}",
                'ROP (MT)':          f"{m['rop']:,.0f}",
                'Reorder Level (MT)':f"{m['reorder_level']:,.0f}",
                'Max Level (MT)':    f"{m['max_level']:,.0f}",
                'Alert Days':        str(m['df']['Alert'].sum()),
                'Alert %':           f"{100*m['df']['Alert'].sum()/len(m['df']):.1f}%",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # Color legend
        st.markdown("""
        **Level Color Guide:**
        RED **Min Level** - Stock must NEVER fall below this (emergency zone)  
        ORANGE **ROP** - Place order NOW when stock hits this  
        YELLOW **Reorder Level** - Watch zone, prepare purchase order  
        GREEN **Max Level** - Do not over-order beyond this  
        """)

    # ---------------------------------
    # HELPER: single-plant tab content (used for Barh, Dadri, Kudgi tabs)
    # ---------------------------------
    def render_plant_tab(plant_name):
        if plant_name not in all_metrics:
            st.warning(f'No data found for {plant_name}.')
            return

        m  = all_metrics[plant_name]
        df = m['df']
        alert_count = int(df['Alert'].sum())
        total_rows  = len(df)

        # --- KPI strip ---
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric('ADC (MT/day)',     f"{m['adc']:,.0f}")
        k2.metric('Safety Stock',     f"{m['safety_stock']:,.0f} MT")
        k3.metric('ROP',              f"{m['rop']:,.0f} MT")
        k4.metric('Min Level',        f"{m['min_level']:,.0f} MT")
        k5.metric('Alert Days',       f"{alert_count} ({100*alert_count/total_rows:.0f}%)",
                  delta=f"{'High' if alert_count/total_rows > 0.15 else 'OK'}",
                  delta_color='inverse')

        st.markdown('---')

        # --- Chart 1: Stock vs all level bands ---
        st.subheader('Stock Level vs Inventory Control Bands')
        st.plotly_chart(
            make_stock_levels_chart(df, m, plant_name),
            use_container_width=True
        )
        st.caption(
            f"**Red dots** = alert days (stock <= ROP of {m['rop']:,.0f} MT). "
            f"**{alert_count}** alert days found out of {total_rows} total days."
        )

        # --- Chart 2 & 3 side by side ---
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader('Daily Consumption')
            st.plotly_chart(
                make_consumption_chart(df, plant_name),
                use_container_width=True
            )
        with col_right:
            st.subheader('Days of Stock Remaining')
            st.plotly_chart(
                make_days_of_stock_chart(df, plant_name),
                use_container_width=True
            )

        # --- Chart 4: Status distribution donut ---
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader('Status Distribution')
            st.plotly_chart(
                make_status_distribution_chart(df, plant_name),
                use_container_width=True
            )
        with col_b:
            st.subheader('Stock Status Breakdown')
            status_df = (df['Stock_Status']
                         .value_counts()
                         .reset_index()
                         .rename(columns={'Stock_Status':'Status','count':'Days'}))
            status_df['% of Time'] = (status_df['Days'] / total_rows * 100).round(1)
            status_colors = {
                'CRITICAL':    'RED', 'REORDER NOW': 'ORANGE',
                'WATCH':       'YELLOW', 'NORMAL':      'GREEN', 'OVERSTOCK': 'BLACK'
            }
            status_df['Status'] = status_df['Status'].map(
                lambda s: f"{status_colors.get(s,'')} {s}"
            )
            st.dataframe(status_df, use_container_width=True, hide_index=True)

            # Meaning table
            st.markdown("""
| Status | Condition | Action |
|--------|-----------|--------|
| RED CRITICAL | Stock <= Min Level | Emergency procurement |
| ORANGE REORDER NOW | Stock <= ROP | Place order immediately |
| YELLOW WATCH | ROP < Stock <= Reorder Level | Prepare purchase order |
| GREEN NORMAL | Within safe range | No action needed |
| BLACK OVERSTOCK | Stock > Max Level | Pause incoming orders |
""")

        st.markdown('---')

        # --- Top 10 data preview ---
        st.subheader(f'{plant_name} - Top 10 Latest Records')
        display_cols = [
            'Date', 'Stock (MT)', 'Consumption (MT)', 'RR Quantity (MT)',
            'Days_of_Stock', 'ROP_Level', 'Min_Level', 'Max_Level',
            'Alert', 'Stock_Status'
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        top10 = df.sort_values('Date', ascending=False).head(10)[available_cols]

        # Color-code the Stock_Status column
        def highlight_status(val):
            colors = {
                'CRITICAL':    'background-color:#FFE0E0;color:#C00000;font-weight:bold',
                'REORDER NOW': 'background-color:#FEE2E2;color:#EF4444;font-weight:bold',
                'WATCH':       'background-color:#FEF3C7;color:#D97706',
                'NORMAL':      'background-color:#D1FAE5;color:#065F46',
                'OVERSTOCK':   'background-color:#F3F4F6;color:#374151',
            }
            return colors.get(val, '')

        styled = top10.style.applymap(
            highlight_status, subset=['Stock_Status']
        ).format({
            'Stock (MT)':       '{:,.0f}',
            'Consumption (MT)': '{:,.0f}',
            'RR Quantity (MT)': '{:,.1f}',
            'Days_of_Stock':    '{:.1f}',
            'ROP_Level':        '{:,.0f}',
            'Min_Level':        '{:,.0f}',
            'Max_Level':        '{:,.0f}',
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown('---')

        # --- Final Analysis + AI Explanation ---
        st.subheader('Chart Final Analysis')

        # Hard-coded statistical summary (no API needed)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
**Inventory Parameters - {plant_name}**
- Average Daily Consumption: **{m['adc']:,.0f} MT/day**
- Peak Consumption Day: **{m['max_consumption']:,.0f} MT**
- Lowest Consumption Day: **{m['min_consumption']:,.0f} MT**
- Std Deviation: **{m['sigma']:,.0f} MT** (variability measure)
- Lead Time: **{LEAD_TIME_DAYS}** base + **{BUFFER_DAYS}** buffer = **{EFFECTIVE_LT} days** effective
- Service Level: **{SERVICE_LEVEL*100:.0f}%** (Z = {Z_SCORE:.3f})
""")
        with col2:
            st.markdown(f"""
**Control Levels - {plant_name}**
- RED **Minimum Level:** {m['min_level']:,.0f} MT - absolute safety floor
- ORANGE **Re-Order Point:** {m['rop']:,.0f} MT - trigger new order here
- YELLOW **Reorder Level:** {m['reorder_level']:,.0f} MT - enter watch zone
- GREEN **Maximum Level:** {m['max_level']:,.0f} MT - overstocking ceiling
- **Safety Stock Embedded:** {m['safety_stock']:,.0f} MT
- **Alert Days:** {alert_count} / {total_rows} ({100*alert_count/total_rows:.1f}%)
""")

        # AI Explanation
        st.subheader('AI Explanation (Plain English)')
        st.caption('Uses Groq Cloud (Kimi-2) to explain the ROP analysis in plain language for procurement managers.')

        ai_key = f'rop_ai_{plant_name}'
        if ai_key not in st.session_state:
            st.session_state[ai_key] = None

        if st.button(f'Generate AI Explanation for {plant_name}', key=f'ai_btn_{plant_name}'):
            with st.spinner('Asking Kimi AI for analysis...'):
                explanation = get_ai_explanation(plant_name, m, alert_count, total_rows)
                st.session_state[ai_key] = explanation

        if st.session_state.get(ai_key):
            # Display AI output as markdown with headings and paragraphs
            ai_text = st.session_state[ai_key].strip()
            # Ensure blank lines between paragraphs
            ai_text_md = ai_text.replace('\r\n', '\n').replace('\r', '\n').replace('\n\n', '\n').replace('\n', '\n\n')
            st.markdown(f"""
### AI Analysis - {plant_name}

{ai_text_md}
""")

    # ---------------------------------
    # TABS 2-4: Individual plant tabs
    # ---------------------------------
    with tab_barh:
        render_plant_tab('Barh')

    with tab_dadri:
        render_plant_tab('Dadri')

    with tab_kudgi:
        render_plant_tab('Kudgi')

    # ---------------------------------
    # TAB 5: DATA EXPLORER
    # ---------------------------------
    with tab_data:
        st.subheader('Full Data Explorer - All Plants with ROP Columns')

        plant_choice = st.selectbox(
            'Select Plant', PLANTS, key='data_explorer_plant'
        )
        if plant_choice in all_metrics:
            df_full = all_metrics[plant_choice]['df']

            # Filters
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                status_filter = st.multiselect(
                    'Filter by Stock Status',
                    options=df_full['Stock_Status'].unique().tolist(),
                    default=df_full['Stock_Status'].unique().tolist(),
                    key='data_status_filter'
                )
            with col_f2:
                date_range = st.date_input(
                    'Date Range',
                    value=(df_full['Date'].min().date(), df_full['Date'].max().date()),
                    key='data_date_filter'
                )

            # Apply filters
            mask = df_full['Stock_Status'].isin(status_filter)
            if len(date_range) == 2:
                mask &= (df_full['Date'].dt.date >= date_range[0]) & (df_full['Date'].dt.date <= date_range[1])
            df_filtered = df_full[mask]

            st.info(f'Showing **{len(df_filtered):,}** rows of **{len(df_full):,}** total')
            st.dataframe(df_filtered, use_container_width=True, hide_index=True)

            # Download button
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f'Download {plant_choice} ROP Data (CSV)',
                data=csv,
                file_name=f'rop_analysis_{plant_choice.lower()}.csv',
                mime='text/csv',
                key=f'dl_{plant_choice}'
            )

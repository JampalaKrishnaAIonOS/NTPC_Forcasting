import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


st.set_page_config(
    page_title='NTPC Forecasting Dashboard',
    page_icon='âš¡',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# â”€â”€ Load IndiGo theme â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with open('styles/theme.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# â”€â”€ Session state defaults â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
defaults = {
    'page': 'home',
    'df_raw': None,
    'eda_done': False,
    'forecast_done': False,
    'forecast_plant': None,
    'forecast_model': None,
    'forecast_figs': {},
    'forecast_dfs': {},
    'forecast_rmse': 0.0,
    'forecast_mae': 0.0,
    'forecast_coverage': 0.0,
    'chat_history': [],
    'events_df': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# â”€â”€ Logo paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
NTPC_LOGO   = 'assets/ntpc_logo.png'
AIONOS_LOGO = 'assets/aionos_logo.png'

# â”€â”€ Navbar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def render_navbar():
    pages = ['home', 'eda', 'forecast', 'validation', 'xai', 'chatbot']
    labels = ['ðŸ  Home', 'ðŸ“Š EDA', 'ðŸ“ˆ Forecast', 'ðŸ”¬ Validation', 'ðŸ§  XAI', 'ðŸ¤– Assistant']
    unlocked = {
        'home':       True,
        'eda':        st.session_state['df_raw'] is not None,
        'forecast':   st.session_state['eda_done'],
        'validation': st.session_state['forecast_done'],
        'xai':        st.session_state['forecast_done'],
        'chatbot':    st.session_state['forecast_done'],
    }
    cols = st.columns([1,1,1,1,1,1,1,1,1,1,1])
    # Left: logos + title
    with cols[0]:
        if os.path.exists(NTPC_LOGO):
            st.image(NTPC_LOGO, width=60)
    with cols[1]:
        if os.path.exists(AIONOS_LOGO):
            st.image(AIONOS_LOGO, width=60)
    with cols[2]:
        st.markdown('<span class=\'navbar-title\'>NTPC Forecast</span>', unsafe_allow_html=True)
    # Right: nav links
    for i, (p, label) in enumerate(zip(pages, labels)):
        with cols[4 + i]:
            cls = 'active' if st.session_state['page'] == p else ''
            cls += ' locked' if not unlocked[p] else ''
            if unlocked[p]:
                if st.button(label, key=f'nav_{p}', use_container_width=True):
                    st.session_state['page'] = p
                    st.rerun()
            else:
                st.markdown(f'<span class=\'nav-link locked\'>{label}</span>',
                            unsafe_allow_html=True)

# â”€â”€ Step progress bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def render_progress():
    steps = [
        ('home',       '1  Upload Data'),
        ('eda',        '2  EDA'),
        ('forecast',   '3  Forecast'),
        ('validation', '4  Validation'),
        ('xai',        '5  XAI'),
        ('chatbot',    '6  AI Assistant'),
    ]
    done_pages = {
        'home':       st.session_state['df_raw'] is not None,
        'eda':        st.session_state['eda_done'],
        'forecast':   st.session_state['forecast_done'],
        'validation': False,
        'xai':        False,
        'chatbot':    False,
    }
    unlocked = {
        'home':       True,
        'eda':        st.session_state['df_raw'] is not None,
        'forecast':   st.session_state['eda_done'],
        'validation': st.session_state['forecast_done'],
        'xai':        st.session_state['forecast_done'],
        'chatbot':    st.session_state['forecast_done'],
    }
    cols = st.columns(len(steps))
    for col, (p, label) in zip(cols, steps):
        if st.session_state['page'] == p: cls = 'active'
        elif done_pages[p]: cls = 'done'
        else: cls = ''
        # Make steps clickable if unlocked
        if unlocked[p]:
            if col.button(label, key=f'step_{p}', use_container_width=True,
                         type='primary' if cls == 'active' else 'secondary',
                         help=f'Go to {label}'):
                st.session_state['page'] = p
                st.rerun()
        else:
            col.markdown(f'<div class=\'step {cls} locked\'>{label}</div>', unsafe_allow_html=True)

# â”€â”€ Router â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Navbar removed to reduce top clutter â€” main content rendered immediately
st.markdown('<div class=\'main-content\'>', unsafe_allow_html=True)
render_progress()

page = st.session_state['page']
if page == 'home':
    from pages.home import render
    render()
elif page == 'eda':
    if st.session_state['df_raw'] is None:
        st.warning('Please upload data on the Home page first.')
    else:
        from pages.eda import render
        render()
elif page == 'forecast':
    if not st.session_state['eda_done']:
        st.warning('Please complete EDA before forecasting.')
    else:
        from pages.forecast import render
        render()
elif page == 'validation':
    if not st.session_state['forecast_done']:
        st.warning('Please complete a Forecast before viewing Validation.')
    else:
        from pages.validation import render
        render()
elif page == 'xai':
    if not st.session_state['forecast_done']:
        st.warning('Please complete a Forecast before viewing XAI.')
    else:
        from pages.xai import render
        render()
elif page == 'chatbot':
    if not st.session_state['forecast_done']:
        st.warning('Please complete a Forecast before using the AI Assistant.')
    else:
        from pages.chatbot import render
        render()

st.markdown('</div>', unsafe_allow_html=True)

"""
Page 6 – AI Assistant (RAG Chatbot using Groq + Kimi-K2)
"""
import os
import streamlit as st
from dotenv import load_dotenv

from core.rag_engine import build_forecast_context, ask_llm

# Load environment variables
load_dotenv()

# Get Groq API key from environment
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in .env file.")


def render():
    st.markdown("""
    <div class='card'>
      <div class='section-pill'>STEP 6 â€” AI ASSISTANT</div>
      <h2 style='color:#001B94;margin:8px 0 4px'>ðŸ¤– AI Assistant</h2>
      <p style='color:#6B7280;font-size:14px'>
        Ask questions about your forecast in plain English.
        The assistant has full context of your forecast results.
      </p>
    </div>""", unsafe_allow_html=True)

    # â”€â”€ Check prerequisites â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not st.session_state.get('forecast_done'):
        st.warning("Run a forecast first so the assistant has data to discuss.")
        return

    # â”€â”€ Pull forecast context from session state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    plant_key  = st.session_state.get('forecast_plant', 'barh')
    model_name = st.session_state.get('forecast_model', 'ets_tsr')

    df_forecast = st.session_state['forecast_dfs'].get(plant_key)
    rmse        = st.session_state.get('forecast_rmse', 0.0)
    mae         = st.session_state.get('forecast_mae', 0.0)
    coverage    = st.session_state.get('forecast_coverage', 0.0)
    events_df   = st.session_state.get('events_df')

    if df_forecast is None:
        st.error("No forecast data found. Re-run the forecast.")
        return

    context = build_forecast_context(
        plant_key, model_name, rmse, mae, coverage,
        df_forecast, events_df
    )

    # â”€â”€ Chat history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # â”€â”€ Suggested starter questions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.markdown("#### ðŸ’¡ Suggested questions")
    suggestions = [
        f"Why will demand increase at {plant_key.upper()} next month?",
        "When is the next PEAK event and how should we prepare?",
        "How reliable is this forecast? What does the RMSE mean?",
        "What months have the highest demand historically?",
        "What action should we take given the next LOW event?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
            st.session_state['pending_question'] = q

    st.markdown("---")

    # â”€â”€ Display chat history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # â”€â”€ Chat input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    user_input = st.chat_input("Ask about your forecast...")

    # Handle suggested question if clicked
    if 'pending_question' in st.session_state:
        user_input = st.session_state.pop('pending_question')

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add to history (without the context injection â€” keep history clean)
        st.session_state['chat_history'].append(
            {"role": "user", "content": user_input}
        )

        # Build history WITHOUT current message (context is fresh each turn)
        history_without_current = st.session_state['chat_history'][:-1]

        # Call Groq API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_llm(
                        context, user_input, history_without_current,
                        api_key=GROQ_API_KEY
                    )
                    st.markdown(answer)
                    st.session_state['chat_history'].append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    err = f"Error calling Groq API: {str(e)}"
                    st.error(err)

    # â”€â”€ Clear chat button â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if st.session_state['chat_history']:
        if st.button("ðŸ—‘ Clear conversation", type="secondary"):
            st.session_state['chat_history'] = []
            st.rerun()

    # â”€â”€ Context preview (expandable) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with st.expander("ðŸ” View context sent to AI"):
        st.code(context, language="text")

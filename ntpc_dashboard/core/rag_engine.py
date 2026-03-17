"""
RAG Engine — builds context from forecast data and queries Groq API (Kimi-K2).
"""

import pandas as pd
from groq import Groq

SYSTEM_PROMPT = """
You are an expert energy analyst assistant for NTPC (National Thermal Power Corporation).
You help operations teams understand coal power plant forecasts.

You will be given structured forecast context including:
- Plant name and model used
- Key performance metrics (RMSE, MAE, Coverage)
- Forecast summary statistics
- Detected PEAK and LOW demand events
- Historical data patterns

Answer questions clearly and concisely. Focus on actionable insights.
When citing numbers, be precise. If the context doesn't contain the answer, say so.
Do not make up data that isn't in the context.
"""


def build_forecast_context(
    plant_key: str,
    model_name: str,
    rmse: float,
    mae: float,
    coverage: float,
    df_forecast: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
) -> str:
    """
    Converts session state data into a structured text block
    that gets injected into each query as context.
    """
    col = df_forecast.columns[0]
    vals = df_forecast[col]

    context = f"""
=== FORECAST CONTEXT ===
Plant:       {plant_key.upper()}
Model:       {model_name.upper()}
Horizon:     {len(df_forecast)} days
Date range:  {vals.index[0].strftime('%Y-%m-%d')} to {vals.index[-1].strftime('%Y-%m-%d')}

=== MODEL PERFORMANCE ===
RMSE:     {rmse:.4f} GW
MAE:      {mae:.4f} GW
Coverage: {coverage:.1f}% (within 96% CI)

=== FORECAST STATISTICS ===
Mean predicted power:    {vals.mean():.3f} GW
Max predicted power:     {vals.max():.3f} GW  (on {vals.idxmax().strftime('%Y-%m-%d')})
Min predicted power:     {vals.min():.3f} GW  (on {vals.idxmin().strftime('%Y-%m-%d')})
Std deviation:           {vals.std():.3f} GW

=== MONTHLY AVERAGES ===
{vals.resample('ME').mean().round(3).to_string()}
"""

    if events_df is not None and len(events_df) > 0:
        peak_events = events_df[events_df['EVENT'] == 'PEAK']
        low_events  = events_df[events_df['EVENT'] == 'LOW']
        context += f"""

=== DETECTED EVENTS ===
PEAK events (top 2% demand): {len(peak_events)}
  Next PEAK: {peak_events.index[0].strftime('%Y-%m-%d') if len(peak_events) > 0 else 'None'}
  Max lead time: {int(peak_events['DAYS_AHEAD'].max()) if len(peak_events) > 0 else 0} days

LOW events (bottom 2% demand): {len(low_events)}
  Next LOW:  {low_events.index[0].strftime('%Y-%m-%d') if len(low_events) > 0 else 'None'}
  Max lead time: {int(low_events['DAYS_AHEAD'].max()) if len(low_events) > 0 else 0} days
"""

    return context


def ask_llm(context: str, question: str, history: list[dict], api_key: str) -> str:
    """
    Sends a question + context to Groq (Kimi-K2) and returns the answer.
    `history` is a list of {role, content} dicts for multi-turn memory.
    """
    client = Groq(api_key=api_key)

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    for turn in history:
        messages.append(turn)

    # Inject context into the current question
    full_question = f"""
{context}

=== USER QUESTION ===
{question}
"""

    messages.append({"role": "user", "content": full_question})

    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        max_tokens=3000,
        messages=messages,
    )

    return response.choices[0].message.content

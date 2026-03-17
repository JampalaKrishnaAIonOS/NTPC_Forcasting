import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import periodogram


POWER_COLS = {
    "barh": "Barh(Power GW)",
    "dadri": "Dadri Thermal(Power GW)",
    "kudgi": "Kudgi(Power GW)"
}


def prepare_series(df_raw):
    """
    Prepare plant series with Date index
    """
    df = df_raw.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    df = df.asfreq("D").interpolate()

    plants = {}

    for key, col in POWER_COLS.items():
        sub = df[[col]].copy()
        sub.columns = ["powerGW"]
        plants[key] = sub

    return plants


def func__periodogram(series_df):
    """Return (period:int, fig:go.Figure)"""
    col = series_df.columns[-1]
    freq, pow_ = periodogram(series_df[col].dropna())
    freq_max  = freq[np.argmax(pow_)]
    period    = int(1 / freq_max)
    print(f'  n periods : {period}')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=pow_, mode='lines',
                             line=dict(color='gray'), name='freq vs pow'))
    fig.add_trace(go.Scatter(
        x=[freq_max, freq_max], y=[0, max(pow_)],
        mode='lines', line=dict(color='black', dash='dash'),
        hoverinfo='skip',
        name=f'max freq: {freq_max:.4f} | period: {period}'))
    return period, fig


def make_train_test(series_dict):
    """Returns dict with train/test splits for each plant."""
    periods = {}
    figs    = {}
    for key in ['barh','dadri','kudgi']:
        period, fig = func__periodogram(series_dict[key])
        periods[key] = period
        figs[key]    = fig

    splits = {}
    for key in ['barh','dadri','kudgi']:
        s = series_dict[key]
        p = periods[key]
        splits[key] = {
            'train':  s.iloc[:-p],
            'test':   s.iloc[-p:],
            'period': p,
            'fig_periodogram': figs[key]
        }
    return splits


def build_features(df):
    """
    Recreate EXACT features used during training
    """

    df = df.copy()

    # ---- LAG FEATURES ----
    for i in range(7):
        df[f"lg{i+1}"] = df["powerGW"].shift(i+1).round(4)

    # ---- STAT FEATURES ----
    lag_cols = [f"lg{i}" for i in range(1, 8)]
    df["lgmn7"] = df[lag_cols].mean(axis=1)
    df["lgvr7"] = df[lag_cols].var(axis=1)
    df["lgsd7"] = df[lag_cols].std(axis=1)

    # ---- CALENDAR FEATURES ----
    df["nyweek"] = df.index.isocalendar().week.astype(int)
    df["nmweek"] = df.index.month
    df["ndweek"] = df.index.dayofweek

    # remove rows where lag not available
    df.dropna(inplace=True)

    # reorder columns EXACTLY as training
    cols = df.columns.tolist()

    df = df[cols[1:] + [cols[0]]]

    return df


def split_xy(df):
    """
    split features and target
    """

    X = df[df.columns.difference(["powerGW"], sort=False)]
    y = df[["powerGW"]]

    return X, y

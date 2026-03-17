import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    """Root Mean Squared Error (RMSE).

    Accepts array-like inputs and returns a scalar.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error (wrapper around sklearn)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred, eps: float = 1e-6):
    """Mean Absolute Percentage Error (in percent).

    Uses `eps` to avoid division by zero when y_true contains zeros.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100

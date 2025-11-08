"""
data_utils.py
Data loading and preprocessing utilities for DLEVS.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

def load_data(filepath):
    """Load CSV data as DataFrame."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Loaded data from {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {filepath}: {e}")
        raise

def scale_data(values):
    """Scale data using MinMaxScaler, ignoring NaNs."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = values[~np.isnan(values)]
    scaled = scaler.fit_transform(values.reshape(-1, 1))
    return scaled, scaler

def split_sequence(sequence, look_back, forecast_horizon):
    """Split a univariate sequence into samples for supervised learning."""
    X, y = list(), list()
    for i in range(len(sequence)):
        lag_end = i + look_back
        forecast_end = lag_end + forecast_horizon
        if forecast_end > len(sequence):
            break
        seq_x, seq_y = sequence[i:lag_end], sequence[lag_end:forecast_end]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

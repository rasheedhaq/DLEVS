"""
model_zoo.py
Model architectures for DLEVS: LSTM, GRU, CNN, CNN-LSTM, CNN-GRU.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm(input_shape, forecast_range):
    model = models.Sequential([
        layers.LSTM(32, input_shape=input_shape),
        layers.RepeatVector(forecast_range),
        layers.Dense(30, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(forecast_range),
        layers.Lambda(lambda x: x * 400)
    ])
    return model

def build_gru(input_shape, forecast_range):
    model = models.Sequential([
        layers.GRU(32, input_shape=input_shape),
        layers.RepeatVector(forecast_range),
        layers.Dense(30, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(forecast_range),
        layers.Lambda(lambda x: x * 400)
    ])
    return model

def build_cnn(input_shape, forecast_range):
    model = models.Sequential([
        layers.Conv1D(32, 2, input_shape=input_shape),
        layers.MaxPooling1D(pool_size=1),
        layers.Flatten(),
        layers.RepeatVector(forecast_range),
        layers.Dense(30, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(forecast_range),
        layers.Lambda(lambda x: x * 400)
    ])
    return model

def build_cnn_lstm(input_shape, forecast_range):
    model = models.Sequential([
        layers.Conv1D(32, 2, activation="relu", input_shape=input_shape),
        layers.MaxPooling1D(pool_size=1),
        layers.Flatten(),
        layers.RepeatVector(forecast_range),
        layers.LSTM(32, activation='relu', return_sequences=True),
        layers.Dense(forecast_range)
    ])
    return model

def build_cnn_gru(input_shape, forecast_range):
    model = models.Sequential([
        layers.Conv1D(32, 2, activation="relu", input_shape=input_shape),
        layers.MaxPooling1D(pool_size=1),
        layers.Flatten(),
        layers.RepeatVector(forecast_range),
        layers.GRU(32, activation='relu', return_sequences=True),
        layers.Dense(forecast_range)
    ])
    return model

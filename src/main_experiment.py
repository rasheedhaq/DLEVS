"""
main_experiment.py
Main experiment script for DLEVS: loads data, trains models, evaluates results.
"""
import logging
import numpy as np
import tensorflow as tf
from data.data_utils import load_data, scale_data, split_sequence
from models.model_zoo import build_lstm, build_gru, build_cnn, build_cnn_lstm, build_cnn_gru
from evaluation.metrics import evaluate_forecast
from training.train import train_model

logging.basicConfig(level=logging.INFO)

def run_experiment(data_path, look_back=30, forecast_range=1, epochs=10, batch_size=32, lr=0.0008):
    df = load_data(data_path)
    values = df['DO'].values  # Adjust column as needed
    scaled, scaler = scale_data(values)
    train_size = int(len(scaled) * 0.8)
    train, test = scaled[:train_size], scaled[train_size:]
    X_train, y_train = split_sequence(train, look_back, forecast_range)
    X_test, y_test = split_sequence(test, look_back, forecast_range)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    models = {
        'LSTM': build_lstm((look_back, 1), forecast_range),
        'GRU': build_gru((look_back, 1), forecast_range),
        'CNN': build_cnn((look_back, 1), forecast_range),
        'CNN-LSTM': build_cnn_lstm((look_back, 1), forecast_range),
        'CNN-GRU': build_cnn_gru((look_back, 1), forecast_range)
    }
    results = {}
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for name, model in models.items():
        logging.info(f"Training {name} model...")
        train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size, optimizer)
        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, forecast_range))
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, forecast_range))
        metrics = evaluate_forecast(y_test_inv, y_pred_inv)
        results[name] = metrics
        logging.info(f"{name} results: {metrics}")
    return results

if __name__ == "__main__":
    # Example usage
    results = run_experiment("../data/WBG_ADAK.csv")
    print(results)

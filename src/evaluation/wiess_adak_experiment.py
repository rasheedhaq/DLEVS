"""
wiess_adak_experiment.py
Script to evaluate DO estimation equations on ADAK dataset (modularized from Wiess_ADAK.ipynb).
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def metrics_time_series(y_true, y_pred):
    mae = round(mean_absolute_error(y_true, y_pred), 7)
    mse = round(mean_squared_error(y_true, y_pred), 7)
    rmse = round((mean_squared_error(y_true, y_pred) ** 0.5), 7)
    mape = round(mean_absolute_percentage_error(y_true, y_pred), 7)
    r2 = round(r2_score(y_true, y_pred), 7)
    return mae, mse, rmse, mape, r2

def main(data_path):
    df = pd.read_csv(data_path)
    do = df['DO_org']
    dow = df['DO_w']
    dob = df['DO_b']
    dog = df['DO_g']
    results = []
    for name, pred in zip(["Weiss", "Benson", "Garcia"], [dow, dob, dog]):
        mae, mse, rmse, mape, r2 = metrics_time_series(do, pred)
        results.append({
            'Name': name,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        })
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df

if __name__ == "__main__":
    # Example usage
    main("../../data/WBG_ADAK.csv")

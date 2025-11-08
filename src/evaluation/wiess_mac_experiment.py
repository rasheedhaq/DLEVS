"""
wiess_mac_experiment.py
Script to evaluate DO estimation equations on MAC dataset (modularized from Wiess_MAC.ipynb).
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def metrics_time_series(y_true, y_pred):
    mae = round(mean_absolute_error(y_true, y_pred), 7)
    mse = round(mean_squared_error(y_true, y_pred), 7)
    rmse = round((mean_squared_error(y_true, y_pred) ** 0.5), 7)
    mape = round(mean_absolute_percentage_error(y_true, y_pred), 7)
    r2 = round(r2_score(y_true, y_pred), 7)
    return mae, mse, rmse, mape, r2

def main(data_path):
    df = pd.read_excel(data_path)
    do = df['DO_org'][2:4800]
    dow = df['DO_w'][2:4800]
    dob = df['DO_b'][2:4800]
    dog = df['DO_g'][2:4800]
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
    # Plotting
    fig = plt.figure(figsize=(10,5))
    plt.plot(do,  linewidth=1.0)
    plt.plot(dow, linewidth=1.0)
    plt.plot(dob, linewidth=1.0)
    plt.plot(dog, linewidth=1.0)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('DO (mg/L)', fontsize=12)
    plt.legend(['Actual value', 'Weiss Equation','Benson & Krause Equation','Garcia & Gordon Equation'],fontsize=12, loc='best')
    plt.grid(linestyle=':')
    fig.savefig('figVS1MAC.pdf', dpi=600)
    return results_df

if __name__ == "__main__":
    # Example usage
    main("../../data/WBG_MAC.xlsx")

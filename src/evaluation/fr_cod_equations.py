"""
fr_cod_equations.py
Implements Weiss, Benson & Krause, and Garcia & Gordon DO equations and error analysis for DLEVS.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def weiss_equation(S, Tc):
    T = Tc + 273.15
    dW0 = 1.42905 * np.exp(-173.4292 + 249.6339 * (100 / T) + 143.3483 * (np.log(T / 100)) - 21.8492 * (T / 100))
    FsW = np.exp(S * (-0.033096 + 0.014259 * (T / 100) - 0.0017 * ((T / 100) ** 2)))
    return dW0 * FsW

def benson_krause_equation(S, Tc):
    T = Tc + 273.15
    dB0 = np.exp(-139.3411 + ((1.575701e5) / T) - ((6.642308e7) / (T ** 2)) +
                ((1.243800e10) / (T ** 3)) - ((8.621949e11) / (T ** 4)))
    FsB = np.exp(-S * (0.017674 - (10.754 / T) + (2140.7 / (T ** 2))))
    return dB0 * FsB

def garcia_gordon_equation(S, Tc):
    Ts = np.log((298.15 - Tc) / (273.15 + Tc))
    dG0 = 1.42905 * np.exp(2.00907 + 3.22014 * Ts + (4.05010 * (Ts ** 2)) + (4.94457 * (Ts ** 4)) + (3.88767 * (Ts ** 5)))
    FsG = np.exp((-0.00624523 - (0.00737614 * Ts) - (0.0103410 * (Ts ** 2)) - (0.00817083 * (Ts ** 3))) * S - (4.88682e-7) * (S ** 2))
    return dG0 * FsG

def compute_do_metrics(DOorg, *DO_models):
    metrics = []
    for DOmodel in DO_models:
        mae = mean_absolute_error(DOorg, DOmodel)
        mse = mean_squared_error(DOorg, DOmodel)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((DOorg - DOmodel) / DOorg)) * 100
        metrics.append({'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape})
    return metrics

def make_latex_table(metrics, names):
    latex = """
\\begin{table}[htbp]
\\centering
\\caption{Error Metrics for DO Equations}
\\begin{tabular}{lcccc}
\\toprule
Equation & MAE & MSE & RMSE & MAPE (\%) \\
\\midrule
"""
    for name, m in zip(names, metrics):
        latex += f"{name} & {m['mae']:.4f} & {m['mse']:.4f} & {m['rmse']:.4f} & {m['mape']:.2f} \\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}"
    return latex

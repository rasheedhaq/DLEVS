"""
fr_cod_experiment.py
Script to run DO equation comparison and error analysis for DLEVS (modularized from Fr_cod.ipynb).
"""
import pandas as pd
import numpy as np
from evaluation.fr_cod_equations import weiss_equation, benson_krause_equation, garcia_gordon_equation, compute_do_metrics, make_latex_table


def main(data_path, output_table_path=None):
    df = pd.read_csv(data_path)
    S, ph, DO, Tc = df["SAL"], df["PH"], df["DO"], df["TEMP"]
    DOorg = DO * 1.42905
    DOWeiss = weiss_equation(S, Tc)
    DOBen = benson_krause_equation(S, Tc)
    DOGar = garcia_gordon_equation(S, Tc)
    metrics = compute_do_metrics(DOorg, DOWeiss, DOBen, DOGar)
    names = ["Weiss", "Benson and Krause", "Garcia and Gordon"]
    latex_table = make_latex_table(metrics, names)
    print("LaTeX Table Code:\n", latex_table)
    if output_table_path:
        with open(output_table_path, "w") as f:
            f.write(latex_table)
    return metrics, latex_table

if __name__ == "__main__":
    # Example usage
    main("../../data/data_adak.csv", output_table_path="table.txt")

from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_residual_distribution(df : pd.DataFrame) -> None:

    residuals_all = defaultdict(list)

    grouped = df.groupby(['event_number', 'particle_id'])
    for (event, pid), group in grouped:
        fit_rows = group[group['is_vertex'] == 0]
        truth_rows = group[group['is_vertex'] == 1]

        if fit_rows.empty or truth_rows.empty:
            continue  # skip incomplete tracks

        final_fit = fit_rows.iloc[-1]
        truth = truth_rows.iloc[0]

        for param in ['d0', 'phi0', 'q/pT', 'z0', 'theta']:
            residual = final_fit[param] - truth[param]
            residuals_all[param].append(residual)

    #* Plot
    for param, data in residuals_all.items():
        plt.figure(figsize=(8, 5))
        mean = np.mean(data)
        std = np.std(data)

        sns.histplot(data, kde=True, bins=30, color="skyblue", edgecolor="black", linewidth=0.5)
        plt.axvline(mean, color='red', linestyle='--', label=f"μ = {mean:.5f}")
        plt.axvline(mean + std, color='green', linestyle='--', label=f"σ = {std:.5f}")
        plt.axvline(mean - std, color='green', linestyle='--')
        plt.title(f"{param} Residuals")
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_pull_distribution(df : pd.DataFrame) -> None:

    pulls_all = defaultdict(list)

    grouped = df.groupby(['event_number', 'particle_id'])
    for (event, pid), group in grouped:
        fit_rows = group[group['is_vertex'] == 0]
        truth_rows = group[group['is_vertex'] == 1]

        if fit_rows.empty or truth_rows.empty:
            continue

        final_fit = fit_rows.iloc[-1]
        truth = truth_rows.iloc[0]

        for param, cov in zip(['d0', 'phi0', 'q/pT', 'z0', 'theta'],
                              ['P00', 'P11', 'P22', 'P33', 'P44']):
            sigma = np.sqrt(final_fit[cov])
            if sigma > 0:
                pull = (final_fit[param] - truth[param]) / sigma
                pulls_all[param].append(pull)

    # Plot
    for param, data in pulls_all.items():
        plt.figure(figsize=(8, 5))
        mean = np.mean(data)
        std = np.std(data)

        sns.histplot(data, kde=True, bins=30, color="mediumpurple", edgecolor="black", linewidth=0.5)
        plt.axvline(mean, color='red', linestyle='--', label=f"μ = {mean:.5f}")
        plt.axvline(mean + std, color='green', linestyle='--', label=f"σ = {std:.5f}")
        plt.axvline(mean - std, color='green', linestyle='--')
        plt.title(f"{param} Pull Distribution")
        plt.xlabel("Pull")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



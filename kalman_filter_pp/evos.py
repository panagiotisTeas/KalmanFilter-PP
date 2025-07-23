import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_fit_parameter_evolution(df : pd.DataFrame, event_number : int, particle_id : int) -> None:

    # Filter track
    track = df[(df['event_number'] == event_number) &
               (df['particle_id'] == particle_id) &
               (df['is_vertex'] == 0)].copy()

    if track.empty:
        print("No fit steps found for this particle.")
        return
    
    track = track.iloc[1:]  # Skip first step

    params = ['d0', 'phi0', 'q/pT', 'z0', 'theta']
    covs = ['P00', 'P11', 'P22', 'P33', 'P44']
    steps = range(len(track))

    fig, axes = plt.subplots(len(params), 1, figsize=(10, 2.8 * len(params)), sharex=True)

    for i, (param, cov) in enumerate(zip(params, covs)):
        y = track[param]
        sigma = np.sqrt(track[cov])

        axes[i].plot(steps, y, label=param, color="mediumblue")
        axes[i].fill_between(steps, y - sigma, y + sigma, alpha=0.3, color="skyblue", label="±1σ")
        axes[i].set_ylabel(param)
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel("Fit Step")
    fig.suptitle(f"Parameter Evolution for Event {event_number}, Particle {particle_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_column_evolution(df : pd.DataFrame, event_number : int, particle_id : int, column_name : str) -> None:

    # Filter track
    track = df[(df['event_number'] == event_number) &
               (df['particle_id'] == particle_id) &
               (df['is_vertex'] == 0)].copy()
    
    track = track.iloc[1:]  # Skip first step

    if track.empty:
        print(f"No data found for event {event_number}, particle {particle_id}.")
        return
    if column_name not in track.columns:
        print(f"Column '{column_name}' not found.")
        return

    steps = range(len(track))
    y = track[column_name]

    plt.figure(figsize=(10, 4))
    sns.lineplot(x=steps, y=y, marker="o", label=column_name)
    plt.title(f"{column_name} Evolution - Event {event_number}, Particle {particle_id}")
    plt.xlabel("Fit Step")
    plt.ylabel(column_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
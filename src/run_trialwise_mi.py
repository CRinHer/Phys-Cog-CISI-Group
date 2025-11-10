"""
run_trialwise_mi.py
Trial-by-trial Mutual Information analysis using NeuralMI, with checkpointing.

This script supports:
  - automatic saving after each trial
  - resume from previous progress
  - customizable lag range and model parameters
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import neural_mi as nmi
from neural_mi.training.trainer import TrainingError


def run_trialwise_mi(
    x_trials, 
    y_trials,
    base_params=None,
    lag_range=None,
    random_seed=42,
    label="Condition",
    ch_pair=None,
    out_dir="../data/processed/mi_results",
    resume=True,
):
    """
    Run trial-by-trial Mutual Information vs lag using NeuralMI, saving progress after each trial.

    Parameters
    ----------
    x_trials, y_trials : np.ndarray
        EEG data arrays of shape (n_trials, n_timepoints)
    base_params : dict, optional
        NeuralMI model hyperparameters
    lag_range : range, optional
        Range of integer lag offsets in samples (e.g., range(-500, 501, 20))
    random_seed : int, default=42
        Random seed for reproducibility
    label : str, default="Condition"
        Descriptive label (e.g., "Auditory")
    ch_pair : tuple(str, str), optional
        Pair of channel names (e.g., ("C3", "T7"))
    out_dir : str or Path, default="../data/processed/mi_results"
        Directory for saving progress and plots
    resume : bool, default=True
        Whether to resume from saved progress if files exist

    Returns
    -------
    summary_df : pd.DataFrame
        Per-trial summary of max MI and optimal lag
    all_curves, all_curves_shuf : list of tuples
        List of (lags, MI values) for each trial
    """

    # --- Default NeuralMI configuration --- #
    if base_params is None:
        base_params = {
            'embedding_dim': 16,
            'hidden_dim': 32,
            'n_layers': 2,
            'learning_rate': 1e-3,
            'batch_size': 128,
            'n_epochs': 300,
            'patience': 30,
        }

    # --- Default lag range --- #
    if lag_range is None:
        lag_range = range(-600, 600, 40)

    sweep_grid = {'critic_type': ['separable']}

    # --- Prepare output paths --- #
    out_dir = Path(out_dir)
    ch_label = f"{ch_pair[0]}_{ch_pair[1]}" if ch_pair else "unknown_channels"
    save_dir = out_dir / f"{label.lower()}_{ch_label}"
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_path = save_dir / "summary.csv"
    curves_path = save_dir / "all_curves.npy"
    curves_shuf_path = save_dir / "all_curves_shuf.npy"

    # --- Resume previous progress --- #
    if resume and summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        completed_trials = set(summary_df['trial'].tolist())
        print(f"🔁 Resuming from {len(completed_trials)} completed trials.")
    else:
        summary_df = pd.DataFrame(columns=['trial', 'max_mi', 'opt_lag', 'max_mi_shuf'])
        completed_trials = set()

    # Load existing curve arrays (if any)
    if resume and curves_path.exists() and curves_shuf_path.exists():
        all_curves = list(np.load(curves_path, allow_pickle=True))
        all_curves_shuf = list(np.load(curves_shuf_path, allow_pickle=True))
    else:
        all_curves, all_curves_shuf = [], []

    # --- Trial loop --- #
    for i in tqdm(range(x_trials.shape[0]), desc=f"Trials ({label})"):
        trial_num = i + 1
        if trial_num in completed_trials:
            continue  # Skip completed

        x = x_trials[i][None, :]
        y = y_trials[i][None, :]

        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            print(f"Skipping trial {trial_num} (flat signal)")
            continue

        rng = np.random.RandomState(random_seed)
        y_shuf = rng.permutation(y.squeeze())[None, :]

        try:
            # --- NeuralMI estimation --- #
            res = nmi.run(
                x_data=x, y_data=y,
                mode='lag',
                processor_type_x='continuous', processor_params_x={'window_size': 32},
                processor_type_y='continuous', processor_params_y={'window_size': 32},
                base_params=base_params,
                lag_range=lag_range,
                split_mode='random',
                n_workers=1, random_seed=random_seed
            )

            res_shuf = nmi.run(
                x_data=x, y_data=y_shuf,
                mode='lag',
                processor_type_x='continuous', processor_params_x={'window_size': 32},
                processor_type_y='continuous', processor_params_y={'window_size': 32},
                base_params=base_params,
                lag_range=lag_range,
                split_mode='random',
                n_workers=1, random_seed=random_seed
            )

        except TrainingError:
            print(f"⚠️ Trial {trial_num}: training failed, skipping.")
            continue
        except Exception as e:
            print(f"⚠️ Trial {trial_num} crashed: {e}")
            continue

        # --- Extract results --- #
        lags = res.dataframe['lag'].to_numpy()
        mi_vals = res.dataframe['mi_mean'].to_numpy()
        mi_shuf = res_shuf.dataframe['mi_mean'].to_numpy()

        all_curves.append((lags, mi_vals))
        all_curves_shuf.append((lags, mi_shuf))

        max_idx = np.argmax(mi_vals)
        new_row = pd.DataFrame([{
            'trial': trial_num,
            'max_mi': mi_vals[max_idx],
            'opt_lag': lags[max_idx],
            'max_mi_shuf': np.max(mi_shuf)
        }])
        summary_df = pd.concat([summary_df, new_row], ignore_index=True)

        # --- Save progress after every trial --- #
        summary_df.to_csv(summary_path, index=False)
        np.save(curves_path, np.array(all_curves, dtype=object))
        np.save(curves_shuf_path, np.array(all_curves_shuf, dtype=object))

        # --- Plot MI vs lag for this trial --- #
        plt.figure(figsize=(8, 4))
        plt.plot(lags, mi_vals, label="Original", color='C0')
        plt.plot(lags, mi_shuf, '--', label="Shuffled", color='C1')
        plt.xlabel("Lag (samples)")
        plt.ylabel("MI (bits)")
        plt.title(f"Trial {trial_num} — {label} ({ch_pair[0]}–{ch_pair[1]})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"trial_{trial_num:03d}_mi_vs_lag.png", dpi=200)
        plt.close()

    print(f"✅ Completed {len(summary_df)} successful trials out of {x_trials.shape[0]}")
    return summary_df, all_curves, all_curves_shuf


# --- Optional command line interface --- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run trial-wise MI analysis")
    parser.add_argument("--lag_min", type=int, default=-600, help="Minimum lag (samples)")
    parser.add_argument("--lag_max", type=int, default=600, help="Maximum lag (samples)")
    parser.add_argument("--lag_step", type=int, default=40, help="Lag step size (samples)")
    parser.add_argument("--label", type=str, default="Condition", help="Condition label")
    parser.add_argument("--resume", action="store_true", help="Resume from saved progress")
    args = parser.parse_args()

    lag_range = range(args.lag_min, args.lag_max, args.lag_step)
    print(f"Lag range: {lag_range.start} to {lag_range.stop} step {lag_range.step}")
    print("👉 Use this function within a Jupyter Notebook for full control.")

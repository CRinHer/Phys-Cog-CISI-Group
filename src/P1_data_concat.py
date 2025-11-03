import numpy as np
import mne
from pathlib import Path

# ----------------- CONFIG ----------------- #
DATA_FOLDER = Path("../data/raw/P1-20251027T182958Z-1-001/P1/")
OUT_AUD_FILE = Path("../data/processed/P1_all_subjects_concat_auditory.npy")
OUT_TAC_FILE = Path("../data/processed/P1_all_subjects_concat_tactile.npy")

SUBJECT_IDS = [x for x in range(2, 18) if x not in (4, 14)]
SET_TEMPLATE = "binepochs filtered ICArej P1AvgBOS{num}.set"

APPLY_BANDPASS = True
BANDPASS = (1.0, 30.0)
ZSCORE_PER_EPOCH = True
BASELINE = (None, 0.0)
# ------------------------------------------ #

def process_epochs(epochs):
    """Applies baseline, filtering, and z-scoring, then concatenates epochs."""
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']

    if APPLY_BANDPASS:
        for e in range(len(data)):
            data[e] = mne.filter.filter_data(
                data[e], sfreq,
                l_freq=BANDPASS[0],
                h_freq=BANDPASS[1],
                verbose=False
            )

    for e in range(len(data)):
        means = data[e].mean(axis=1, keepdims=True)
        stds = data[e].std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        data[e] = (data[e] - means) / stds

    # Concatenate all epochs in time
    concat = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    return concat


# ------------- MAIN LOOP ------------- #
auditory_all = []
tactile_all = []
common_ch_names = None

for subj in SUBJECT_IDS:
    fname = SET_TEMPLATE.format(num=subj)
    set_path = DATA_FOLDER / fname
    if not set_path.exists():
        print(f"Skipping missing subject {subj}")
        continue

    print(f"Processing subject {subj}...")

    epochs = mne.io.read_epochs_eeglab(set_path)
    if BASELINE is not None:
        epochs.apply_baseline(BASELINE)

    # Split by stimulus type
    auditory_epochs = epochs[::2]
    tactile_epochs = epochs[1::2]

    auditory_concat = process_epochs(auditory_epochs)
    tactile_concat = process_epochs(tactile_epochs)

    if common_ch_names is None:
        common_ch_names = epochs.ch_names
    else:
        assert epochs.ch_names == common_ch_names, f"Channel mismatch in subject {subj}!"

    auditory_all.append(auditory_concat)
    tactile_all.append(tactile_concat)

# --- Stack all subjects ---
auditory_grand = np.concatenate(auditory_all, axis=1)
tactile_grand = np.concatenate(tactile_all, axis=1)

print(f"Auditory grand shape: {auditory_grand.shape}")
print(f"Tactile grand shape:  {tactile_grand.shape}")

# --- Save ---
np.save(OUT_AUD_FILE, auditory_grand)
np.save(OUT_TAC_FILE, tactile_grand)

print(f"Saved auditory data to {OUT_AUD_FILE}")
print(f"Saved tactile data to  {OUT_TAC_FILE}")

import numpy as np
import mne
from pathlib import Path

# ----------------- CONFIG ----------------- #
DATA_FOLDER = Path("../data/raw/P1-20251027T182958Z-1-001/P1/")
OUT_FILE = Path("../data/processed/P1_all_subjects_concat.npy")

SUBJECT_IDS = [x for x in range(2, 18) if x not in (4, 14)]
SET_TEMPLATE = "binepochs filtered ICArej P1AvgBOS{num}.set"

APPLY_BANDPASS = True
BANDPASS = (1.0, 30.0)
ZSCORE_PER_EPOCH = True
BASELINE = (None, 0.0)
# ------------------------------------------ #

def process_subject(set_path):
    epochs = mne.io.read_epochs_eeglab(set_path)
    if BASELINE is not None:
        epochs.apply_baseline(BASELINE)
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']

    if APPLY_BANDPASS:
        for e in range(len(data)):
            data[e] = mne.filter.filter_data(data[e], sfreq,
                                             l_freq=BANDPASS[0],
                                             h_freq=BANDPASS[1],
                                             verbose=False)

    # z-score per epoch
    for e in range(len(data)):
        means = data[e].mean(axis=1, keepdims=True)
        stds = data[e].std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        data[e] = (data[e] - means) / stds

    # concatenate epochs along time
    return data.transpose(1, 0, 2).reshape(data.shape[1], -1), epochs.ch_names

# ------------- MAIN LOOP ------------- #
all_subjects = []
common_ch_names = None

for subj in SUBJECT_IDS:
    fname = SET_TEMPLATE.format(num=subj)
    set_path = DATA_FOLDER / fname
    if not set_path.exists():
        print(f"Skipping missing subject {subj}")
        continue
    print(f"Processing subject {subj}...")
    concat_data, ch_names = process_subject(set_path)

    if common_ch_names is None:
        common_ch_names = ch_names
    else:
        assert ch_names == common_ch_names, f"Channel mismatch in subject {subj}!"

    all_subjects.append(concat_data)

# Stack all subjects along time dimension
all_concat = np.concatenate(all_subjects, axis=1)  # shape (n_channels, total_timepoints_all_subjects)

print(f"Final concatenated shape: {all_concat.shape}")
np.save(OUT_FILE, all_concat)
print(f"Saved combined data to {OUT_FILE}")
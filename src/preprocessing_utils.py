import numpy as np
import mne
from pathlib import Path


def preprocess_subject_data(
    protocol_folder: str,
    subject_num: int,
    apply_bandpass: bool = True,
    bandpass_range: tuple = (1.0, 30.0),
    baseline: tuple = (None, 0.0),
    out_base: Path = Path("../data/processed"),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load, preprocess, and save EEG data for a single subject from a given protocol.

    Parameters
    ----------
    protocol_folder : str
        Folder under "../data/raw" containing the protocol data (e.g. "P1-20251027T182958Z-1-su001/P1").
    subject_num : int
        Subject number to load (e.g. 2, 3, 5...).
    apply_bandpass : bool, default=True
        Whether to apply bandpass filtering.
    bandpass_range : tuple(float, float), default=(1.0, 30.0)
        The low and high cutoff frequencies for the bandpass filter.
    baseline : tuple, default=(None, 0.0)
        Baseline correction period (passed to mne.Epochs.apply_baseline()).
    out_base : Path, default=Path("../data/processed")
        Base directory where processed data will be saved.

    Returns
    -------
    aud_data : np.ndarray
        Auditory condition data, shape (n_channels, n_trials, n_times).
    tac_data : np.ndarray
        Tactile condition data, shape (n_channels, n_trials, n_times).
    meta : dict
        Metadata dictionary containing sampling rate, channel names, and parameters.
    """

    # --- Setup paths --- #
    data_root = Path("../data/raw") / protocol_folder
    out_dir = out_base / f"{Path(protocol_folder).stem}_subject{subject_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_template = f"binepochs filtered ICArej P1AvgBOS{subject_num}.set"
    set_path = data_root / set_template

    if not set_path.exists():
        raise FileNotFoundError(f"Subject {subject_num} file not found at: {set_path}")

    print(f"Processing subject {subject_num} from {protocol_folder}...")
    epochs = mne.io.read_epochs_eeglab(set_path)
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names

    # --- Apply baseline correction --- #
    if baseline is not None:
        epochs.apply_baseline(baseline)

    # --- Split by condition --- #
    auditory_epochs = epochs[::2]
    tactile_epochs = epochs[1::2]

    # --- Helper function for preprocessing --- #
    def process_epochs(epochs_obj):
        data = epochs_obj.get_data()  # (n_epochs, n_channels, n_times)

        # Apply optional bandpass filtering
        if apply_bandpass:
            for e in range(len(data)):
                data[e] = mne.filter.filter_data(
                    data[e],
                    sfreq,
                    l_freq=bandpass_range[0],
                    h_freq=bandpass_range[1],
                    verbose=False,
                )

        # Z-score per epoch per channel
        for e in range(len(data)):
            means = data[e].mean(axis=1, keepdims=True)
            stds = data[e].std(axis=1, keepdims=True)
            stds[stds == 0] = 1.0
            data[e] = (data[e] - means) / stds

        # Rearrange to (channels, trials, time)
        return data.transpose(1, 0, 2)

    # --- Process auditory & tactile epochs --- #
    aud_data = process_epochs(auditory_epochs)
    tac_data = process_epochs(tactile_epochs)

    # --- Save results --- #
    np.save(out_dir / "auditory.npy", aud_data)
    np.save(out_dir / "tactile.npy", tac_data)

    meta = {
        "subject": subject_num,
        "protocol": protocol_folder,
        "sfreq": sfreq,
        "ch_names": ch_names,
        "bandpass_range": bandpass_range if apply_bandpass else None,
        "baseline": baseline,
    }
    np.save(out_dir / "meta.npy", meta, allow_pickle=True) # type: ignore

    print(f"Saved processed data for subject {subject_num} to {out_dir}")
    print(f"Shapes — Auditory: {aud_data.shape}, Tactile: {tac_data.shape}")

    return aud_data, tac_data, meta

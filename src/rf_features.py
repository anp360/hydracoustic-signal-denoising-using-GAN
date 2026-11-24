"""
Shared spectrogram feature/label extraction for the Random Forest
time-frequency classifier (src/train_rf.py, src/evaluate_rf.py).

The RF classifies each STFT cell of the *noisy* (mixed) spectrogram as
"clean signal" (1) or "ship noise" (0). Training labels come from the
ideal binary mask: for every cell, whichever of the known clean signal or
residual noise (mixed - clean) has more power is the ground-truth class.
Only features derived from the mixed spectrogram are used, since that is
all that is available at inference time.

Per cell, the feature vector is: the cell's own log magnitude, its 4
immediate neighbours in time/frequency, how far above its frequency row's
median level it sits, and its normalized frequency position.
"""
import numpy as np
import librosa

N_FFT = 512
HOP_LENGTH = N_FFT // 4

FEATURE_NAMES = [
    "log_mag_db",
    "neighbor_up_db",
    "neighbor_down_db",
    "neighbor_left_db",
    "neighbor_right_db",
    "dev_above_freq_median_db",
    "freq_norm",
]


def stft_db(signal, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Complex STFT and its per-clip-normalized log-magnitude (dB)."""
    S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S, S_db


def ideal_binary_mask(clean_signal, noise_signal, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """1 where the clean signal dominates a T-F cell, 0 where noise dominates."""
    clean_power = np.abs(librosa.stft(clean_signal, n_fft=n_fft, hop_length=hop_length)) ** 2
    noise_power = np.abs(librosa.stft(noise_signal, n_fft=n_fft, hop_length=hop_length)) ** 2
    return (clean_power > noise_power).astype(np.int64)


def compute_features(S_db):
    """Build the per-cell feature matrix from a (n_freq, n_time) log-magnitude
    spectrogram. Returns an array of shape (n_freq * n_time, len(FEATURE_NAMES)),
    flattened in row-major (freq-major) order matching S_db.flatten()."""
    n_freq, n_time = S_db.shape

    # 4 immediate neighbours in time/frequency, edge-padded by repeating the
    # boundary cell so every cell (including borders) has a well-defined value.
    padded = np.pad(S_db, pad_width=1, mode="edge")
    neighbor_up = padded[:-2, 1:-1]      # freq - 1
    neighbor_down = padded[2:, 1:-1]     # freq + 1
    neighbor_left = padded[1:-1, :-2]    # time - 1
    neighbor_right = padded[1:-1, 2:]    # time + 1

    freq_median = np.median(S_db, axis=1, keepdims=True) * np.ones((1, n_time))
    dev_above_freq_median = S_db - freq_median

    freq_norm = np.linspace(0.0, 1.0, n_freq)[:, None] * np.ones((1, n_time))

    feats = np.stack([
        S_db,
        neighbor_up,
        neighbor_down,
        neighbor_left,
        neighbor_right,
        dev_above_freq_median,
        freq_norm,
    ], axis=-1)

    return feats.reshape(-1, len(FEATURE_NAMES))

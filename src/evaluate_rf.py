"""
Evaluates the trained Random Forest T-F classifier on the held-out
validation set. The predicted per-cell "clean" mask is applied to the
noisy STFT and inverted back to audio, then scored exactly like the GAN in
evaluate.py:
    - SNR improvement (dB) vs the noisy input
    - Spectrogram before/after plots for a few example clips

Usage:
    python src/evaluate_rf.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import joblib
import librosa
import librosa.display

from config import MIXED_DIR, CHECKPOINT_DIR, OUTPUT_DIR, SAMPLE_RATE
from evaluate import snr_db
from rf_features import stft_db, compute_features


def denoise_with_mask(mixed_signal, clf, n_fft, hop_length):
    S, S_db = stft_db(mixed_signal, n_fft=n_fft, hop_length=hop_length)
    feats = compute_features(S_db)
    mask = clf.predict(feats).reshape(S_db.shape).astype(np.float32)
    denoised = librosa.istft(S * mask, hop_length=hop_length, length=len(mixed_signal))
    return denoised.astype(np.float32), mask


def main():
    data = np.load(os.path.join(MIXED_DIR, "val.npz"))
    mixed, clean = data["mixed"], data["clean"]

    bundle = joblib.load(os.path.join(CHECKPOINT_DIR, "rf_classifier.joblib"))
    clf, n_fft, hop_length = bundle["model"], bundle["n_fft"], bundle["hop_length"]

    denoised = np.stack([
        denoise_with_mask(m, clf, n_fft, hop_length)[0] for m in mixed
    ])

    snr_before = np.array([snr_db(c, m) for c, m in zip(clean, mixed)])
    snr_after = np.array([snr_db(c, d) for c, d in zip(clean, denoised)])

    print(f"Mean SNR before denoising: {snr_before.mean():.2f} dB")
    print(f"Mean SNR after denoising:  {snr_after.mean():.2f} dB")
    print(f"Mean SNR improvement:      {(snr_after - snr_before).mean():.2f} dB")

    print("\nFeature importances:")
    for name, imp in sorted(zip(bundle["feature_names"], clf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:22s} {imp:.4f}")

    # spectrogram plots for a handful of examples
    n_examples = min(4, len(mixed))
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = axes[None, :]

    for i in range(n_examples):
        for col, (sig, title) in enumerate([
            (mixed[i], "Noisy input"),
            (denoised[i], "RF-masked output"),
            (clean[i], "Clean target"),
        ]):
            S = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
            librosa.display.specshow(S, sr=SAMPLE_RATE, x_axis="time", y_axis="hz", ax=axes[i, col])
            axes[i, col].set_title(f"{title} (ex {i})")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "spectrogram_comparison_rf.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved spectrogram comparison to {out_path}")


if __name__ == "__main__":
    main()

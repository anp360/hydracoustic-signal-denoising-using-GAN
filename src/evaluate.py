"""
Evaluates the trained generator on the held-out validation set:
    - SNR improvement (dB) vs the noisy input
    - Spectrogram before/after plots for a few example clips

Usage:
    python src/evaluate.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import librosa.display

from config import MIXED_DIR, CHECKPOINT_DIR, OUTPUT_DIR, SAMPLE_RATE
import model  # noqa: F401  (registers the custom RandomLatent layer)


def snr_db(clean, estimate, eps=1e-8):
    noise = clean - estimate
    return 10 * np.log10((np.sum(clean ** 2) + eps) / (np.sum(noise ** 2) + eps))


def main():
    data = np.load(os.path.join(MIXED_DIR, "val.npz"))
    mixed, clean = data["mixed"], data["clean"]

    generator = tf.keras.models.load_model(
        os.path.join(CHECKPOINT_DIR, "generator_final.keras"), compile=False
    )

    inputs = mixed[..., None]
    denoised = generator.predict(inputs, batch_size=32).squeeze(-1)

    snr_before = np.array([snr_db(c, m) for c, m in zip(clean, mixed)])
    snr_after = np.array([snr_db(c, d) for c, d in zip(clean, denoised)])

    print(f"Mean SNR before denoising: {snr_before.mean():.2f} dB")
    print(f"Mean SNR after denoising:  {snr_after.mean():.2f} dB")
    print(f"Mean SNR improvement:      {(snr_after - snr_before).mean():.2f} dB")

    # spectrogram plots for a handful of examples
    n_examples = min(4, len(mixed))
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = axes[None, :]

    for i in range(n_examples):
        for col, (sig, title) in enumerate([
            (mixed[i], "Noisy input"),
            (denoised[i], "Denoised output"),
            (clean[i], "Clean target"),
        ]):
            S = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=512)), ref=np.max)
            librosa.display.specshow(S, sr=SAMPLE_RATE, x_axis="time", y_axis="hz", ax=axes[i, col])
            axes[i, col].set_title(f"{title} (ex {i})")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "spectrogram_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved spectrogram comparison to {out_path}")


if __name__ == "__main__":
    main()

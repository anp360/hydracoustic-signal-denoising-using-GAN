"""
Live-presentation demo: generates one simulated noisy clip and denoises it
with both the GAN and the Random Forest baseline side by side.

Usage:
    python src/demo.py --snr 0
    python src/demo.py --snr 0 --play
"""
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib
import tensorflow as tf

from config import CHECKPOINT_DIR, OUTPUT_DIR, SAMPLE_RATE, FRAME_SECONDS
from audio_utils import mix_at_snr, save_audio
from simulator import generate_clean_signal, generate_noise_signal
from evaluate import snr_db
from evaluate_rf import denoise_with_mask
import model  # noqa: F401  (registers the custom RandomLatent layer)

DEMO_DIR = os.path.join(OUTPUT_DIR, "demo")


def build_demo_clip(snr_db_target, clean_freq, noise_freq, seed):
    rng = np.random.RandomState(seed)
    clean = generate_clean_signal(clean_freq, duration=FRAME_SECONDS, sr=SAMPLE_RATE, rng=rng)
    noise = generate_noise_signal(noise_freq, duration=FRAME_SECONDS, sr=SAMPLE_RATE, rng=rng)
    mixed, clean_out = mix_at_snr(clean, noise, snr_db_target)
    return mixed, clean_out


def run_gan(generator, mixed):
    start = time.perf_counter()
    denoised = generator.predict(mixed[None, :, None], verbose=0).squeeze()
    elapsed = time.perf_counter() - start
    return denoised.astype(np.float32), elapsed


def run_rf(bundle, mixed):
    clf, n_fft, hop_length = bundle["model"], bundle["n_fft"], bundle["hop_length"]
    start = time.perf_counter()
    denoised, _ = denoise_with_mask(mixed, clf, n_fft, hop_length)
    elapsed = time.perf_counter() - start
    return denoised, elapsed


def save_spectrogram_figure(path, panels, n_fft=512):
    hop_length = n_fft // 4
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6))
    for ax, (sig, title) in zip(axes, panels):
        S = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
        librosa.display.specshow(S, sr=SAMPLE_RATE, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    # write to a temp file and replace atomically, in case a previous run's
    # image viewer still has `path` open (os.startfile leaves it open)
    tmp_path = path + ".tmp.png"
    plt.savefig(tmp_path, dpi=150)
    plt.close(fig)
    try:
        os.replace(tmp_path, path)
        return path
    except OSError:
        # `path` is still locked open by a previous viewer window; fall back
        # to a fresh filename rather than failing the whole demo run
        fallback_path = os.path.join(
            os.path.dirname(path), f"comparison_{int(time.time())}.png"
        )
        os.replace(tmp_path, fallback_path)
        print(f"(Note: {path} was locked by a still-open viewer; saved to {fallback_path} instead.)")
        return fallback_path


def open_image(path):
    try:
        os.startfile(path)
    except Exception as e:
        print(f"(Could not auto-open image: {e})")


def play_in_order(paths_and_labels):
    import sounddevice as sd
    from audio_utils import load_audio
    for path, label in paths_and_labels:
        print(f"Playing: {label} ({path})")
        audio = load_audio(path, sr=SAMPLE_RATE)
        sd.play(audio, SAMPLE_RATE)
        sd.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=0.0, help="Mix SNR in dB for the demo clip.")
    parser.add_argument("--clean_freq", type=float, default=1000.0, help="Clean signal frequency (Hz).")
    parser.add_argument("--noise_freq", type=float, default=50.0, help="Ship noise fundamental frequency (Hz).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible signals.")
    parser.add_argument("--play", action="store_true", help="Play noisy/GAN/RF/clean audio in order via sounddevice.")
    args = parser.parse_args()

    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Generating demo clip: clean {args.clean_freq:.0f} Hz + noise {args.noise_freq:.0f} Hz "
          f"@ {args.snr:.1f} dB SNR ({FRAME_SECONDS:.1f}s @ {SAMPLE_RATE} Hz)")
    mixed, clean = build_demo_clip(args.snr, args.clean_freq, args.noise_freq, args.seed)

    print("Loading models...")
    generator = tf.keras.models.load_model(
        os.path.join(CHECKPOINT_DIR, "generator_final.keras"), compile=False
    )
    rf_bundle = joblib.load(os.path.join(CHECKPOINT_DIR, "rf_classifier.joblib"))

    gan_denoised, gan_time = run_gan(generator, mixed)
    rf_denoised, rf_time = run_rf(rf_bundle, mixed)

    snr_in = snr_db(clean, mixed)
    gan_improve = snr_db(clean, gan_denoised) - snr_in
    rf_improve = snr_db(clean, rf_denoised) - snr_in

    print(f"\nInput SNR: {snr_in:.2f} dB\n")
    header = f"{'Model':<12}{'SNR improvement (dB)':<24}{'Processing time (ms)':<22}"
    print(header)
    print(f"{'GAN':<12}{gan_improve:<24.2f}{gan_time * 1000:<22.1f}")
    print(f"{'RF':<12}{rf_improve:<24.2f}{rf_time * 1000:<22.1f}")

    fig_path = os.path.join(DEMO_DIR, "comparison.png")
    fig_path = save_spectrogram_figure(fig_path, [
        (mixed, "Noisy"),
        (gan_denoised, "GAN"),
        (rf_denoised, "Random Forest"),
        (clean, "Clean"),
    ])
    print(f"\nSaved comparison figure to {fig_path}")
    open_image(fig_path)

    wav_paths = {
        "noisy": os.path.join(DEMO_DIR, "noisy.wav"),
        "gan": os.path.join(DEMO_DIR, "gan.wav"),
        "rf": os.path.join(DEMO_DIR, "rf.wav"),
        "clean": os.path.join(DEMO_DIR, "clean.wav"),
    }
    save_audio(wav_paths["noisy"], mixed, sr=SAMPLE_RATE)
    save_audio(wav_paths["gan"], gan_denoised, sr=SAMPLE_RATE)
    save_audio(wav_paths["rf"], rf_denoised, sr=SAMPLE_RATE)
    save_audio(wav_paths["clean"], clean, sr=SAMPLE_RATE)
    print(f"Saved noisy/gan/rf/clean .wav files to {DEMO_DIR}")

    if args.play:
        play_in_order([
            (wav_paths["noisy"], "Noisy"),
            (wav_paths["gan"], "GAN"),
            (wav_paths["rf"], "Random Forest"),
            (wav_paths["clean"], "Clean"),
        ])


if __name__ == "__main__":
    main()

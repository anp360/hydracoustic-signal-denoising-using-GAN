"""
Denoise a new hydrophone .wav file of arbitrary length by processing it in
overlapping 1-second frames and cross-fading the outputs back together.

Usage:
    python src/infer.py --input some_recording.wav --output outputs/denoised.wav
"""
import argparse
import os

import numpy as np
import tensorflow as tf

from config import CHECKPOINT_DIR, FRAME_LEN, SAMPLE_RATE
from audio_utils import load_audio, save_audio
import model  # noqa: F401  (registers the custom RandomLatent layer)


def process_long_file(generator, audio, frame_len=FRAME_LEN, hop=None):
    hop = hop or frame_len // 2
    window = np.hanning(frame_len).astype(np.float32)

    n = len(audio)
    pad = frame_len - (n % hop) if n % hop else 0
    audio_padded = np.pad(audio, (0, pad + frame_len))

    output = np.zeros_like(audio_padded)
    norm = np.zeros_like(audio_padded)

    starts = list(range(0, len(audio_padded) - frame_len, hop))
    frames = np.stack([audio_padded[s:s + frame_len] for s in starts])[..., None]

    denoised_frames = generator.predict(frames, batch_size=32).squeeze(-1)

    for s, frame in zip(starts, denoised_frames):
        output[s:s + frame_len] += frame * window
        norm[s:s + frame_len] += window

    norm[norm == 0] = 1.0
    output = output / norm
    return output[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to noisy hydrophone .wav")
    parser.add_argument("--output", default="outputs/denoised.wav")
    args = parser.parse_args()

    generator = tf.keras.models.load_model(
        os.path.join(CHECKPOINT_DIR, "generator_final.keras"), compile=False
    )

    audio = load_audio(args.input, sr=SAMPLE_RATE)
    denoised = process_long_file(generator, audio)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_audio(args.output, denoised, sr=SAMPLE_RATE)
    print(f"Saved denoised audio to {args.output}")


if __name__ == "__main__":
    main()

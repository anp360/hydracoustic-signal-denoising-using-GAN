"""
Builds the paired (noisy, clean) training set.

Real-data mode (default):
    Reads every file in data/clean/ and data/noise/, crops random 1-second
    frames, mixes them at a random SNR, and saves the pairs to
    data/mixed/train.npz and data/mixed/val.npz.

Synthetic mode (--synthetic):
    Generates placeholder clean/noise signals with NumPy so you can run the
    full pipeline before you've downloaded ShipsEar/Watkins audio.

Usage:
    python src/make_mixed_dataset.py
    python src/make_mixed_dataset.py --synthetic --n_samples 500
"""
import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

from config import (
    CLEAN_DIR, NOISE_DIR, MIXED_DIR, FRAME_LEN, SNR_RANGE_DB,
    TRAIN_SPLIT, RANDOM_SEED,
)
from audio_utils import (
    load_audio, random_crop_or_pad, mix_at_snr,
    synthetic_clean_tone, synthetic_ship_noise,
)


def list_audio_files(directory):
    exts = ("*.wav", "*.WAV", "*.flac", "*.mp3")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(directory, e)))
    return files


def build_from_real_files(n_samples, rng):
    clean_files = list_audio_files(CLEAN_DIR)
    noise_files = list_audio_files(NOISE_DIR)
    if not clean_files or not noise_files:
        raise FileNotFoundError(
            f"No audio found in {CLEAN_DIR} or {NOISE_DIR}. "
            "Download ShipsEar/Watkins audio first, or run with --synthetic."
        )

    print(f"Found {len(clean_files)} clean files, {len(noise_files)} noise files.")
    # Pre-load full clips once (fine for hours of audio at 16kHz)
    clean_clips = [load_audio(f) for f in tqdm(clean_files, desc="loading clean")]
    noise_clips = [load_audio(f) for f in tqdm(noise_files, desc="loading noise")]

    mixed_list, clean_list = [], []
    for _ in tqdm(range(n_samples), desc="mixing pairs"):
        c = random_crop_or_pad(clean_clips[rng.randint(len(clean_clips))], FRAME_LEN, rng)
        n = random_crop_or_pad(noise_clips[rng.randint(len(noise_clips))], FRAME_LEN, rng)
        snr_db = rng.uniform(*SNR_RANGE_DB)
        mixed, clean_out = mix_at_snr(c, n, snr_db)
        mixed_list.append(mixed)
        clean_list.append(clean_out)
    return np.stack(mixed_list), np.stack(clean_list)


def build_synthetic(n_samples, rng):
    mixed_list, clean_list = [], []
    for _ in tqdm(range(n_samples), desc="synthesizing pairs"):
        c = synthetic_clean_tone(FRAME_LEN, rng=rng)
        n = synthetic_ship_noise(FRAME_LEN, rng=rng)
        snr_db = rng.uniform(*SNR_RANGE_DB)
        mixed, clean_out = mix_at_snr(c, n, snr_db)
        mixed_list.append(mixed)
        clean_list.append(clean_out)
    return np.stack(mixed_list), np.stack(clean_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                         help="Use synthetic placeholder audio instead of real files.")
    parser.add_argument("--n_samples", type=int, default=3000,
                         help="Total number of (noisy, clean) pairs to generate.")
    args = parser.parse_args()

    rng = np.random.RandomState(RANDOM_SEED)

    if args.synthetic:
        mixed, clean = build_synthetic(args.n_samples, rng)
    else:
        mixed, clean = build_from_real_files(args.n_samples, rng)

    n_train = int(len(mixed) * TRAIN_SPLIT)
    idx = rng.permutation(len(mixed))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    np.savez(os.path.join(MIXED_DIR, "train.npz"),
              mixed=mixed[train_idx], clean=clean[train_idx])
    np.savez(os.path.join(MIXED_DIR, "val.npz"),
              mixed=mixed[val_idx], clean=clean[val_idx])

    print(f"Saved {len(train_idx)} train pairs and {len(val_idx)} val pairs to {MIXED_DIR}")


if __name__ == "__main__":
    main()

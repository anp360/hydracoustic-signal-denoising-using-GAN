"""
Trains a Random Forest classifier as a non-regression baseline against the
GAN: it labels each STFT time-frequency cell of the noisy (mixed) signal as
"clean signal" (1) or "ship noise" (0), using the ideal binary mask derived
from the known clean/noise parts of the simulated mixes as ground truth.

Usage:
    python src/train_rf.py
    python src/train_rf.py --n_samples 300 --cells_per_sample 800
"""
import argparse
import os
import time

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from config import MIXED_DIR, CHECKPOINT_DIR, RANDOM_SEED
from rf_features import N_FFT, HOP_LENGTH, FEATURE_NAMES, stft_db, ideal_binary_mask, compute_features


def load_dataset(split):
    data = np.load(os.path.join(MIXED_DIR, f"{split}.npz"))
    return data["mixed"], data["clean"]


def build_training_table(mixed, clean, n_samples, cells_per_sample, rng):
    """Sample `n_samples` clips and `cells_per_sample` random T-F cells from
    each, computing mixed-spectrogram features and ideal-binary-mask labels."""
    n_samples = min(n_samples, len(mixed))
    sample_idx = rng.choice(len(mixed), size=n_samples, replace=False)

    X_parts, y_parts = [], []
    for i in sample_idx:
        noise = mixed[i] - clean[i]
        _, mixed_db = stft_db(mixed[i])
        labels = ideal_binary_mask(clean[i], noise)
        feats = compute_features(mixed_db)
        labels_flat = labels.flatten()

        n_cells = feats.shape[0]
        n_pick = min(cells_per_sample, n_cells)
        cell_idx = rng.choice(n_cells, size=n_pick, replace=False)
        X_parts.append(feats[cell_idx])
        y_parts.append(labels_flat[cell_idx])

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=300,
                         help="Number of training clips to draw spectrogram cells from.")
    parser.add_argument("--cells_per_sample", type=int, default=500,
                         help="Number of random T-F cells sampled per clip.")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=18)
    args = parser.parse_args()

    rng = np.random.RandomState(RANDOM_SEED)

    mixed_train, clean_train = load_dataset("train")
    mixed_val, clean_val = load_dataset("val")
    print(f"Loaded {len(mixed_train)} train clips, {len(mixed_val)} val clips "
          f"(STFT n_fft={N_FFT}, hop={HOP_LENGTH}).")

    print("Building training table from ideal binary mask labels...")
    X_train, y_train = build_training_table(
        mixed_train, clean_train, args.n_samples, args.cells_per_sample, rng
    )
    X_val, y_val = build_training_table(
        mixed_val, clean_val, args.n_samples, args.cells_per_sample, rng
    )
    print(f"Train cells: {X_train.shape[0]}, Val cells: {X_val.shape[0]}, "
          f"features: {X_train.shape[1]}")
    print(f"Class balance (train, frac clean): {y_train.mean():.3f}")

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )

    start = time.time()
    clf.fit(X_train, y_train)
    print(f"Trained Random Forest in {time.time() - start:.1f}s")

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")

    print("\nFeature importances:")
    importances = clf.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda t: -t[1]):
        print(f"  {name:22s} {imp:.4f}")

    model_path = os.path.join(CHECKPOINT_DIR, "rf_classifier.joblib")
    joblib.dump({
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
    }, model_path)
    print(f"\nSaved Random Forest model to {model_path}")


if __name__ == "__main__":
    main()

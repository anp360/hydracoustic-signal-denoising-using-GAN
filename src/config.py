"""
All hyperparameters and paths in one place. Edit this file rather than
hunting through the other scripts.
"""
import os

# ---- paths ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(ROOT, "data", "clean")
NOISE_DIR = os.path.join(ROOT, "data", "noise")
MIXED_DIR = os.path.join(ROOT, "data", "mixed")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# ---- audio ----
SAMPLE_RATE = 16000          # resample everything to this rate
FRAME_SECONDS = 1.0          # length of each training frame
FRAME_LEN = int(SAMPLE_RATE * FRAME_SECONDS)
SNR_RANGE_DB = (-5, 15)      # random SNR range used when mixing clean+noise

# ---- dataset building ----
TRAIN_SPLIT = 0.85
RANDOM_SEED = 42

# ---- model ----
GEN_FILTERS = [16, 32, 32, 64, 64, 128, 128]   # encoder channel sizes
KERNEL_SIZE = 31
STRIDE = 2
LATENT_NOISE_DIM = 8   # small stochastic latent concatenated at bottleneck

DISC_FILTERS = [16, 32, 64, 128, 256]

# ---- training ----
BATCH_SIZE = 16
EPOCHS = 10
LR_G = 2e-4
LR_D = 2e-4
BETA_1 = 0.5
L1_LAMBDA = 100.0     # weight on time-domain reconstruction loss
CHECKPOINT_EVERY = 5

for d in (CLEAN_DIR, NOISE_DIR, MIXED_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

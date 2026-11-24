"""
Audio I/O, resampling, framing, and clean+noise SNR mixing utilities.
"""
import numpy as np
import librosa
import soundfile as sf

from config import SAMPLE_RATE, FRAME_LEN


def load_audio(path, sr=SAMPLE_RATE):
    """Load a wav (or any librosa-supported) file, mono, resampled to sr."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def save_audio(path, audio, sr=SAMPLE_RATE):
    sf.write(path, audio, sr)


def random_crop_or_pad(signal, length=FRAME_LEN, rng=None):
    """Return a fixed-length crop of `signal`, looping/padding if too short."""
    rng = rng or np.random
    if len(signal) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(signal) < length:
        reps = int(np.ceil(length / len(signal)))
        signal = np.tile(signal, reps)
    start = rng.randint(0, len(signal) - length + 1) if len(signal) > length else 0
    return signal[start:start + length].astype(np.float32)


def rms(x, eps=1e-8):
    return np.sqrt(np.mean(x ** 2) + eps)


def mix_at_snr(clean, noise, snr_db):
    """
    Scale `noise` so that mixing with `clean` yields the requested SNR (dB),
    then return (mixed, clean) both length-matched to `clean`.
    SNR = 20*log10(rms(clean)/rms(scaled_noise))
    """
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scale = target_noise_rms / (noise_rms + 1e-8)
    scaled_noise = noise * scale
    mixed = clean + scaled_noise

    # peak-normalize both to avoid clipping, applying the same factor to
    # keep clean/mixed on the same scale
    peak = np.max(np.abs(mixed)) + 1e-8
    if peak > 1.0:
        mixed = mixed / peak
        clean_out = clean / peak
    else:
        clean_out = clean
    return mixed.astype(np.float32), clean_out.astype(np.float32)


def synthetic_clean_tone(length=FRAME_LEN, sr=SAMPLE_RATE, rng=None):
    """Placeholder 'clean marine sound': a few chirps + soft broadband hiss,
    used only when no real data is available yet (--synthetic mode)."""
    rng = rng or np.random
    t = np.arange(length) / sr
    n_chirps = rng.randint(1, 4)
    sig = np.zeros(length, dtype=np.float32)
    for _ in range(n_chirps):
        f0 = rng.uniform(500, 4000)
        f1 = f0 + rng.uniform(-500, 500)
        dur = rng.uniform(0.1, 0.4)
        start = rng.uniform(0, max(t[-1] - dur, 0))
        mask = (t >= start) & (t < start + dur)
        local_t = t[mask] - start
        freq = np.linspace(f0, f1, mask.sum())
        sig[mask] += 0.5 * np.sin(2 * np.pi * freq * local_t)
    sig += 0.02 * rng.randn(length)
    return sig.astype(np.float32)


def synthetic_ship_noise(length=FRAME_LEN, sr=SAMPLE_RATE, rng=None):
    """Placeholder 'ship engine noise': low-frequency hum + harmonics + broadband
    engine rumble, used only in --synthetic mode."""
    rng = rng or np.random
    t = np.arange(length) / sr
    fundamental = rng.uniform(20, 80)  # engine hum fundamental
    sig = np.zeros(length, dtype=np.float32)
    for harmonic in range(1, 6):
        amp = 0.3 / harmonic
        sig += amp * np.sin(2 * np.pi * fundamental * harmonic * t + rng.uniform(0, np.pi))
    # broadband rumble via filtered noise (simple moving-average low-pass)
    noise = rng.randn(length)
    kernel = np.ones(15) / 15
    rumble = np.convolve(noise, kernel, mode="same")
    sig += 0.3 * rumble
    return sig.astype(np.float32)

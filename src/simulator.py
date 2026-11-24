"""
Hydroacoustic Noise Simulator
=============================
An interactive desktop tool to simulate underwater acoustic signals:

  PATH A (Clean Path)  -> a synthetic "clean" marine sound at a chosen
                           frequency (stands in for whale/fish/biological sound)
  PATH B (Noise Path)  -> a synthetic ship-engine noise at a chosen
                           frequency (low-frequency hum + harmonics + rumble)
  MODE 3 (Mixed)       -> Path A and Path B combined at a chosen SNR (dB)
  DENOISE (RF)         -> runs the trained Random Forest T-F classifier on
                           the current Mixed signal and plays/plots the result

Three buttons let you generate and play each mode. Each generation is also
plotted (waveform + spectrogram) so you can *see* the effect, not just hear it.
A "Save to dataset" button drops the generated clip straight into this
project's data/clean, data/noise, or data/mixed folders, so simulator output
feeds directly into the GAN denoising pipeline (src/train.py etc).

Run:
    python src/simulator.py

Requires: numpy, scipy, matplotlib, sounddevice, scikit-learn, joblib, librosa
(all in requirements.txt). The Denoise (RF) button needs a trained model at
checkpoints/rf_classifier.joblib -- run `python train_rf.py` first.
"""
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import librosa
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

from config import SAMPLE_RATE, CLEAN_DIR, NOISE_DIR, MIXED_DIR, CHECKPOINT_DIR
from audio_utils import save_audio
from rf_features import stft_db, compute_features

RF_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "rf_classifier.joblib")


# ---------------------------------------------------------------------------
# PATH A: Clean signal generator (parameterized by frequency)
# ---------------------------------------------------------------------------
def generate_clean_signal(freq_hz, duration=3.0, sr=SAMPLE_RATE, rng=None):
    """Simulates a clean marine sound: a tonal sweep centered at freq_hz
    (stands in for a whale/dolphin call or biological signal) plus a touch
    of natural ambient hiss."""
    rng = rng or np.random
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sweep_width = freq_hz * 0.15
    f_inst = freq_hz + sweep_width * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    tone = 0.6 * np.sin(phase)
    envelope = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / duration, 0, 1)))  # fade in/out
    ambient_hiss = 0.02 * rng.randn(len(t))
    signal = tone * envelope + ambient_hiss
    return signal.astype(np.float32)


# ---------------------------------------------------------------------------
# PATH B: Noise generator (parameterized by frequency)
# ---------------------------------------------------------------------------
def generate_noise_signal(freq_hz, duration=3.0, sr=SAMPLE_RATE, rng=None):
    """Simulates ship-engine noise: a low-frequency hum at freq_hz plus
    harmonics, plus broadband engine rumble."""
    rng = rng or np.random
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros(len(t), dtype=np.float32)
    for harmonic in range(1, 6):
        amp = 0.35 / harmonic
        signal += amp * np.sin(2 * np.pi * freq_hz * harmonic * t + rng.uniform(0, np.pi))
    broadband = rng.randn(len(t))
    kernel = np.ones(21) / 21
    rumble = np.convolve(broadband, kernel, mode="same")
    signal += 0.3 * rumble
    return signal.astype(np.float32)


# ---------------------------------------------------------------------------
# MODE 3: Mix the two paths at a chosen SNR
# ---------------------------------------------------------------------------
def mix_signals(clean, noise, snr_db):
    clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-8)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-8)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    mixed = clean + scaled_noise
    peak = np.max(np.abs(mixed)) + 1e-8
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


# ---------------------------------------------------------------------------
# DENOISE: apply the trained Random Forest T-F mask to a mixed signal
# ---------------------------------------------------------------------------
def denoise_with_rf(signal, clf, n_fft, hop_length):
    """Classify every STFT cell as clean/noise with the Random Forest, zero
    out the noise cells, and invert back to a time-domain signal."""
    S, S_db = stft_db(signal, n_fft=n_fft, hop_length=hop_length)
    feats = compute_features(S_db)
    mask = clf.predict(feats).reshape(S_db.shape).astype(np.float32)
    denoised = librosa.istft(S * mask, hop_length=hop_length, length=len(signal))
    return denoised.astype(np.float32)


def snr_db(clean, estimate, eps=1e-8):
    noise = clean - estimate
    return 10 * np.log10((np.sum(clean ** 2) + eps) / (np.sum(noise ** 2) + eps))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class SimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hydroacoustic Noise Simulator")
        self.root.geometry("880x680")

        self.current_signal = None
        self.current_label = None
        self.current_clean_ref = None  # clean component of the last Mode 3 mix, for SNR
        self.rng = np.random.RandomState()

        try:
            self.rf_bundle = joblib.load(RF_MODEL_PATH)
        except Exception:
            self.rf_bundle = None

        self._build_controls()
        self._build_plot()

    # -- UI layout -----------------------------------------------------
    def _build_controls(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(frame, text="Clean signal frequency (Hz):").grid(row=0, column=0, sticky="w")
        self.clean_freq = tk.DoubleVar(value=1000)
        ttk.Scale(frame, from_=100, to=5000, variable=self.clean_freq,
                  orient="horizontal", length=250).grid(row=0, column=1, padx=8)
        self.clean_freq_lbl = ttk.Label(frame, text="1000 Hz")
        self.clean_freq_lbl.grid(row=0, column=2)
        self.clean_freq.trace_add("write", lambda *_: self.clean_freq_lbl.config(
            text=f"{int(self.clean_freq.get())} Hz"))

        ttk.Label(frame, text="Ship noise frequency (Hz):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.noise_freq = tk.DoubleVar(value=50)
        ttk.Scale(frame, from_=20, to=200, variable=self.noise_freq,
                  orient="horizontal", length=250).grid(row=1, column=1, padx=8, pady=(8, 0))
        self.noise_freq_lbl = ttk.Label(frame, text="50 Hz")
        self.noise_freq_lbl.grid(row=1, column=2, pady=(8, 0))
        self.noise_freq.trace_add("write", lambda *_: self.noise_freq_lbl.config(
            text=f"{int(self.noise_freq.get())} Hz"))

        ttk.Label(frame, text="Mix SNR (dB):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.snr_db = tk.DoubleVar(value=5)
        ttk.Scale(frame, from_=-10, to=20, variable=self.snr_db,
                  orient="horizontal", length=250).grid(row=2, column=1, padx=8, pady=(8, 0))
        self.snr_lbl = ttk.Label(frame, text="5 dB")
        self.snr_lbl.grid(row=2, column=2, pady=(8, 0))
        self.snr_db.trace_add("write", lambda *_: self.snr_lbl.config(
            text=f"{self.snr_db.get():.0f} dB"))

        # the three mode buttons
        btn_frame = ttk.Frame(self.root, padding=(12, 4))
        btn_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btn_frame, text="\u25b6 Path A: Clean Signal",
                   command=self.run_clean).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="\u25b6 Path B: Ship Noise",
                   command=self.run_noise).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="\u25b6 Mode 3: Mixed (Noisy)",
                   command=self.run_mixed).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="\U0001F333 Denoise (Random Forest)",
                   command=self.run_denoise_rf).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="\U0001F4BE Save to dataset",
                   command=self.save_current).pack(side=tk.LEFT, padx=20)

        self.status_lbl = ttk.Label(self.root, text="Pick a mode above to generate a signal.",
                                     padding=(12, 4), foreground="#2E5A8C")
        self.status_lbl.pack(side=tk.TOP, fill=tk.X)

    def _build_plot(self):
        self.fig, (self.ax_wave, self.ax_live, self.ax_spec) = plt.subplots(1, 3, figsize=(11, 3.6))
        self.fig.tight_layout(pad=3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # playhead line artists (created lazily on first draw)
        self.wave_playhead = None
        self.spec_playhead = None
        self._anim_after_id = None
        self._play_start_time = None

    # -- button actions --------------------------------------------------
    def run_clean(self):
        self.current_clean_ref = None
        sig = generate_clean_signal(self.clean_freq.get(), rng=self.rng)
        self._display_and_play(sig, "clean", f"Clean signal @ {int(self.clean_freq.get())} Hz")

    def run_noise(self):
        self.current_clean_ref = None
        sig = generate_noise_signal(self.noise_freq.get(), rng=self.rng)
        self._display_and_play(sig, "noise", f"Ship noise @ {int(self.noise_freq.get())} Hz")

    def run_mixed(self):
        clean = generate_clean_signal(self.clean_freq.get(), rng=self.rng)
        noise = generate_noise_signal(self.noise_freq.get(), rng=self.rng)
        mixed = mix_signals(clean, noise, self.snr_db.get())
        self.current_clean_ref = clean
        self._display_and_play(mixed, "mixed",
                                f"Mixed: clean {int(self.clean_freq.get())} Hz "
                                f"+ noise {int(self.noise_freq.get())} Hz @ {self.snr_db.get():.0f} dB SNR")

    def run_denoise_rf(self):
        if self.current_label != "mixed" or self.current_signal is None:
            messagebox.showwarning("No mixed signal",
                                    "Generate a Mode 3: Mixed signal first, then denoise it.")
            return
        if self.rf_bundle is None:
            messagebox.showerror(
                "Random Forest model not found",
                f"Could not load {RF_MODEL_PATH}.\nRun `python train_rf.py` first, then reopen the simulator."
            )
            return

        clf = self.rf_bundle["model"]
        n_fft, hop_length = self.rf_bundle["n_fft"], self.rf_bundle["hop_length"]
        mixed_signal = self.current_signal
        denoised = denoise_with_rf(mixed_signal, clf, n_fft, hop_length)

        title = (f"RF-denoised: clean {int(self.clean_freq.get())} Hz "
                 f"+ noise {int(self.noise_freq.get())} Hz @ {self.snr_db.get():.0f} dB SNR")
        self._display_and_play(denoised, "denoised_rf", title)

        if self.current_clean_ref is not None:
            before = snr_db(self.current_clean_ref, mixed_signal)
            after = snr_db(self.current_clean_ref, denoised)
            self.status_lbl.config(
                text=f"{title}  |  SNR {before:.1f} dB → {after:.1f} dB ({after - before:+.1f} dB)"
            )

    def save_current(self):
        if self.current_signal is None:
            messagebox.showwarning("Nothing to save", "Generate a signal first.")
            return
        target_map = {"clean": CLEAN_DIR, "noise": NOISE_DIR, "mixed": MIXED_DIR}
        if self.current_label not in target_map:
            messagebox.showwarning("Can't save this",
                                    "Only Clean / Noise / Mixed signals can be saved as dataset clips.")
            return
        target_dir = target_map[self.current_label]
        filename = f"sim_{self.current_label}_{int(time.time())}.wav"
        path = os.path.join(target_dir, filename)
        save_audio(path, self.current_signal, sr=SAMPLE_RATE)
        self.status_lbl.config(text=f"Saved to {path}")

    # -- shared display/playback logic -----------------------------------
    def _display_and_play(self, signal, label, title):
        # cancel any previous animation loop before starting a new one
        if self._anim_after_id is not None:
            self.root.after_cancel(self._anim_after_id)
            self._anim_after_id = None

        self.current_signal = signal
        self.current_label = label
        self.current_title = title
        self.status_lbl.config(text=f"Generated: {title}")

        duration = len(signal) / SAMPLE_RATE
        t = np.arange(len(signal)) / SAMPLE_RATE

        # full waveform (static reference, playhead moves across it)
        self.ax_wave.clear()
        self.ax_wave.plot(t, signal, linewidth=0.6, color="#2E5A8C")
        self.ax_wave.set_title(f"Waveform \u2014 {title}", fontsize=9)
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.wave_playhead = self.ax_wave.axvline(0, color="red", linewidth=1.2)

        # full spectrogram (static reference, playhead moves across it)
        self.ax_spec.clear()
        self.ax_spec.specgram(signal, Fs=SAMPLE_RATE, cmap="magma")
        self.ax_spec.set_title("Spectrogram", fontsize=9)
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.spec_playhead = self.ax_spec.axvline(0, color="cyan", linewidth=1.2)

        # live oscilloscope panel starts empty, fills in as sound "plays"
        self.ax_live.clear()
        self.ax_live.set_title("Live signal (scrolling)", fontsize=9)
        self.ax_live.set_xlabel("Time (s, relative)")
        self.ax_live.set_ylim(-1.05, 1.05)
        self.live_line, = self.ax_live.plot([], [], linewidth=0.8, color="#2E5A8C")

        self.canvas.draw()

        if AUDIO_AVAILABLE:
            try:
                sd.play(signal, SAMPLE_RATE)
            except Exception as e:
                self.status_lbl.config(text=f"Generated (audio playback failed: {e})")
        else:
            self.status_lbl.config(text=self.status_lbl.cget("text") + "  [audio playback unavailable, live diagram still runs]")

        # kick off the live animation loop, timed to the signal's own duration
        self._play_start_time = time.time()
        self._animate(signal, duration)

    def _animate(self, signal, duration):
        """Advances the playhead and scrolling oscilloscope in step with
        playback time. Runs every ~40ms (about 25 fps) via Tk's event loop."""
        elapsed = time.time() - self._play_start_time
        if elapsed >= duration:
            # snap everything to the end and stop
            self.wave_playhead.set_xdata([duration, duration])
            self.spec_playhead.set_xdata([duration, duration])
            self.canvas.draw_idle()
            self._anim_after_id = None
            self.status_lbl.config(text=f"Finished: {self.current_title}")
            return

        # move the two playhead lines
        self.wave_playhead.set_xdata([elapsed, elapsed])
        self.spec_playhead.set_xdata([elapsed, elapsed])

        # scrolling oscilloscope: a small window of samples centered on "now"
        window_sec = 0.15
        center = int(elapsed * SAMPLE_RATE)
        half_win = int(window_sec * SAMPLE_RATE / 2)
        start = max(0, center - half_win)
        end = min(len(signal), center + half_win)
        chunk = signal[start:end]
        chunk_t = (np.arange(start, end) - center) / SAMPLE_RATE
        self.live_line.set_data(chunk_t, chunk)
        self.ax_live.set_xlim(-window_sec / 2, window_sec / 2)

        self.canvas.draw_idle()
        self._anim_after_id = self.root.after(40, lambda: self._animate(signal, duration))


def main():
    root = tk.Tk()
    app = SimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
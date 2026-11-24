# Hydroacoustic Signal Denoising using GANs

Remove ship-engine noise from hydrophone recordings using a 1D convolutional
GAN (SEGAN-style generator + PatchGAN discriminator), trained in TensorFlow.

No special hardware needed — this runs on CPU (slow but works) or any GPU
Colab/VS Code can see. Everything below assumes you're working in VS Code
with a Python terminal open in this folder.

---

## 1. Why "synthetic mixing" instead of a ready-made dataset

There is no dataset of "the same underwater scene recorded with and without
a ship engine running" — that pair doesn't exist in nature. Every published
approach to this problem does the same trick:

1. Take a **clean** signal (biological / ambient underwater sound, no ships)
2. Take a **noise** signal (real recorded ship engine noise)
3. Digitally mix them at a controlled loudness ratio (SNR)
4. Train the GAN to map `mixed -> clean`, using the real clean clip as the
   ground truth label

This gives you unlimited, perfectly-paired training data built entirely from
real recordings.

## 2. Datasets (all free, no sign-up hoops beyond an email/agreement)

| Purpose | Dataset | Link | Notes |
|---|---|---|---|
| Noise source (ship engines) | **ShipsEar** | http://atlanttic.uvigo.es/underwaternoise/ | 90 recordings, 11 vessel classes, 52,734 Hz. Request/download the `.wav` set. |
| Noise source (more volume) | **DeepShip** | search "DeepShip underwater acoustic dataset GitHub" (irfankamboh/DeepShip) | ~47 hrs, 4 vessel classes, 32 kHz |
| Clean source (no ships) | **Watkins Marine Mammal Sound Database** | https://cis.whoi.edu/science/B/whalesounds/ | Whale/dolphin calls + ambient ocean sound, free download by species |
| Clean source (alt) | ShipsEar's own "natural noise" class | same ShipsEar link above | Segments recorded with no vessel present |

Download steps:
1. Get ShipsEar `.wav` files → put vessel-noise recordings in `data/noise/`
2. Get Watkins clips (or ShipsEar natural-noise clips) → put in `data/clean/`
3. Run `src/make_mixed_dataset.py` to generate paired training data automatically.

You do NOT need to manually align anything — the script randomly crops and
mixes clips at run time.

## 3. Setup in VS Code

```bash
# from the hydroacoustic_gan/ folder
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Open this folder in VS Code (`code .`), select the `venv` interpreter
(bottom-right corner or Ctrl+Shift+P -> "Python: Select Interpreter").

## 4. Project layout

```
hydroacoustic_gan/
├── data/
│   ├── clean/          <- put clean/ambient .wav files here
│   ├── noise/          <- put ship-engine .wav files here
│   └── mixed/          <- auto-generated noisy/clean pairs (npz)
├── src/
│   ├── config.py        # all hyperparameters in one place
│   ├── audio_utils.py    # loading, resampling, framing, SNR mixing
│   ├── make_mixed_dataset.py  # builds paired training data
│   ├── model.py          # Generator (U-Net 1D) + Discriminator (PatchGAN)
│   ├── train.py          # GAN training loop
│   ├── evaluate.py       # SNR metrics + spectrogram comparison plots
│   └── infer.py          # denoise a single new .wav file
├── checkpoints/          # saved model weights
├── outputs/              # denoised audio + plots land here
└── requirements.txt
```

## 5. Running it end to end

```bash
# 1. build the paired dataset (run after downloading real audio into data/clean, data/noise)
python src/make_mixed_dataset.py

# 2. train
python src/train.py

# 3. evaluate on held-out pairs (SNR improvement + spectrogram plots)
python src/evaluate.py

# 4. denoise any new hydrophone .wav file
python src/infer.py --input path/to/some_recording.wav --output outputs/denoised.wav
```

## 6. If you have zero real data right now

`src/make_mixed_dataset.py` has a `--synthetic` flag that generates
placeholder clean tones + synthetic "engine noise" (band-limited noise with
a low-frequency hum) purely with NumPy/SciPy, so you can run the entire
pipeline today and swap in real ShipsEar/Watkins audio later without
changing any code:

```bash
python src/make_mixed_dataset.py --synthetic --n_samples 500
```

## 7. Method summary (for your report/writeup)

- **Representation**: raw waveform, 1-second frames at 16 kHz (resampled
  down from ShipsEar's 52,734 Hz — keeps the model small and fast; you can
  bump `SAMPLE_RATE` in `config.py` if your machine can handle it).
- **Generator**: 1D convolutional encoder-decoder with skip connections
  (SEGAN-style), maps noisy waveform -> denoised waveform.
- **Discriminator**: 1D PatchGAN, judges (noisy, clean-or-generated) pairs
  as real/fake at the patch level rather than the whole clip.
- **Loss**: adversarial (LSGAN) loss + weighted L1 time-domain loss (keeps
  training stable and prevents the generator from producing plausible but
  unrelated audio).
- **Evaluation**: SNR improvement (dB), spectrogram before/after plots.
  PESQ/STOI are speech-specific and not meaningful for ship/marine acoustics,
  so we rely on SNR + spectrogram inspection instead.

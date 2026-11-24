"""
Generator: 1D convolutional encoder-decoder with skip connections (SEGAN-style).
    Input:  noisy waveform, shape (batch, FRAME_LEN, 1)
    Output: denoised waveform, same shape

Discriminator: 1D PatchGAN.
    Input: concatenation of (noisy, {clean or generated}) along channel dim
    Output: patch-wise real/fake logits
"""
import tensorflow as tf
import keras
from tensorflow.keras import layers

from config import GEN_FILTERS, KERNEL_SIZE, STRIDE, LATENT_NOISE_DIM, DISC_FILTERS, FRAME_LEN


@keras.saving.register_keras_serializable(package="hydro_gan")
class RandomLatent(layers.Layer):
    """Generates a (batch, time, latent_dim) standard-normal tensor matching
    the time dimension of its input. Used to inject stochasticity at the
    generator bottleneck. A proper Layer (not a Lambda) so the model can be
    saved/loaded safely."""

    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def call(self, x):
        time_steps = tf.shape(x)[1]
        batch = tf.shape(x)[0]
        return tf.random.normal((batch, time_steps, self.latent_dim))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.latent_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config


def build_generator():
    inputs = layers.Input(shape=(FRAME_LEN, 1), name="noisy_waveform")

    skips = []
    x = inputs
    for filters in GEN_FILTERS:
        x = layers.Conv1D(filters, KERNEL_SIZE, strides=STRIDE, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        skips.append(x)

    # bottleneck: concat small stochastic latent so the generator isn't purely
    # deterministic (keeps it a proper GAN rather than a plain autoencoder)
    z = RandomLatent(LATENT_NOISE_DIM)(x)
    x = layers.Concatenate(axis=-1)([x, z])

    # decoder, mirrored, with skip connections
    for filters, skip in zip(reversed(GEN_FILTERS[:-1]), reversed(skips[:-1])):
        x = layers.Conv1DTranspose(filters, KERNEL_SIZE, strides=STRIDE, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        # crop/pad to match skip length before concatenating
        x = _match_length(x, skip)
        x = layers.Concatenate(axis=-1)([x, skip])

    x = layers.Conv1DTranspose(1, KERNEL_SIZE, strides=STRIDE, padding="same")(x)
    x = _match_length(x, inputs)
    outputs = layers.Activation("tanh", name="denoised_waveform")(x)

    return tf.keras.Model(inputs, outputs, name="Generator")


def _match_length(x, ref):
    """Pad or crop the time dimension of x to match ref's time dimension."""
    diff = ref.shape[1] - x.shape[1]
    if diff > 0:
        x = layers.ZeroPadding1D((0, diff))(x)
    elif diff < 0:
        x = layers.Cropping1D((0, -diff))(x)
    return x


def build_discriminator():
    noisy_in = layers.Input(shape=(FRAME_LEN, 1), name="noisy_waveform")
    cand_in = layers.Input(shape=(FRAME_LEN, 1), name="clean_or_generated")
    x = layers.Concatenate(axis=-1)([noisy_in, cand_in])

    for filters in DISC_FILTERS:
        x = layers.Conv1D(filters, KERNEL_SIZE, strides=STRIDE, padding="same")(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    patch_logits = layers.Conv1D(1, 3, strides=1, padding="same", name="patch_logits")(x)
    return tf.keras.Model([noisy_in, cand_in], patch_logits, name="Discriminator")


if __name__ == "__main__":
    g = build_generator()
    d = build_discriminator()
    g.summary()
    d.summary()

"""
Trains the denoising GAN (LSGAN adversarial loss + L1 reconstruction loss).

Usage:
    python src/train.py
"""
import os
import time

import numpy as np
import tensorflow as tf

from config import (
    MIXED_DIR, CHECKPOINT_DIR, BATCH_SIZE, EPOCHS, LR_G, LR_D, BETA_1,
    L1_LAMBDA, CHECKPOINT_EVERY, FRAME_LEN,
)
from model import build_generator, build_discriminator


def load_dataset(split):
    path = os.path.join(MIXED_DIR, f"{split}.npz")
    data = np.load(path)
    mixed = data["mixed"][..., None]   # (N, FRAME_LEN, 1)
    clean = data["clean"][..., None]
    ds = tf.data.Dataset.from_tensor_slices((mixed, clean))
    ds = ds.shuffle(buffer_size=len(mixed), seed=42)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds, len(mixed)


mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()


@tf.function
def train_step(generator, discriminator, gen_opt, disc_opt, noisy, clean):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(noisy, training=True)

        real_logits = discriminator([noisy, clean], training=True)
        fake_logits = discriminator([noisy, generated], training=True)

        # LSGAN losses: real -> target 1, fake -> target 0
        disc_loss = 0.5 * (mse(tf.ones_like(real_logits), real_logits) +
                             mse(tf.zeros_like(fake_logits), fake_logits))

        gen_adv_loss = mse(tf.ones_like(fake_logits), fake_logits)
        gen_l1_loss = mae(clean, generated)
        gen_loss = gen_adv_loss + L1_LAMBDA * gen_l1_loss

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return gen_loss, disc_loss, gen_adv_loss, gen_l1_loss


def main():
    train_ds, n_train = load_dataset("train")
    print(f"Training on {n_train} paired samples.")

    generator = build_generator()
    discriminator = build_discriminator()

    gen_opt = tf.keras.optimizers.Adam(LR_G, beta_1=BETA_1)
    disc_opt = tf.keras.optimizers.Adam(LR_D, beta_1=BETA_1)

    ckpt = tf.train.Checkpoint(generator=generator, discriminator=discriminator,
                                 gen_opt=gen_opt, disc_opt=disc_opt)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f"Restored from {ckpt_manager.latest_checkpoint}")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        g_losses, d_losses = [], []
        for noisy, clean in train_ds:
            g_loss, d_loss, g_adv, g_l1 = train_step(
                generator, discriminator, gen_opt, disc_opt, noisy, clean
            )
            g_losses.append(float(g_loss))
            d_losses.append(float(d_loss))

        print(f"Epoch {epoch}/{EPOCHS} - gen_loss: {np.mean(g_losses):.4f} "
              f"disc_loss: {np.mean(d_losses):.4f} "
              f"({time.time() - start:.1f}s)")

        if epoch % CHECKPOINT_EVERY == 0 or epoch == EPOCHS:
            save_path = ckpt_manager.save()
            print(f"  Saved checkpoint: {save_path}")

    # also save the generator in the native Keras format for easy inference
    generator.save(os.path.join(CHECKPOINT_DIR, "generator_final.keras"))
    print("Training complete.")


if __name__ == "__main__":
    main()

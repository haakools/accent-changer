import tensorflow as tf
import os
import sys
from pathlib import Path
from tensorflow_examples.models.pix2pix import pix2pix

repoDir = os.path.abspath(Path(__file__).parent.parent.parent)
sys.path.append(repoDir)
dataDir = os.path.join(repoDir, 'preprocessed_data', 'mel_spectrums', 'female')

from utils.file_utilities import load_npz_from_path, save_wave_file_to_path
from preprocessing.spectogram import convert_mel_spectrum_to_wav
from generator import generator_f, generator_g
from discriminator import discriminator_x, discriminator_y
from losses import generator_loss, discriminator_loss, calc_cycle_loss, identity_loss

# Load the data
bangla = load_npz_from_path(os.path.join(dataDir, 'bangla.npz'))
american = load_npz_from_path(os.path.join(dataDir, 'american.npz'))

import numpy as np
bangla = np.expand_dims(bangla, axis=-1)
american = np.expand_dims(american, axis=-1)


# splitting the data
print(f"Bangla data shape: {bangla.shape}")
print(f"American data shape: {american.shape}")

exit()
N_samples = min(bangla.shape[0], american.shape[0])

train_bangla = bangla[:int(N_samples*0.8)]
train_american = american[:int(N_samples*0.8)]
test_bangla = bangla[int(N_samples*0.8):]
test_american = american[int(N_samples*0.8):]


generator_g_optimizer= tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer= tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Checkpoint
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer= generator_g_optimizer,
                           generator_f_optimizer= generator_f_optimizer,
                           discriminator_x_optimizer =  discriminator_x_optimizer,
                           discriminator_y_optimizer =  discriminator_y_optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 10
BATCH_SIZE = 1

# load the data
print(f"train_bangla.shape: {train_bangla.shape}")
print(f"train_american.shape: {train_american.shape}")

train_bangla_dataset = tf.data.Dataset.from_tensor_slices(train_bangla).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_america_dataset = tf.data.Dataset.from_tensor_slices(train_american).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_gradients.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

def generate_voice(model, test_input, epoch):
  prediction = model(test_input)

  wav_file = convert_mel_spectrum_to_wav(prediction[0])

  save_wave_file_to_path(wav_file, f'gen_f_prediciton_{epoch}.wav')


if __name__ == "__main__":
  import time
  EPOCHS = 40
  for epoch in range(EPOCHS):
    start = time.time()
    generate_voice(generator_f, test_american, "gen_f_before_training")
    generate_voice(generator_f, test_bangla, "gen_f_before_training")
    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_bangla_dataset, train_america_dataset)):
      train_step(image_x, image_y)
      if n % 10 == 0:
        print ('.', end='')
      n += 1

    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    generate_voice(generator_f, test_american, "gen_f_before_training")
    generate_voice(generator_f, test_bangla, "gen_f_before_training")

    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                          ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))

from keras.utils.image_dataset import image_dataset_from_directory
from keras.layers import Dense, Flatten, Reshape, Input, BatchNormalization, MaxPooling2D,\
                         Conv2D, Conv2DTranspose, LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
import tensorflow as tf
import keras

import time
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Params
hidden_dim = 60
batch_size = 35
EPOCHS = 3

# Dataset (https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set)
dataset = image_dataset_from_directory('archive/thumbnails128x128',
                                       labels=None, batch_size=batch_size,
                                       image_size=(64, 64), interpolation='bicubic')
dataset = dataset.map(lambda img: img / 255)

def show_ds_faces(count=3):
    for batch in dataset:
        print(batch.shape, np.max(batch), np.min(batch))
        for i in range(count):
            plt.imshow(batch[i])
            plt.show()
        break

def progress_bar(n_iter, n_total, prefix='Progress: ', suffix='', length=55, fill='█', lost='-'):
    percent = f"{100 * (n_iter / float(n_total)) :.1f}"
    filled_length = round(length * n_iter // n_total)
    bar = fill * filled_length + lost * (length - filled_length)
    print(f'\r{prefix}[{n_iter}/{n_total}] |{bar}| {percent}% {suffix}', end='')
    if n_iter == n_total:
        print()

# Models
generator = Sequential([
    Input(shape=(hidden_dim,)),
    Dense(8 * 8 * 196, activation='relu'),
    BatchNormalization(),
    Reshape((8, 8, 196)),  # 8x8
    Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),  # 16x16
    BatchNormalization(momentum=0.8),
    Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),  # 32x32
    BatchNormalization(momentum=0.8),
    Conv2DTranspose(196, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),  # 64x64
    BatchNormalization(momentum=0.8),
    Conv2DTranspose(3, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='sigmoid')])  # 64x64

discriminator = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32 + 16, kernel_size=(4, 4), strides=(1, 1), padding='same', activation=LeakyReLU(0.2)),
    Conv2D(32 + 16, kernel_size=(4, 4), strides=(1, 1), padding='same', activation=LeakyReLU(0.2)),
    MaxPooling2D(),  # 32x32
    Conv2D(64 + 32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=LeakyReLU(0.2)),
    Conv2D(64 + 32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=LeakyReLU(0.2)),
    MaxPooling2D(),  # 16x16
    Conv2D(128 + 64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=LeakyReLU(0.2)),
    Conv2D(128 + 64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=LeakyReLU(0.2)),
    MaxPooling2D(),  # 8x8
    Conv2D(256 + 128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=LeakyReLU(0.2)),
    MaxPooling2D(),  # 4x4
    Conv2D(512 + 128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=LeakyReLU(0.2)),
    MaxPooling2D(),  # 2x2
    Flatten(),
    Dense(1, activation='sigmoid')])  # linear

generator.summary()
discriminator.summary()

# Losses and optimizers
generator_optimizer = RMSprop(0.00005)  # rho=0.7
discriminator_optimizer = RMSprop(0.00005)  # rho=0.97, momentum=0.35
binary_crossentropy = keras.losses.BinaryCrossentropy(from_logits=False)  # True

def generator_loss(fake_output):
    return binary_crossentropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Train functions
@tf.function
def train_step(train_images):
    noises = tf.random.normal([batch_size, hidden_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noises, training=True)

        real_output = discriminator(train_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # return gen_loss, disc_loss

def train(epochs: int):
    all_steps = 70_000 // batch_size  # 70000 = len dataset

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}/{epochs}')
        start_time = time.time()

        if epoch == 0:
            generator_optimizer.learning_rate.assign(0.00012)
            discriminator_optimizer.learning_rate.assign(0.00012)
        elif epoch < 10:
            generator_optimizer.learning_rate.assign(0.00005)
            discriminator_optimizer.learning_rate.assign(0.00005)
        elif epoch < 20:
            generator_optimizer.learning_rate.assign(0.00002)
            discriminator_optimizer.learning_rate.assign(0.00002)
        else:
            generator_optimizer.learning_rate.assign(0.00001)
            discriminator_optimizer.learning_rate.assign(0.00001)

        step = 1
        for image_batch in dataset:
            train_step(image_batch)
            progress_bar(step, all_steps)
            # gen_loss, disc_loss = train_step(image_batch) suffix=f'\tgen: {gen_loss:.2f}\tdis: {disc_loss:.2f}'
            if step == all_steps: break
            step += 1

        # generator.save('models/face/GAN_face_gen_60.h5')
        # discriminator.save('models/face/GAN_face_disk_60.h5')
        generator.save('name_gen.h5')
        discriminator.save('name_disc.h5')
        progress_bar(all_steps, all_steps, prefix='', length=30,
                     suffix=f'\ttime: {time.time() - start_time:.1f} sec')  # \t loss: {train_history[-1]:.3f}

# Training
# generator = load_model('models/face/GAN_face_gen_60_pro.h5')
# discriminator = load_model('models/face/GAN_face_disk_60_pro.h5')
train(EPOCHS)

# Testing
side = 4
for i in range(2):
    images = generator.predict(tf.random.normal([side*side, hidden_dim]))

    num = 0
    plt.figure(figsize=(side*2, side*2))
    for i in range(side):
        for j in range(side):
            plt.subplot(side, side, num+1)
            plt.imshow(images[num].squeeze())
            plt.axis('off')
            num += 1

    plt.show()


# n = 100
# images = generator.predict(tf.random.normal([n, hidden_dim]))
# for i in range(n):
#     plt.imshow(images[i])
#     plt.axis('off')
#     plt.show()
#     answer = input("Save as: ")
#     if answer:
#         plt.imsave(f'f_{answer}.png', images[i], format='png')

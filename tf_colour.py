import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# Сглаживание меток


def smooth_positive(y):
    return y - 0.2 + (tf.random.uniform(y.shape) * 0.2)


def smooth_negative(y):
    return y + (tf.random.uniform(y.shape) * 0.3)


train_set, test_set = keras.datasets.cifar10.load_data()

dog_class = 5

train_images, train_labels = train_set
train_images = train_images[train_labels.flatten() == dog_class]
train_labels = np.full(train_images.shape, dog_class)

BATCH_SIZE = 32
# Нормализация изображений в диапазон [-1, 1]
train_images = (train_images) / 127.5 - 1.0

dataset = tf.data.Dataset.from_tensor_slices(train_images)

dataset = dataset.shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
# Загружаем один батч данных асинхронно, чтобы ускорить обучение
dataset = dataset.prefetch(1)


noise_input = 100


class Generator(keras.Model):
    def __init__(self, noise_input):
        super().__init__()
        self.model = keras.models.Sequential([
            keras.layers.Dense(
                4 * 4 * 128, input_shape=[noise_input],
                activation=keras.layers.LeakyReLU(alpha=0.2)
            ),

            keras.layers.Reshape([4, 4, 128]),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2DTranspose(
                128, kernel_size=4, strides=2, padding="SAME",
                activation=keras.layers.LeakyReLU(alpha=0.2)
            ),
            keras.layers.BatchNormalization(),

            # Second UpSample doubling the size to 16x16
            keras.layers.Conv2DTranspose(
                128, kernel_size=4, strides=2, padding="SAME",
                activation=keras.layers.LeakyReLU(alpha=0.2)
            ),
            keras.layers.BatchNormalization(),

            # Last UpSample doubling the size to 32x32
            keras.layers.Conv2DTranspose(
                3, kernel_size=4, strides=2, padding="SAME",
                activation='tanh'
            )
        ])

    def call(self, inputs):
        return self.model(inputs)


generator = Generator(noise_input=noise_input)

class Discriminator(keras.Model):
    def __init__(self):
        super().__init__()
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME",
                                activation=keras.layers.LeakyReLU(alpha=0.2),
                                input_shape=[32, 32, 3]),
            keras.layers.Dropout(0.4),

            keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="SAME",
                                activation=keras.layers.LeakyReLU(alpha=0.2)),
            keras.layers.Dropout(0.4),

            keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="SAME",
                                activation=keras.layers.LeakyReLU(alpha=0.2)),
            keras.layers.Dropout(0.4),

            keras.layers.Flatten(),
            keras.layers.Dense(1, activation="sigmoid")
        ])

    def call(self, inputs):
        return self.model(inputs)


discriminator = Discriminator()

lr = 0.0001
num_epochs = 50


optimizer_disc = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
optimizer_gen = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer_disc, metrics=['accuracy'])
discriminator.trainable = False


for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    for real_samples in dataset:

        real_samples = tf.cast(real_samples, tf.float32)
        batch_size = real_samples.shape[0]
        #Обучение дискриминатора 
        noise = tf.random.normal(shape=[batch_size, noise_input])
        fake_images = generator(noise)
        fake_images = tf.cast(fake_images, tf.float32)
        mixed_images = tf.concat([fake_images, real_samples], axis=0)

        discriminator_zeros = smooth_negative(np.zeros((batch_size, 1)))
        discriminator_ones = smooth_positive(np.ones((batch_size, 1)))
        discriminator_labels = tf.convert_to_tensor(
            np.concatenate((discriminator_zeros, discriminator_ones)) , dtype=tf.float32)

        with tf.GradientTape() as tape:
            discriminator.trainable = True
            discriminator_output = discriminator(mixed_images)
            discriminator_loss = tf.keras.losses.binary_crossentropy(
            discriminator_labels, discriminator_output)

        grads = tape.gradient(discriminator_loss,
                          discriminator.trainable_variables)
        optimizer_disc.apply_gradients(
            zip(grads, discriminator.trainable_variables))

        # Обучение генератора
        noise = tf.random.normal(shape=[batch_size, noise_input])
        with tf.GradientTape() as tape:
            generator.trainable = True
            fake_images = generator(noise)
            discriminator_output_generated = discriminator(fake_images)
            generator_loss = tf.keras.losses.binary_crossentropy(
                np.ones((batch_size, 1)), discriminator_output_generated)

        grads = tape.gradient(generator_loss, generator.trainable_variables)
        optimizer_gen.apply_gradients(zip(grads, generator.trainable_variables))

    # Печать потерь и генерация изображений
    
    print(f"Epoch: {epoch+1} Loss D.: {discriminator_loss.numpy().mean()}")
    print(f"Epoch: {epoch+1} Loss G.: {generator_loss.numpy().mean()}")

    # Показ изображений
    plt.clf()
    plt.title(f"After {epoch+1} epoch(s)")
    fake_images = (fake_images + 1) / 2  # Де-нормализация
    for j in range(32):
        ax = plt.subplot(8, 4, j + 1)
        plt.imshow(fake_images[j])
        plt.xticks([])
        plt.yticks([])
    plt.pause(0.001)

generator.save('generator_model.h5')
discriminator.save('discriminator_model.h5')

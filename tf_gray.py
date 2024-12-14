import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import summary

def smooth_positive(y):
    return y - 0.2 + (tf.random.uniform(y.shape) * 0.2)


def smooth_negative(y):
    return y + (tf.random.uniform(y.shape) * 0.3)


# Загрузка MNIST
(train_images, _), (test_images, _) = keras.datasets.mnist.load_data()

# Нормализация изображений
train_images = train_images / 127.5 - 1.0  # Изображения в диапазоне [-1, 1]
# Добавляем размерность канала
train_images = np.expand_dims(train_images, axis=-1)
dataset = tf.data.Dataset.from_tensor_slices(train_images)

BATCH_SIZE = 32
dataset = dataset.shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# Дискриминатор


class Discriminator(keras.Model):
    def __init__(self):
        super().__init__()
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28, 1)),
            keras.layers.Dense(
                1024, activation=keras.layers.LeakyReLU(alpha=0.2)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(
                512, activation=keras.layers.LeakyReLU(alpha=0.2)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(
                256, activation=keras.layers.LeakyReLU(alpha=0.2)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        return self.model(inputs)


discriminator = Discriminator()
print('Discriminator summary start')
discriminator.summary()
print('Discriminator summary')
# Генератор
noise_input = 100
class Generator(keras.Model):
    def __init__(self, noise_input):
        super().__init__()
        self.model = keras.Sequential([
            keras.layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=0.2),
                               input_shape=(noise_input,)),
            keras.layers.Dense(
                512, activation=keras.layers.LeakyReLU(alpha=0.2)),
            keras.layers.Dense(
                1024, activation=keras.layers.LeakyReLU(alpha=0.2)),
            keras.layers.Dense(784, activation='tanh'),
            keras.layers.Reshape((28, 28, 1))
        ])

    def call(self, inputs):
        return self.model(inputs)


generator = Generator(noise_input=noise_input)
print('generator summary start')
generator.summary()
print('generator summary')

# Параметры
num_epochs = 50
lr = 0.0001
# Оптимизаторы
optimizer_disc = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
optimizer_gen = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer_disc, metrics=['accuracy'])
discriminator.trainable = False

# Обучение
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    for real_samples in dataset:
        real_samples = tf.cast(real_samples, tf.float32)
        batch_size = real_samples.shape[0]

        # Генерация шума
        noise = tf.random.normal(shape=[batch_size, noise_input])
        fake_images = generator(noise)

        # Сглаживание меток для дискриминатора
        real_labels = smooth_positive(np.ones((batch_size, 1)))
        fake_labels = smooth_negative(np.zeros((batch_size, 1)))

        # Обучение дискриминатора
        with tf.GradientTape() as tape:
            discriminator.trainable = True
            real_output = discriminator(real_samples)
            fake_output = discriminator(fake_images)
            disc_loss = tf.keras.losses.binary_crossentropy(
                real_labels, real_output) + tf.keras.losses.binary_crossentropy(fake_labels, fake_output)

        grads = tape.gradient(disc_loss, discriminator.trainable_variables)
        optimizer_disc.apply_gradients(
            zip(grads, discriminator.trainable_variables))

        # Обучение генератора
        noise = tf.random.normal(shape=[batch_size, noise_input])
        with tf.GradientTape() as tape:
            generator.trainable = True
            fake_images = generator(noise)
            fake_output = discriminator(fake_images)
            gen_loss = tf.keras.losses.binary_crossentropy(
                real_labels, fake_output)

        grads = tape.gradient(gen_loss, generator.trainable_variables)
        optimizer_gen.apply_gradients(
            zip(grads, generator.trainable_variables))

        # Печать и визуализация результатов
    print(f"Epoch: {epoch+1} Loss D.: {disc_loss.numpy().mean()}")
    print(f"Epoch: {epoch+1} Loss G.: {gen_loss.numpy().mean()}")

    plt.clf()
    plt.title(f"After {epoch+1} epoch(s)")
    fake_images = (fake_images + 1) / 2  # Де-нормализация изображений
    for j in range(32):
        ax = plt.subplot(8, 4, j + 1)
        plt.imshow(fake_images[j, :, :, 0], cmap="gray_r")
        plt.xticks([])
        plt.yticks([])

    plt.pause(0.001)

# Сохранение моделей
generator.save('generator_mnist.h5')
discriminator.save('discriminator_mnist.h5')

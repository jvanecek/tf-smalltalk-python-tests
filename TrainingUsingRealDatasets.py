import unittest
import tensorflow as tf
import numpy as np
from tensorflow import keras

tf.compat.v1.disable_eager_execution()

class TrainingUsingRealDatasetsTest(unittest.TestCase):
    def test_initializedToZeros(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.zeros),
            keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.zeros)
        ])
        model.compile(
            optimizer=tf.optimizers.Adagrad(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

        history = model.fit(x=train_images, y=train_labels, epochs=5, batch_size=60000, shuffle=False)

        self.assertEqual(history.history['loss'], [2.302597999572754,
                                                   2.302597999572754,
                                                   2.302597999572754,
                                                   2.302597999572754,
                                                   2.302597999572754])
        self.assertEqual(history.history['accuracy'], [0.10000000149011612,
                                                       0.10000000149011612,
                                                       0.10000000149011612,
                                                       0.10000000149011612,
                                                       0.10000000149011612])

    def test_initializedWithGlorotUniform(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        train_images = tf.reshape(tf.convert_to_tensor(train_images), shape=(60000, 28 * 28))
        train_labels = tf.convert_to_tensor(train_labels)

        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=28 * 28),
            keras.layers.Dense(128, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1)),
            keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1))
        ])
        model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

        history = model.fit(x=train_images, y=train_labels, epochs=5, batch_size=60000, steps_per_epoch=1,
                            shuffle=False)

        self.assertEqual(history.history['loss'], [8.887983322143555,
                                                   4.939030170440674,
                                                   3.7245869636535645,
                                                   3.5525035858154297,
                                                   3.52937388420105])
        np.testing.assert_array_almost_equal(
            history.history['accuracy'], [0.10226667, 0.20715, 0.2781, 0.29085, 0.28815])

    def test_initializedWithGlorotUniform1(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_imagefs, test_labels) = fashion_mnist.load_data()

        train_images = tf.data.Dataset(tf.reshape(tf.convert_to_tensor(train_images), shape=(60000, 28 * 28)))
        train_labels = tf.convert_to_tensor(train_labels)

        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=28 * 28),
            keras.layers.Dense(128, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1)),
            keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1))
        ])
        model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

        history = model.fit(x=train_images, y=train_labels, epochs=5, batch_size=60000, steps_per_epoch=1,
                            shuffle=False)

        self.assertEqual(history.history['loss'], [8.887983322143555,
                                                   4.939030170440674,
                                                   3.7245869636535645,
                                                   3.5525035858154297,
                                                   3.52937388420105])
        np.testing.assert_array_almost_equal(
            history.history['accuracy'], [0.10226667, 0.20715, 0.2781, 0.29085, 0.28815])


if __name__ == '__main__':
    unittest.main()

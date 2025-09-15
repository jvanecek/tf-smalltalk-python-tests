import unittest
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

class Convolution2D(unittest.TestCase):
    def testExample1(self):
        x = tf.constant([[
            [[11], [12], [13], [14]],
            [[21], [22], [23], [24]],
            [[31], [32], [33], [34]],
            [[41], [42], [43], [44]]
        ]], dtype=tf.float32)

        self.assertEqual(x.shape, (1, 4, 4, 1))

        filter = tf.constant([[
            [[0.11, 0.12], [0.21, 0.22]],
            [[1.11, 1.12], [1.21, 1.22]]]],
            dtype=tf.float32)
        self.assertEqual(filter.shape, (1, 2, 2, 2))

        y = Conv2D(
            filters=2,
            kernel_size=(2, 2),
            activation='relu',
            input_shape=x.shape[1:],
            kernel_initializer=tf.keras.initializers.Constant(value=filter),
            use_bias=False)(x)

        self.assertEqual(y.shape, (1, 3, 3, 2))

        np.testing.assert_allclose(y.numpy(), [[
            [[53.660004, 54.32], [56.300003, 57.], [58.940002, 59.68]],
            [[80.06, 81.12], [82.700005, 83.8], [85.34, 86.48]],
            [[106.46001, 107.92], [109.100006, 110.600006], [111.740005, 113.28]]
        ]])

    def testConv2DAsLayer(self):
        x = tf.constant([[
            [[11], [12], [13], [14]],
            [[21], [22], [23], [24]],
            [[31], [32], [33], [34]],
            [[41], [42], [43], [44]]
        ]], dtype=tf.float32)

        self.assertEqual(x.shape, (1, 4, 4, 1))

        filter = filter = tf.constant([
            [
                [[0.11, 0.12]],
                [[0.21, 0.22]]
            ],
            [
                [[1.11, 1.12]],
                [[1.21, 1.22]]
            ]
        ], dtype=tf.float32)
        self.assertEqual(filter.shape, (2, 2, 1, 2))

        model_cnn = Sequential([
            Conv2D(
                filters=2,
                kernel_size=(2, 2),
                input_shape=x.shape[1:],
                kernel_initializer=tf.keras.initializers.Constant(value=filter),
                use_bias=False)
        ])

        y = model_cnn(x)

        self.assertEqual(y.shape, (1, 3, 3, 2))

        np.testing.assert_allclose(y.numpy(), [[
            [[53.660004, 54.32], [56.300003, 57.], [58.940002, 59.68]],
            [[80.06, 81.12], [82.700005, 83.8], [85.34, 86.48]],
            [[106.46001, 107.92], [109.100006, 110.600006], [111.740005, 113.28]]
        ]])
    
    
    def testConv2DAsLayerActivatedByReLU(self):
        x = tf.constant([[
            [[11], [12], [13], [14]],
            [[21], [22], [23], [24]],
            [[31], [32], [33], [34]],
            [[41], [42], [43], [44]]
        ]], dtype=tf.float32)

        self.assertEqual(x.shape, (1, 4, 4, 1))

        filter = filter = tf.constant([
            [
                [[0.11, 0.12]],
                [[0.21, 0.22]]
            ],
            [
                [[1.11, 1.12]],
                [[1.21, 1.22]]
            ]
        ], dtype=tf.float32)
        self.assertEqual(filter.shape, (2, 2, 1, 2))

        model_cnn = Sequential([
            Conv2D(
                filters=2,
                kernel_size=(2, 2),
                activation='relu',
                input_shape=x.shape[1:],
                kernel_initializer=tf.keras.initializers.Constant(value=filter),
                use_bias=False)
        ])

        y = model_cnn(x)

        self.assertEqual(y.shape, (1, 3, 3, 2))

        np.testing.assert_allclose(y.numpy(), [[
            [[53.660004, 54.32], [56.300003, 57.], [58.940002, 59.68]],
            [[80.06, 81.12], [82.700005, 83.8], [85.34, 86.48]],
            [[106.46001, 107.92], [109.100006, 110.600006], [111.740005, 113.28]]
        ]])
    
    def testConv2DAndMaxPooling2DLayer(self):
        x = tf.constant([[
            [[11], [12], [13], [14]],
            [[21], [22], [23], [24]],
            [[31], [32], [33], [34]],
            [[41], [42], [43], [44]]
        ]], dtype=tf.float32)

        self.assertEqual(x.shape, (1, 4, 4, 1))

        filter = tf.constant([[
            [[0.11, 0.12], [0.21, 0.22]],
            [[1.11, 1.12], [1.21, 1.22]]]],
            dtype=tf.float32)
        self.assertEqual(filter.shape, (1, 2, 2, 2))

        model_cnn = Sequential([
            Conv2D(
                filters=2,
                kernel_size=(2, 2),
                activation='relu',
                input_shape=x.shape[1:],
                kernel_initializer=tf.keras.initializers.Constant(value=filter),
                use_bias=False),
            MaxPooling2D()
        ])

        y = model_cnn(x)

        self.assertEqual(y.shape, (1, 1, 1, 2))

        np.testing.assert_allclose(y.numpy(), [[[[82.700005, 83.8]]]])

if __name__ == '__main__':
    unittest.main()

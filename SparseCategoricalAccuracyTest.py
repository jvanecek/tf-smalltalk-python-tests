import unittest

import tensorflow as tf
from tensorflow import keras

tf.compat.v1.enable_eager_execution()

class SparseCategoricalAccuracyTest(unittest.TestCase):

    def assertExpectedAccuracyValue(self, real, prediction, expectedValue):
        metric = keras.metrics.SparseCategoricalAccuracy()
        metric.update_state(real, prediction)

        self.assertEqual(metric.result().numpy(), expectedValue)

    def testAccuracyBetweenTwo32BitIntegerTensor(self):
        prediction = [[0.7, 0.2, 0.1], [0.8, 0.98, 0.9], [0.21, 0.2, 0.1], [0.49, 0.5, 0.23]]
        real = [[0], [1], [1], [1]]

        self.assertExpectedAccuracyValue(real, prediction, 0.75)

    def testAccuracyBetweenTwoFloatTensors(self):
        prediction = [[0.1, 0.6, 0.3], [0.05, 0.95, 0]]
        real = [[2], [1]]

        self.assertExpectedAccuracyValue(real, prediction, 0.5)

    def testAccuracyWithFlattenTarget(self):
        prediction = [[0.1, 0.6, 0.3], [0.05, 0.95, 0]]
        real = [2, 1]

        self.assertExpectedAccuracyValue(real, prediction, 0.5)

if __name__ == '__main__':
    unittest.main()

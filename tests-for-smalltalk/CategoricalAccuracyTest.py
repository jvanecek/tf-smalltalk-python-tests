import unittest

import tensorflow as tf
from tensorflow import keras

tf.compat.v1.enable_eager_execution()

class CategoricalAccuracyTest(unittest.TestCase):

    def assertExpectedAccuracyValue(self, real, prediction, expectedValue):
        metric = keras.metrics.CategoricalAccuracy()
        metric.update_state(real, prediction)

        self.assertEqual(metric.result().numpy(), expectedValue)

    def testAccuracyBetweenTwo32BitIntegerTensor(self):
        prediction = [1, 2, 3, 4]
        real = [0, 2, 3, 4]

        self.assertExpectedAccuracyValue(real, prediction, 1)

    #def testAccuracyBetweenTwoDifferentBitIntegerTensor(self):

    def testAccuracyBetweenTwoFloatTensors(self):
        prediction = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
        real = [[0, 0, 1], [0, 1, 0]]

        self.assertExpectedAccuracyValue(real, prediction, 0.5)

if __name__ == '__main__':
    unittest.main()

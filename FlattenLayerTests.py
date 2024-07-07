import unittest
import tensorflow as tf
import numpy as np
from tensorflow import keras

tf.compat.v1.disable_eager_execution()


class FlattenLayerTest(unittest.TestCase):

  def _passThroughFlattenLayer(self, input):
    # Create an input placeholder
    input_placeholder = tf.compat.v1.placeholder(shape=input.shape, dtype=tf.float32)

    flatten_layer = tf.keras.layers.Flatten()(input_placeholder)

    # Create a session and initialize variables
    with tf.compat.v1.Session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())

      # Predict on the input array
      output = sess.run(flatten_layer, feed_dict={input_placeholder: input})
    print(output)
    return output

  def testFlattenMatrix(self):
    input = np.array([
      [[1.1], [1.2]],
      [[2.1], [2.2]]
    ])
    expectedOutput = [[1.1, 1.2], [2.1, 2.2]]

    output = self._passThroughFlattenLayer(input)

    self.assertTupleEqual(output.shape, (2, 2))
    self._assertElementsAreAllClose(output.tolist(), expectedOutput)

  def testFlatten3DimensionTensor(self):
    input = np.array([
      [[1.11, 1.12], [1.21, 1.22]],
      [[2.11, 2.12], [2.21, 2.22]],
      [[3.11, 3.12], [3.21, 3.22]]
    ])
    expectedOutput = [
      [1.11, 1.12, 1.21, 1.22],
      [2.11, 2.12, 2.21, 2.22],
      [3.11, 3.12, 3.21, 3.22]
    ]

    output = self._passThroughFlattenLayer(input)

    self.assertTupleEqual(output.shape, (3, 4))
    self._assertElementsAreAllClose(output.tolist(), expectedOutput)

  def testFlatten3DimensionTensorCase1(self):
    input = np.arange(6).reshape((3, 2, 1))
    expectedOutput = np.arange(6).reshape((3, 2))

    output = self._passThroughFlattenLayer(input)

    self.assertTupleEqual(output.shape, (3, 2))
    self._assertElementsAreAllClose(output.tolist(), expectedOutput)

  def testFlatten3DimensionTensorCase2(self):
    input = np.arange(24).reshape((3, 4, 2))
    expectedOutput = np.arange(24).reshape((3, 8))

    output = self._passThroughFlattenLayer(input)

    self.assertTupleEqual(output.shape, (3, 8))
    self._assertElementsAreAllClose(output.tolist(), expectedOutput)

  def testFlatten4DimensionTensor(self):
    input = np.arange(72).reshape((3, 4, 2, 3))
    expectedOutput = np.arange(72).reshape((3, 24))

    output = self._passThroughFlattenLayer(input)

    self.assertTupleEqual(output.shape, (3, 24))
    self._assertElementsAreAllClose(output.tolist(), expectedOutput)

  def testFlatten1DimensionTensor(self):
    input = np.array([1, 2, 3, 4])
    expectedOutput = np.array([[1], [2], [3], [4]])

    output = self._passThroughFlattenLayer(input)

    self.assertTupleEqual(output.shape, (4, 1))
    self._assertElementsAreAllClose(output.tolist(), expectedOutput)


  def _assertElementsAreAllClose(self, anArray, anExpectedArray):
    np.testing.assert_allclose(
      anArray, anExpectedArray,
      rtol=1e-5, atol=0)

if __name__ == '__main__':
  unittest.main()

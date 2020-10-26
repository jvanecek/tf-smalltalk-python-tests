import unittest
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.utils import losses_utils

tf.compat.v1.enable_eager_execution()

class TrainingUsingOptimizationTest(object):
    inputTensor = np.array( [
        [ 0, 0, 1 ],
        [ 0, 1, 1],
        [ 1, 0, 0],
        [ 1, 1, 1 ]])
    expectedProbabilityByLabel = np.array([ [0., 1.], [1., 0.], [0., 1.], [1., 1.] ])
    expectedLabels = np.array([ 0, 1, 0, 0 ])

    def _modelWithTwoOutputUnits(self):
        weightInit = tf.keras.initializers.Constant(value=((0, 0),(0, 0),(0, 0)))
        biasInit = tf.keras.initializers.Constant(value=(0.2, 0.8))

        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(3,)))
        model.add(tf.keras.layers.Dense(2, kernel_initializer=weightInit, bias_initializer=biasInit))

        return model

    def _assertElementsAreAllClose(self, anArray, anExpectedArray):
        np.testing.assert_allclose(
            anArray, anExpectedArray,
            rtol=1e-5, atol=0)

    def _assertHasTheSameElementsThat(self, aTensorFlowArray, anExpectedArray ):
        self._assertElementsAreAllClose( aTensorFlowArray.numpy(), anExpectedArray )

    def _subclassResponsibility(self):
        raise NotImplementedError('Subclass responsibility')

    def optimizationAlgorithm(self):
        self._subclassResponsibility()

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        self._subclassResponsibility()

    def expectedLossWhenMinimizingMeanSquaredError(self):
        self._subclassResponsibility()

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        self._subclassResponsibility()

    def categoricalCrossEntropy(self):
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def mse(self):
        return tf.keras.losses.MeanSquaredError()

    def sparseCategoricalCrossEntropy(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def testMinimizingCategoricalCrossEntropy(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.categoricalCrossEntropy() )
        summary = model.fit(self.inputTensor, self.expectedProbabilityByLabel, epochs=5)

        self._assertElementsAreAllClose(
            summary.history['loss'],
            self.expectedLossWhenMinimizingCategoricalCrossEntropy()
        )

    def testMinimizingMeanSquaredError(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.mse() )
        summary = model.fit(self.inputTensor, self.expectedProbabilityByLabel, epochs=5)

        self._assertElementsAreAllClose(
            summary.history['loss'],
            self.expectedLossWhenMinimizingMeanSquaredError()
        )

    def testMinimizingSparseCategoricalCrossEntropy(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.sparseCategoricalCrossEntropy() )
        summary = model.fit(self.inputTensor, self.expectedLabels, epochs=5)

        self._assertElementsAreAllClose(
            summary.history['loss'],
            self.expectedLossWhenMinimizingSparseCategoricalCrossEntropy()
        )

class TrainingUsingAdagradTest(TrainingUsingOptimizationTest, unittest.TestCase):
    def optimizationAlgorithm(self):
        return tf.keras.optimizers.Adagrad()

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        return [0.846859931945801, 0.846550583839417, 0.846288204193115, 0.846055388450623, 0.845843434333801]

    def expectedLossWhenMinimizingMeanSquaredError(self):
        return [0.265, 0.264025, 0.263223, 0.262523, 0.261893]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        return [0.887488, 0.886098, 0.884969, 0.883992, 0.883118]

class TrainingUsingAdamTest(TrainingUsingOptimizationTest, unittest.TestCase):
    def optimizationAlgorithm(self):
        return tf.keras.optimizers.Adam()

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        return [0.84686, 0.846392, 0.845924, 0.845458, 0.844992]

    def expectedLossWhenMinimizingMeanSquaredError(self):
        return [0.265, 0.263406, 0.261825, 0.260258, 0.258703]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        return [0.887488, 0.885441, 0.883401, 0.881369, 0.879346]

class TrainingUsingGradientDescentTest(TrainingUsingOptimizationTest, unittest.TestCase):
    def optimizationAlgorithm(self):
        return tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        return [0.846859931945801, 0.845578074455261, 0.844308912754059, 0.843052387237549, 0.841808199882507]

    def expectedLossWhenMinimizingMeanSquaredError(self):
        return [0.265, 0.260642, 0.256446, 0.252405, 0.248514]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        return [0.887488, 0.881097, 0.874819, 0.86865, 0.86259]

class TrainingUsingMomentumTest(TrainingUsingOptimizationTest, unittest.TestCase):
    def optimizationAlgorithm(self):
        return tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        return [0.846859931945801, 0.84673154354095, 0.846487581729889, 0.84614038467407, 0.845700562000275]

    def expectedLossWhenMinimizingMeanSquaredError(self):
        return [0.265, 0.26456 , 0.263728, 0.262549, 0.261064]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        return [0.887488, 0.886846, 0.885629, 0.883899, 0.88171]

class TrainingUsingRMSPropTest(TrainingUsingOptimizationTest, unittest.TestCase):
    def optimizationAlgorithm(self):
        return tf.keras.optimizers.RMSprop()

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        return [0.846859931945801, 0.84538102149963, 0.844323873519897, 0.84344661235809, 0.842673122882843]

    def expectedLossWhenMinimizingMeanSquaredError(self):
        return [0.265, 0.260003, 0.256497, 0.25363, 0.251136]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        return [0.887488, 0.88104 , 0.876435, 0.872622, 0.869269]

if __name__ == '__main__':
    unittest.main()

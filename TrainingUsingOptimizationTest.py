import unittest
import numpy as np
import tensorflow as tf

from tensorflow import keras

tf.compat.v1.disable_eager_execution()


class CustomCallback(keras.callbacks.Callback):
    """
        Training Callbacks
    """
    def on_train_begin(self, logs=None):
        print("\nStarting training; got logs: {}".format(logs))

    def on_epoch_begin(self, epoch, logs=None):
        print("\nStart epoch {} of training; got logs: {}".format(epoch, logs))

    def on_train_batch_begin(self, batch, logs=None):
        print("\n...Training: start of batch {}; got logs: {}".format(batch, logs))

    def on_train_batch_end(self, batch, logs=None):
        print("\n...Training: end of batch {}; got logs: {}".format(batch, logs))

    def on_epoch_end(self, epoch, logs=None):
        print("\nEnd epoch {} of training; got logs: {}".format(epoch, logs))

    def on_train_end(self, logs=None):
        print("\nStop training; got logs: {}".format(logs))

    """ 
        Validation Callbacks
    """
    def on_test_begin(self, logs=None):
        print("\nStart testing; got logs: {}".format(logs))

    def on_test_batch_begin(self, batch, logs=None):
        print("\n...Evaluating: start of batch {}; got logs: {}".format(batch, logs))

    def on_test_batch_end(self, batch, logs=None):
        print("\n...Evaluating: end of batch {}; got logs: {}".format(batch, logs))

    def on_test_end(self, logs=None):
        print("\nStop testing; got logs: {}".format(logs))

    """ 
       Testing Callbacks
   """
    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("\nStart predicting; got log keys: {}".format(keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("\n...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("\n...Predicting: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("\nStop predicting; got log keys: {}".format(keys))


class TrainingUsingOptimizationTest(object):
    inputTensor = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 1]])
    expectedProbabilityByLabel = np.array([[0., 1.], [1., 0.], [0., 1.], [1., 1.]])
    expectedLabels = np.array([0, 1, 0, 0])

    validationInputTensor = np.array([
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0]])
    expectedValidationProbabilityByLabel = np.array([[0., 1.], [1., 0.], [1., 0.], [0., 1.], [1., 0.]])
    expectedValidationLabels = np.array([1, 0, 0, 1, 0])

    def _modelWithTwoOutputUnits(self):
        weightInit = tf.keras.initializers.Constant(value=((0, 0), (0, 0), (0, 0)))
        biasInit = tf.keras.initializers.Constant(value=(0.2, 0.8))

        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(3,)))
        model.add(tf.keras.layers.Dense(2, kernel_initializer=weightInit, bias_initializer=biasInit))

        return model

    def _assertElementsAreAllClose(self, anArray, anExpectedArray):
        np.testing.assert_allclose(
            anArray, anExpectedArray,
            rtol=1e-5, atol=0)

    def _assertHasTheSameElementsThat(self, aTensorFlowArray, anExpectedArray):
        self._assertElementsAreAllClose(aTensorFlowArray.numpy(), anExpectedArray)

    def _subclassResponsibility(self):
        raise NotImplementedError('Subclass responsibility')

    def optimizationAlgorithm(self):
        self._subclassResponsibility()

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        self._subclassResponsibility()

    def expectedLossWhenMinimizingCategoricalCrossEntropyInBatches(self):
        self._subclassResponsibility()

    def expectedLossWhenMinimizingMeanSquaredError(self):
        self._subclassResponsibility()

    def expectedLossWhenMinimizingMeanSquaredErrorInBatches(self):
        self._subclassResponsibility()

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        self._subclassResponsibility()

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropyInBatches(self):
        self._subclassResponsibility()

    def expectedValidationLossWhenMinimizingMeanSquaredErrorInBatches(self):
        return [0, 0, 0, 0, 0]

    def categoricalCrossEntropy(self):
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def mse(self):
        return tf.keras.losses.MeanSquaredError()

    def sparseCategoricalCrossEntropy(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def testMinimizingCategoricalCrossEntropy(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.categoricalCrossEntropy())
        summary = model.fit(self.inputTensor, self.expectedProbabilityByLabel, epochs=5)

        self._assertElementsAreAllClose(
            summary.history['loss'],
            self.expectedLossWhenMinimizingCategoricalCrossEntropy()
        )

    def testMinimizingCategoricalCrossEntropyInBatches(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.categoricalCrossEntropy())
        summary = model.fit(self.inputTensor, self.expectedProbabilityByLabel, epochs=5, batch_size=2, shuffle=False)

        self._assertElementsAreAllClose(
            summary.history['loss'],
            self.expectedLossWhenMinimizingCategoricalCrossEntropyInBatches()
        )

    def testMinimizingMeanSquaredError(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.mse())
        summary = model.fit(self.inputTensor, self.expectedProbabilityByLabel, epochs=5)

        self._assertElementsAreAllClose(
            summary.history['loss'],
            self.expectedLossWhenMinimizingMeanSquaredError()
        )

    def testMinimizingMeanSquaredErrorInBatches(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.mse())
        summary = model.fit(self.inputTensor, self.expectedProbabilityByLabel, epochs=5, batch_size=2, shuffle=False)

        self._assertElementsAreAllClose(
            summary.history['loss'],
            self.expectedLossWhenMinimizingMeanSquaredErrorInBatches()
        )

    def testValidationLossWhenMinimizingMeanSquaredErrorInBatches(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.mse())
        summary = model.fit(
            self.inputTensor,
            self.expectedProbabilityByLabel,
            validation_data=(self.validationInputTensor, self.expectedValidationProbabilityByLabel),
            epochs=5, batch_size=2, shuffle=False,
            callbacks=[CustomCallback()])

        self._assertElementsAreAllClose(
            summary.history['val_loss'],
            self.expectedValidationLossWhenMinimizingMeanSquaredErrorInBatches()
        )

    def testMinimizingSparseCategoricalCrossEntropy(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.sparseCategoricalCrossEntropy())
        summary = model.fit(self.inputTensor, self.expectedLabels, epochs=5)

        self._assertElementsAreAllClose(
            summary.history['loss'],
            self.expectedLossWhenMinimizingSparseCategoricalCrossEntropy()
        )

    def testMinimizingSparseCategoricalCrossEntropyInBatches(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.sparseCategoricalCrossEntropy())
        summary = model.fit(self.inputTensor, self.expectedLabels, epochs=5, batch_size=2, shuffle=False)

        self._assertElementsAreAllClose(
            summary.history['loss'],
            self.expectedLossWhenMinimizingSparseCategoricalCrossEntropyInBatches()
        )

    def testAccuracyMinimizingMeanSquaredErrorInBatches(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.mse(), metrics=['accuracy'])
        summary = model.fit(self.inputTensor, self.expectedProbabilityByLabel, epochs=5, batch_size=2, shuffle=False)

        self._assertElementsAreAllClose(
            summary.history['accuracy'],
            [0.5, 0.5, 0.5, 0.5, 0.5]
        )

    def testAccuracyMinimizingSparseCategoricalCrossEntropyInBatches(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.sparseCategoricalCrossEntropy(), metrics=['accuracy'])
        summary = model.fit(self.inputTensor, self.expectedLabels, epochs=5, batch_size=2, shuffle=False)

        self._assertElementsAreAllClose(
            summary.history['accuracy'],
            [0.25, 0.25, 0.25, 0.25, 0.25]
        )

    def testAccuracyMinimizingCategoricalCrossEntropyInBatches(self):
        model = self._modelWithTwoOutputUnits()
        model.compile(optimizer=self.optimizationAlgorithm(), loss=self.categoricalCrossEntropy(), metrics=['accuracy'])
        summary = model.fit(self.inputTensor, self.expectedProbabilityByLabel, epochs=5, batch_size=2, shuffle=False)

        self._assertElementsAreAllClose(
            summary.history['accuracy'],
            [0.5, 0.5, 0.5, 0.5, 0.5]
        )

class TrainingUsingAdagradTest(TrainingUsingOptimizationTest, unittest.TestCase):
    def optimizationAlgorithm(self):
        return tf.keras.optimizers.Adagrad()

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        return [0.846859931945801, 0.846550583839417, 0.846288204193115, 0.846055388450623, 0.845843434333801]

    def expectedLossWhenMinimizingCategoricalCrossEntropyInBatches(self):
        return [0.846709, 0.846175, 0.845769, 0.845428, 0.845129]

    def expectedLossWhenMinimizingMeanSquaredError(self):
        return [0.265, 0.264025, 0.263223, 0.262523, 0.261893]

    def expectedLossWhenMinimizingMeanSquaredErrorInBatches(self):
        return [0.264746, 0.263134, 0.261926, 0.260919, 0.260039]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        return [0.887488, 0.886098, 0.884969, 0.883992, 0.883118]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropyInBatches(self):
        return [0.887241, 0.885236, 0.883753, 0.88252 , 0.881443]

    def expectedValidationLossWhenMinimizingMeanSquaredErrorInBatches(self):
        return [0.399217, 0.398649, 0.398183, 0.397781, 0.397422]

class TrainingUsingAdamTest(TrainingUsingOptimizationTest, unittest.TestCase):
    def optimizationAlgorithm(self):
        return tf.keras.optimizers.Adam()

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        return [0.84686, 0.846392, 0.845924, 0.845458, 0.844992]

    def expectedLossWhenMinimizingCategoricalCrossEntropyInBatches(self):
        return [0.846602, 0.845378, 0.844389, 0.84345, 0.842537]

    def expectedLossWhenMinimizingMeanSquaredError(self):
        return [0.265, 0.263406, 0.261825, 0.260258, 0.258703]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        return [0.887488, 0.885441, 0.883401, 0.881369, 0.879346]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropyInBatches(self):
        return [0.886843, 0.883549, 0.880392, 0.87724, 0.874098]

    def expectedLossWhenMinimizingMeanSquaredErrorInBatches(self):
        return [0.264653, 0.261752, 0.259041, 0.256396, 0.253798]

    def expectedValidationLossWhenMinimizingMeanSquaredErrorInBatches(self):
        return [0.398473, 0.397191, 0.395982, 0.394817, 0.393687]

class TrainingUsingGradientDescentTest(TrainingUsingOptimizationTest, unittest.TestCase):
    def optimizationAlgorithm(self):
        return tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        return [0.846859931945801, 0.845578074455261, 0.844308912754059, 0.843052387237549, 0.841808199882507]

    def expectedLossWhenMinimizingCategoricalCrossEntropyInBatches(self):
        return [0.846232, 0.843706, 0.841229, 0.8388  , 0.836417]

    def expectedLossWhenMinimizingMeanSquaredError(self):
        return [0.265, 0.260642, 0.256446, 0.252405, 0.248514]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        return [0.887488, 0.881097, 0.874819, 0.86865, 0.86259]

    def expectedLossWhenMinimizingMeanSquaredErrorInBatches(self):
        return [0.263827, 0.255408, 0.2476, 0.240354, 0.233622]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropyInBatches(self):
        return [0.88665, 0.874014, 0.861818, 0.850049, 0.838693]

    def expectedValidationLossWhenMinimizingMeanSquaredErrorInBatches(self):
        return [0.396317, 0.392976, 0.389949, 0.387208, 0.384729]

class TrainingUsingMomentumTest(TrainingUsingOptimizationTest, unittest.TestCase):
    def optimizationAlgorithm(self):
        return tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        return [0.846859931945801, 0.84673154354095, 0.846487581729889, 0.84614038467407, 0.845700562000275]

    def expectedLossWhenMinimizingCategoricalCrossEntropyInBatches(self):
        return [0.846796, 0.846275, 0.845369, 0.844159, 0.842708]

    def expectedLossWhenMinimizingMeanSquaredError(self):
        return [0.265, 0.26456, 0.263728, 0.262549, 0.261064]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        return [0.887488, 0.886846, 0.885629, 0.883899, 0.88171]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropyInBatches(self):
        return [0.887404, 0.884977, 0.880603, 0.874698, 0.867613]

    def expectedLossWhenMinimizingMeanSquaredErrorInBatches(self):
        return [0.26488, 0.263229, 0.260271, 0.256324, 0.251661]

    def expectedValidationLossWhenMinimizingMeanSquaredErrorInBatches(self):
        return [0.399338, 0.398093, 0.396411, 0.39442 , 0.392229]

class TrainingUsingRMSPropTest(TrainingUsingOptimizationTest, unittest.TestCase):
    def optimizationAlgorithm(self):
        return tf.keras.optimizers.RMSprop()

    def expectedLossWhenMinimizingCategoricalCrossEntropy(self):
        return [0.846859931945801, 0.84538102149963, 0.844323873519897, 0.84344661235809, 0.842673122882843]

    def expectedLossWhenMinimizingCategoricalCrossEntropyInBatches(self):
        return [0.84606 , 0.843515, 0.841959, 0.840685, 0.839558]

    def expectedLossWhenMinimizingMeanSquaredError(self):
        return [0.265, 0.260003, 0.256497, 0.25363, 0.251136]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropy(self):
        return [0.887488, 0.88104, 0.876435, 0.872622, 0.869269]

    def expectedLossWhenMinimizingSparseCategoricalCrossEntropyInBatches(self):
        return [0.885448, 0.877409, 0.872078, 0.867611, 0.863616]

    def expectedLossWhenMinimizingMeanSquaredErrorInBatches(self):
        return [0.263918, 0.257379, 0.252959, 0.249319, 0.246109]

    def expectedValidationLossWhenMinimizingMeanSquaredErrorInBatches(self):
        return [0.396749, 0.394732, 0.393124, 0.391734, 0.390484]

if __name__ == '__main__':
    unittest.main()

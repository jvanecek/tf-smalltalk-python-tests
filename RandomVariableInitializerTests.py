import unittest
import numpy as np
import tensorflow as tf


class VariableInitializerTest(object):
    def _subclassResponsibility(self):
        raise NotImplementedError('Subclass responsibility')

    def _createDefaultInitializer(self):
        self._subclassResponsibility()

    def _createCustomInitializer(self):
        self._subclassResponsibility()

    def _expectedMatrixValues(self):
        self._subclassResponsibility()

    def _expectedScalarValue(self):
        self._subclassResponsibility()

    def _expectedVectorValues(self):
        self._subclassResponsibility()

    def testInitializeMatrixVariable(self):
        initializer = self._createDefaultInitializer()
        variable = initializer(shape=(2, 3))
        np.testing.assert_array_almost_equal(self._expectedMatrixValues(), variable.numpy())

    def testInitializeScalarVariable(self):
        initializer = self._createDefaultInitializer()
        variable = initializer(shape=())
        self.assertAlmostEqual(self._expectedScalarValue(), variable.numpy())

    def testInitializeVectorVariable(self):
        initializer = self._createCustomInitializer()
        variable = initializer(shape=(3,))
        np.testing.assert_array_almost_equal(self._expectedVectorValues(),
                                             variable.numpy())


class GlorotNormalInitializerTest(VariableInitializerTest, unittest.TestCase):
    def _createDefaultInitializer(self):
        return tf.keras.initializers.GlorotNormal(seed=1)

    def _createCustomInitializer(self):
        return tf.keras.initializers.GlorotNormal(seed=2)

    def _expectedMatrixValues(self):
        return [[0.091062, -0.354482, 0.453829],
                [-0.567185, -0.654192, -0.287002]]

    def _expectedScalarValue(self):
        return 0.14398205

    def _expectedVectorValues(self):
        return [-0.419695287942886, -0.122742906212807, -0.543764114379883]


class GlorotUniformInitializerTest(VariableInitializerTest, unittest.TestCase):
    def _createDefaultInitializer(self):
        return tf.keras.initializers.GlorotUniform(seed=1)

    def _createCustomInitializer(self):
        return tf.keras.initializers.GlorotUniform(seed=2)

    def _expectedMatrixValues(self):
        return [[0.829226, -0.087679, 0.219727],
                [-0.235307, -0.540726, -0.122034]]

    def _expectedScalarValue(self):
        return 1.3111216

    def _expectedVectorValues(self):
        return [0.601958, 0.409434, 0.394356]


class RandomUniformInitializerTest(VariableInitializerTest, unittest.TestCase):
    def _createDefaultInitializer(self):
        return tf.keras.initializers.RandomUniform(seed=1)

    def _createCustomInitializer(self):
        return tf.keras.initializers.RandomUniform(maxval=2, seed=3)

    def _expectedMatrixValues(self):
        return [[0.037849, -0.004002, 0.010029],
                [-0.01074, -0.024681, -0.00557]]

    def _expectedScalarValue(self):
        return 0.03784882

    def _expectedVectorValues(self):
        return [0.351621, 1.875658, 1.041164]


class TruncatedNormalInitializerTest(VariableInitializerTest, unittest.TestCase):
    def _createDefaultInitializer(self):
        return tf.keras.initializers.TruncatedNormal(seed=1)

    def _createCustomInitializer(self):
        return tf.keras.initializers.TruncatedNormal(mean=0.3, stddev=0.9, seed=2)

    def _expectedMatrixValues(self):
        return [[0.006333, -0.024651, 0.03156],
                [-0.039442, -0.045493, -0.019958]]

    def _expectedScalarValue(self):
        return 0.0063325153

    def _expectedVectorValues(self):
        return [-0.275486, 0.131695, -0.44561]


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.utils import losses_utils

tf.compat.v1.enable_eager_execution()

class TrainingMinimizingLossTest(object):
    inputTensor = np.array( [
        [ 0, 0, 1 ],
        [ 0, 1, 1],
        [ 1, 0, 0],
        [ 1, 1, 1 ]])
    expectedProbabilityByLabel = np.array([ [0., 1.], [1., 0.], [0., 1.], [1., 1.] ])
    expectedLabels = np.array([ 0, 1, 0, 0 ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.2)

    def _modelWithTwoOutputUnits(self):
        weightInit = tf.keras.initializers.Constant(value=((0, 0),(0, 0),(0, 0)))
        biasInit = tf.keras.initializers.Constant(value=(0.2, 0.8))

        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(3,)))
        model.add(tf.keras.layers.Dense(2, kernel_initializer=weightInit, bias_initializer=biasInit))

        return model

    def _assertHasTheSameElementsThat(self, aTensorFlowArray, anExpectedArray ):
        np.testing.assert_allclose(
            aTensorFlowArray.numpy(), anExpectedArray,
            rtol=1e-5, atol=0)

    def _subclassResponsibility(self):
        raise NotImplementedError('Subclass responsibility')

    def targetTensor(self):
        self._subclassResponsibility()

    def loss(self):
        self._subclassResponsibility()

    def expectedAccuracyAfterOneEpoch(self):
        self._subclassResponsibility()

    def expectedLogitsAfterOneEpoch(self):
        self._subclassResponsibility()

    def expectedLossAfterOneEpoch(self):
        self._subclassResponsibility()

    def expectedWeightAfterOneEpoch(self):
        self._subclassResponsibility()

    def expectedLossValueThroughTenEpochs(self):
        self._subclassResponsibility()

    def expectedAccuracyThroughTenEpochs(self):
        self._subclassResponsibility()

    def _computeLossUsing( self, loss, model ):
        self._subclassResponsibility()

    def testAccuracyAfterMeanSquaredError(self):
        model = self._modelWithTwoOutputUnits()

        loss = self.loss()
        model.compile(optimizer=self.optimizer, loss=loss, metrics=['accuracy'])
        result = model.fit(self.inputTensor, self.targetTensor(), epochs=1)

        assert( result.history['acc'] == self.expectedAccuracyAfterOneEpoch() )

    def testLogitsAfterOneEpoch(self):
        model = self._modelWithTwoOutputUnits()

        loss = self.loss()
        model.compile(optimizer=self.optimizer, loss=loss )
        model.fit(self.inputTensor, self.targetTensor(), epochs=1)

        self._assertHasTheSameElementsThat(
            model( self.inputTensor ),
            self.expectedLogitsAfterOneEpoch()
        )

    def testLossValueAfterOneEpoch(self):
        model = self._modelWithTwoOutputUnits()

        loss = self.loss()
        model.compile(optimizer=self.optimizer, loss=loss )
        model.fit(self.inputTensor, self.targetTensor(), epochs=1)

        self._assertHasTheSameElementsThat(
            self._computeLossUsing( loss, model ),
            self.expectedLossAfterOneEpoch()
        )

    def testLossValueThroughTenEpochs(self):
        model = self._modelWithTwoOutputUnits()

        loss = self.loss()
        model.compile(optimizer=self.optimizer, loss=loss )
        result = model.fit(self.inputTensor, self.targetTensor(), epochs=10)

        np.testing.assert_allclose(
            result.history['loss'],  self.expectedLossValueThroughTenEpochs(),
            rtol=1e-5, atol=0)

    def testAccuracyThroughTenEpochs(self):
        model = self._modelWithTwoOutputUnits()

        loss = self.loss()

        metric = 'accuracy'
        metric_key = 'acc'

        model.compile(optimizer=self.optimizer, loss=loss, metrics=[metric] )
        result = model.fit(self.inputTensor, self.targetTensor(), epochs=10)

        np.testing.assert_allclose(
            result.history[metric_key],  self.expectedAccuracyThroughTenEpochs(),
            rtol=1e-5, atol=0)

    def testWeightAfterOneEpoch(self):
        model = self._modelWithTwoOutputUnits()

        loss = self.loss()
        model.compile(optimizer=self.optimizer, loss=loss )
        model.fit(self.inputTensor, self.targetTensor(), epochs=1)

        self._assertHasTheSameElementsThat(
            model.trainable_weights[0],
            self.expectedWeightAfterOneEpoch()
        )

class TrainingMinimizingMeanSquaredErrorTest(TrainingMinimizingLossTest, unittest.TestCase):
    def targetTensor(self):
        return super().expectedProbabilityByLabel

    def loss(self):
        return tf.keras.losses.MeanSquaredError()

    def expectedAccuracyAfterOneEpoch(self):
        return [0.5]

    def expectedWeightAfterOneEpoch(self):
        return [[ 0.03      ,  0.02],
            [ 0.08000001 , -0.03],
            [ 0.07       , -0.02]]

    def expectedLogitsAfterOneEpoch(self):
        return [[0.32999998, 0.77000004],
            [0.41       , 0.74      ],
            [0.29       , 0.81      ],
            [0.44       , 0.76      ]]

    def expectedLossAfterOneEpoch(self):
        return [0.193613]

    def expectedLossValueThroughTenEpochs(self):
        return [0.26500004529953003, 0.19361251592636108, 0.1633041501045227, 0.14681315422058105, 0.13540230691432953, 0.12621885538101196, 0.11828607320785522, 0.11123108863830566, 0.10488058626651764, 0.09913133084774017]

    def expectedAccuracyThroughTenEpochs(self):
        return [0.5, 0.5, 0.5, 0.5, 0.75, 1, 1, 1, 1, 1]

    def _computeLossUsing(self, loss, model):
        return loss( model(self.inputTensor), self.targetTensor() )

class TrainingMinimizingCategoricalCrossEntropyTest(TrainingMinimizingLossTest, unittest.TestCase):
    def targetTensor(self):
        return super().expectedProbabilityByLabel

    def loss(self):
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def expectedAccuracyAfterOneEpoch(self):
        return [0.5]

    def expectedWeightAfterOneEpoch(self):
        return [[ 0.01456563, 0.03543437],
            [ 0.06456564 ,-0.01456563],
            [ 0.04684845 , 0.00315156]]

    def expectedLogitsAfterOneEpoch(self):
        return [[0.27597973, 0.82402027],
            [0.34054536 , 0.8094547 ],
            [0.2436969  , 0.8563031 ],
            [0.355111   , 0.84488904]]

    def expectedLossAfterOneEpoch(self):
        return [0.822441]

    def expectedLossValueThroughTenEpochs(self):
        return [0.8468599915504456, 0.8224405646324158, 0.8024106025695801, 0.7854786515235901, 0.7707569003105164, 0.7576471567153931, 0.7457488179206848, 0.7347933053970337, 0.7245980501174927, 0.7150363326072693]

    def expectedAccuracyThroughTenEpochs(self):
        return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75]

    def _computeLossUsing(self, loss, model):
        return loss( self.targetTensor(), model(self.inputTensor) )

class TrainingMinimizingSparseCategoricalCrossEntropyTest(TrainingMinimizingLossTest, unittest.TestCase):
    def targetTensor(self):
        return super().expectedLabels

    def loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def expectedAccuracyAfterOneEpoch(self):
        return [0.25]

    def expectedWeightAfterOneEpoch(self):
        return [[ 0.06456564, -0.06456563],
            [ 0.01456563, -0.01456563],
            [ 0.04684845, -0.04684844]]

    def expectedLogitsAfterOneEpoch(self):
        return [[0.3259797, 0.67402035],
            [0.34054536, 0.6594547 ],
            [0.3436969 , 0.65630317],
            [0.40511099, 0.59488904]]

    def expectedLossAfterOneEpoch(self):
        return [0.770683]

    def expectedLossValueThroughTenEpochs(self):
        return [0.8874880075454712, 0.7706831693649292, 0.6920742988586426, 0.6382837295532227, 0.5999782681465149, 0.571312427520752, 0.548761248588562, 0.530205249786377, 0.5143527388572693, 0.5004007816314697]

    def expectedAccuracyThroughTenEpochs(self):
        return [0.25, 0.25, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

    def _computeLossUsing(self, loss, model):
        return loss( self.targetTensor(), model(self.inputTensor) )

if __name__ == '__main__':
    unittest.main()

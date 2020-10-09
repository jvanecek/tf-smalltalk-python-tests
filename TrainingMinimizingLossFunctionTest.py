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

    def _computeLossUsing(self, loss, model): 
        return loss( self.targetTensor(), model(self.inputTensor) )

if __name__ == '__main__':
    unittest.main()

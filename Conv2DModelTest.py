import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.initializers import GlorotUniform, Zeros

import unittest
import numpy as np

class TestConv2Models(unittest.TestCase):

    def inputForConv2D(self):
        return np.ones((1, 28, 28, 1))

    def assertModelOutput(self, model_layers, input, expected_output_size, expected_output=None):

        model_cnn = Sequential(model_layers)

        model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        output = model_cnn.predict(input)

        self.assertEqual(output.shape, expected_output_size)

        if expected_output != None:
            np.testing.assert_almost_equal(output, expected_output, decimal=6)

    def testModel1Layer(self):

        self.assertModelOutput(
            model_layers=[
                Input((28, 28, 1)),
                Conv2D(32, (3, 3), kernel_initializer=GlorotUniform(seed=42)),
            ],
            input=self.inputForConv2D(),
            expected_output_size=(1, 26, 26, 32)
        )


    def testModel2Layer(self):

        self.assertModelOutput(
            model_layers=[
                Input((28, 28, 1)),
                Conv2D(32, (3, 3), kernel_initializer=GlorotUniform(seed=42)),
                MaxPooling2D((2, 2))
            ],
            input=self.inputForConv2D(),
            expected_output_size=(1, 13, 13, 32)
        )

        
    def testModel3Layers(self):                    

        self.assertModelOutput(
            model_layers=[
                Input((28, 28, 1)),
                Conv2D(32, (3, 3), kernel_initializer=GlorotUniform(seed=42)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), kernel_initializer=GlorotUniform(seed=42))
            ],
            input=self.inputForConv2D(),
            expected_output_size=(1, 11, 11, 64)
        )
    
    def testModel4Layers(self):                    

        self.assertModelOutput(
            model_layers=[
                Input((28, 28, 1)),
                Conv2D(32, (3, 3), kernel_initializer=GlorotUniform(seed=42)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), kernel_initializer=GlorotUniform(seed=42)),
                MaxPooling2D((2, 2))
            ],
            input=self.inputForConv2D(),
            expected_output_size=(1, 5, 5, 64)
        )
   
    def testModel5Layers(self):                    

        self.assertModelOutput(
            model_layers=[
                Input((28, 28, 1)),
                Conv2D(32, (3, 3), kernel_initializer=GlorotUniform(seed=42)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), kernel_initializer=GlorotUniform(seed=42)),
                MaxPooling2D((2, 2)),
                Flatten(),
            ],
            input=self.inputForConv2D(),
            expected_output_size=(1, 1600)
        )

    def testModel6Layers(self):

        self.assertModelOutput(
            model_layers=[
                Input((28, 28, 1)),
                Conv2D(32, (3, 3), kernel_initializer=GlorotUniform(seed=42)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), kernel_initializer=GlorotUniform(seed=42)),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu', kernel_initializer=GlorotUniform(seed=42))
            ],
            input=self.inputForConv2D(),
            expected_output_size=(1, 128)
        )

    def testModel7Layers(self):

        self.assertModelOutput(
            model_layers=[
                Input((28, 28, 1)),
                Conv2D(32, (3, 3), kernel_initializer=GlorotUniform(seed=42)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), kernel_initializer=GlorotUniform(seed=42)),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu', kernel_initializer=GlorotUniform(seed=42)),
                Dense(10, activation='softmax', kernel_initializer=GlorotUniform(seed=42))
            ],
            input=self.inputForConv2D(),
            expected_output_size=(1, 10),
            expected_output=[[0.07875562, 0.06159625, 0.1312697, 0.07310834, 0.05539258, 0.18855572, 0.07790892, 0.08097104, 0.20204508, 0.05039675]]
        )
        
if __name__ == '__main__':
    unittest.main()

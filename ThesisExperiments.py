import unittest
import tensorflow as tf
from tensorflow import keras

tf.compat.v1.disable_eager_execution()

class ThesisExperimentTest(unittest.TestCase):

    def testExperiment1(self):
        # https://www.geeksforgeeks.org/python-classifying-handwritten-digits-with-tensorflow/
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        train_images = keras.utils.normalize(train_images, axis=1)
        test_images = keras.utils.normalize(test_images, axis=1)

        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()])

        log_dir = "logs/experiment-1/python"
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                                              write_steps_per_second=True)
        model.fit(
            x=train_images,
            y=train_labels,
            epochs=10,
            validation_data=(test_images, test_labels),
            callbacks=[
                tensorboard_callback
            ])

        val_loss, val_acc = model.evaluate(test_images, test_labels)
        print("loss-> ", val_loss, "\nacc-> ", val_acc)

    def testExperiment2(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        train_images = train_images / 255.0
        test_images = test_images / 255.0
    
        # Create tf.data.Dataset objects for training and testing
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

        # Shuffle and batch the datasets
        train_dataset = train_dataset.shuffle(buffer_size=60000).batch(32)
        test_dataset = test_dataset.batch(32)

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])

        model.compile(optimizer='adam', # keras.optimizers.Adam(learning_rate=0.001)
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()])

        # Train the model using the Dataset objects
        history = model.fit(
            train_dataset,
            epochs=10,
            validation_data=test_dataset
        )

        val_loss, val_acc = model.evaluate(test_images, test_labels)
        print("loss-> ", val_loss, "\nacc-> ", val_acc)

    def testExperiment2Prediction(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])
        prediction = model.predict(
            x=train_images[:1],
            batch_size=32
        )
        print(prediction.shape)
        print(prediction)

    def testProfileExperiment2(self):

        #pr = cProfile.Profile()
        #pr.enable()

        self.testExperiment2()

        #pr.disable()
        #s = io.StringIO()
        #sortby = SortKey.CUMULATIVE
        #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #ps.print_stats()
        #print(s.getvalue())

if __name__ == '__main__':
    unittest.main()

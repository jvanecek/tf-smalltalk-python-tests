import unittest
import tensorflow as tf
from tensorflow import keras
from keras.engine import data_adapter

import cProfile, pstats, io
from pstats import SortKey

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

class ThesisExperimentTest(unittest.TestCase):

    def testExperiment1(self):
        # https://www.geeksforgeeks.org/python-classifying-handwritten-digits-with-tensorflow/
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        train_images = tf.keras.utils.normalize(train_images, axis=1)
        test_images = tf.keras.utils.normalize(test_images, axis=1)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        log_dir = "logs/experiment-1/python"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
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

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        log_dir = "logs/experiment-2/python"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_steps_per_second=True)

        history = model.fit(
            x=train_images,
            y=train_labels,
            epochs=10,
            batch_size=32,
            validation_data=(test_images, test_labels),
            callbacks=[
                tensorboard_callback
            ]
        )

        val_loss, val_acc = model.evaluate(test_images, test_labels)
        print("loss-> ", val_loss, "\nacc-> ", val_acc)

    def testExperiment2Prediction(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
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

    def testStoreSessionGraph(self):
        from keras import backend as K
        tf.compat.v1.disable_eager_execution()

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
        tf.io.write_graph(frozen_graph, ".", "experiment2-keras-before-compile.pb", as_text=False)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])

        frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
        tf.io.write_graph(frozen_graph, ".", "experiment2-keras-after-compile.pb", as_text=False)

        # save in folder with metadata in separated file
        model.save('experiment2-keras')
        # save all in a .h5 file
        model.save('./experiment2-keras.h5')

        print(model.inputs)
        # [<tf.Tensor 'flatten_input:0' shape=(None, 28, 28) dtype=float32>]
        print(model.outputs)
        # [<tf.Tensor 'dense_1/BiasAdd:0' shape=(None, 10) dtype=float32>]

        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        train_images = train_images / 255.0
        prediction = model.predict(
            x=train_images[:1],
            batch_size=32
        )
        print(prediction.shape)
        print(prediction)

    def testRestoreModel(self):
        from keras.models import load_model

        model = load_model('experiment2-keras')
        #print(model.outputs)
        # [<tf.Tensor 'dense_1/BiasAdd:0' shape=(None, 10) dtype=float32>]
        #print(model.inputs)
        # [<tf.Tensor 'flatten_input:0' shape=(None, 28, 28) dtype=float32>]

        # load data, and predict something
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        print( train_images[0].sum() )
        print( test_images[0].sum() )

        # print(test_images[:2])
        prediction = model.predict(test_images[:1])
        print( prediction )
        # [[
        #    -75.89334    43.540054  -78.760605  -58.523266 -123.21959
        #    237.84773  -212.87595  -127.078354   29.158485   99.33598
        # ]]

    def testRestoreSessionGraph(self):
        from keras import backend as K
        from tensorflow.python.platform import gfile

        with gfile.GFile("./experiment2-keras-after-compile.pb", 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            # Parses a serialized binary message into the current message.
            graph_def.ParseFromString(f.read())

        session = K.get_session()
        session.graph.as_default()

        # Import a serialized TensorFlow `GraphDef` protocol buffer
        # and place into the current default `Graph`.
        tf.import_graph_def(graph_def)

        # load data, and predict something
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        output = session.graph.get_tensor_by_name('import/dense_1/BiasAdd:0')
        predictions = session.run(output, {'import/flatten_input:0': test_images[:1]})
        # [[
        #   -75.89334    43.540054  -78.760605  -58.523266 -123.21959
        #   237.84773  -212.87595  -127.078354   29.158485   99.33598
        # ]]

    def testRestoreSmalltalkModel(self):
        from keras import backend as K
        from tensorflow.python.platform import gfile

        with gfile.GFile("C:/Users/juann/proyectos-personales/tensorflow-vast-12/experiment2-smalltalk-model.pb", 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            # Parses a serialized binary message into the current message.
            graph_def.ParseFromString(f.read())

        session = K.get_session()
        session.graph.as_default()

        # Import a serialized TensorFlow `GraphDef` protocol buffer
        # and place into the current default `Graph`.
        tf.import_graph_def(graph_def)

        # load data, and predict something
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        # Initialize variables
        init_op = tf.compat.v1.global_variables_initializer()
        session.run(init_op)

        output = session.graph.get_tensor_by_name('import/dense_1/BiasAdd:0')
        predictions = session.run(output, {'import/flatten_input:0': test_images[:1]})
        # [[
        #   -75.89334    43.540054  -78.760605  -58.523266 -123.21959
        #   237.84773  -212.87595  -127.078354   29.158485   99.33598
        # ]]

    def testDatasetHandlers(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        steps_per_execution = tf.Variable(
            1,
            dtype="int64",
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

        data_handler = data_adapter.get_data_handler(
            x=x_train,
            y=y_train,
            sample_weight=None,
            batch_size=None,
            steps_per_epoch=None,
            initial_epoch=0,
            epochs=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            model=self,
            steps_per_execution=steps_per_execution,
        )
        for epoch, iterator in data_handler.enumerate_epochs():
            print(epoch)
            with data_handler.catch_stop_iteration():
                data_handler._initial_step = data_handler._initial_step
                for step in data_handler.steps():
                    print(step)
                    print(iterator)
                    end_step = step + data_handler.step_increment
                    print(end_step)

if __name__ == '__main__':
    unittest.main()

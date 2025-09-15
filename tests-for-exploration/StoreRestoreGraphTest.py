import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
import unittest

tf.compat.v1.disable_eager_execution()

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

class StoreRestoreGraphTest(unittest.TestCase):

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
            # steps_per_execution defaults to 1,
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
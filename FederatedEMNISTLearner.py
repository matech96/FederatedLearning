import collections
from typing import List

import tensorflow_federated as tff
import tensorflow as tf

from FederatedLearner import FederatedLearner


class FederatedMNISTLearner(FederatedLearner):
    def load_data(self) -> List:  # [BatchDataset]
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()  # TODO use test dataset
        # TODO dynamicly set these
        NUM_CLIENTS = 10
        NUM_EPOCHS = 10
        BATCH_SIZE = 20
        SHUFFLE_BUFFER = 500

        def preprocess(dataset):
            def element_fn(element):
                return collections.OrderedDict(
                    [
                        ("x", tf.reshape(element["pixels"], [-1])),
                        ("y", tf.reshape(element["label"], [1])),
                    ]
                )

            return (
                dataset.repeat(NUM_EPOCHS)
                .map(element_fn)
                .shuffle(SHUFFLE_BUFFER)
                .batch(BATCH_SIZE)
            )

        sample_clients = emnist_train.client_ids[
            0:NUM_CLIENTS
        ]  # TODO dynamic client selection
        return [
            preprocess(emnist_train.create_tf_dataset_for_client(x))
            for x in sample_clients
        ]

    def build_model(self) -> tf.keras.Model:
        # print("build_model")
        # model = tf.keras.models.Sequential(
        #     [
        #         tf.keras.layers.Dense(
        #             10,
        #             activation=tf.nn.softmax,
        #             kernel_initializer="zeros",
        #             input_shape=(784,),
        #         )
        #     ]
        # )
        # print("model")

        # model.compile(
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        #     optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
        #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        # )
        # print("compiled")
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(10, kernel_initializer='zeros'),
            tf.keras.layers.Softmax(),
        ])

        # model.compile(
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        #     optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
        #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        # )
        return model

    def get_loss(self) -> tf.keras.losses.Loss:
        return tf.keras.losses.SparseCategoricalCrossentropy()

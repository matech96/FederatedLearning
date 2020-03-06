from abc import ABC, abstractmethod
from typing import List

import tensorflow as tf
import tensorflow_federated as tff
from comet_ml.exceptions import InterruptedExperiment
# from tensorflow.python.data.ops.dataset_ops import BatchDataset

from comet_ml import Experiment


class FederatedLearner(ABC):

    def __init__(self, experiment: Experiment) -> None:
        """
        Initialises the training.
        :param experiment: Comet.ml experiment object for online logging.
        """
        super().__init__()
        self.experiment = experiment

    @abstractmethod
    def load_data(self) -> List: # BatchDataset
        """
        Loads the data to BatchDataset. The BatchDataset should contain and OrderedDict([(x, tf.Tensor(
        dtype=tf.float32)), (y, tf.Tensor(tf.int32))]) @return: BatchDataset
        """
        pass

    @abstractmethod
    def compile_model(self) -> tf.keras.Model:
        """
        Builds a keras Model without compiling it.
        @return: not compiled keras Model
        """
        pass

    @abstractmethod
    def get_loss(self) -> tf.keras.losses.Loss:
        """
        Returns the training loss
        @return: training loss
        """
        pass

    def train(self, n_rounds: int, client_sample_percent: float) -> None:
        """
        Runs the federated training, reports to comet.ml and runs an evaluation at the end.
        @param n_rounds: The number of round for training (analogous for number of epochs).
        @param client_sample_percent: The percentage of clients to participate in a round. Muss be between 0 and 1.
        """
        federated_train_data = self.load_data()
        sample_batch = next(iter(federated_train_data[0]))

        def model_fn():
            keras_model = self.compile_model()
            loss = self.get_loss()
            return tff.learning.from_keras_model(keras_model, sample_batch, loss)

        iterative_process = tff.learning.build_federated_averaging_process(model_fn)

        state = iterative_process.initialize()
        try:
            for round_num in range(n_rounds):
                state, metrics = iterative_process.next(state, federated_train_data)
                for value, name in zip(metrics, dir(metrics)):
                    self.experiment.log_metric(name, value, step=round_num)
        except InterruptedExperiment:
            pass


from typing import Callable

import numpy as np
from numpy.random import default_rng

rand_gen = default_rng()


class Perceptron:
    def __init__(self, dims: int):
        self._d = dims
        # initialize weights randomly
        # bias trick, starting theta is 0
        self._w = np.append(rand_gen.standard_normal(dims), 0)
        self._epochs = 0

    @staticmethod
    def _activation_function(y_prima: np.ndarray) -> int:
        return 1 if np.greater(y_prima, 0) else 0

    @staticmethod
    def _error(y: np.ndarray, y_prima: np.ndarray, target: np.ndarray):
        return target - y

    def epoch(self,
              data: np.ndarray,
              eta: float = 1.0,
              callback: Callable[[int, np.ndarray, np.ndarray, int],
                                 None] = lambda e, w, dw, m: None) \
            -> bool:
        # last element of each sample should be the class
        assert data.shape[1] == self._d + 1

        # shuffle data
        samples = data.copy()
        rand_gen.shuffle(samples)

        # deltaW
        dw = np.zeros(shape=self._w.shape)

        # count misclassified
        misses = 0

        for x in samples:
            x, target = np.split(x, [-1])
            # bias trick
            x = np.append(x, 1)

            # threshold
            y_prima = np.dot(self._w.T, x)
            y = self._activation_function(y_prima)

            # learning step
            if not np.isclose(y, target):
                misses += 1
                error = self._error(y, y_prima, target)
                etaX = (error * eta) * x
                dw = dw + etaX

        # finally, add dw to the weight vector
        self._w = self._w + dw
        self._epochs += 1

        callback(self._epochs, self._w, dw, misses)
        return misses == 0  # finishing condition

    def train(self,
              data: np.ndarray,
              eta0: float = 1.0,
              epoch_cb: Callable[[int, np.ndarray, np.ndarray, int],
                                 None] = lambda e, w, dw: None):
        """
        Train this perceptron on the given input data.

        :param data: Data to train on. Each row in this matrix should
        correspond to a training sample, where the last element of the row
        represents the expected class for the sample.

        :param eta0: Starting training rate.

        :param epoch_cb: Optional callback executed after each epoch. This
        function will be called with parameters epoch, weight vector,
        dW and the number of classification misses. Useful for plotting the
        evolution of the decision boundary.
        """
        # last element of each sample should be the class
        assert data.shape[1] == self._d + 1
        while True:
            eta = eta0  # / (epoch * 0.01)  # TODO: parameterize hyperparameter?
            if self.epoch(data, eta=eta, callback=epoch_cb):
                return

    def test(self, data: np.ndarray):
        # last element of each sample should be the class
        assert data.shape[1] == self._d + 1

        # shuffle data
        samples = data.copy()
        rand_gen.shuffle(samples)

        for x in samples:
            x, target = np.split(x, [-1])
            # bias trick
            x = np.append(x, 1)

            # threshold
            y_prima = np.dot(self._w.T, x)
            y = self._activation_function(y_prima)
            try:
                assert np.isclose(y, target)
            except AssertionError:
                print(f'Classification error for input {x}: '
                      f'expected {target}, got {y}!')


class DeltaPerceptron(Perceptron):
    @staticmethod
    def _activation_function(y_prima: np.ndarray) -> int:
        return 1 if np.greater(y_prima, 0) else -1

    @staticmethod
    def _error(y: np.ndarray, y_prima: np.ndarray, target: np.ndarray):
        return target - y_prima

from typing import Callable, Tuple

import numpy as np
from numpy.random import default_rng

rand_gen = default_rng()


# noinspection PyAttributeOutsideInit
class Perceptron:
    def __init__(self, dims: int):
        self._d = dims
        self.reset()

    @property
    def weights(self) -> np.ndarray:
        return self._w.copy()

    @property
    def epochs(self) -> int:
        return self._epochs

    def reset(self):
        # initialize weights randomly
        # bias trick, starting theta is 0
        self._w = np.append(rand_gen.standard_normal(self._d), 0)
        self._epochs = 0

    @staticmethod
    def _activation_function(y_prima: float) -> int:
        return 1 if y_prima > 0 else 0

    def _batch_learning(self, X, T, Y_prima, eta) -> Tuple[np.ndarray, int]:
        Y = np.array([self._activation_function(y) for y in Y_prima])
        E = T - Y
        dW = eta * np.dot(X, E)
        return dW, np.count_nonzero(E)

    @staticmethod
    def _prepare_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # shuffle data
        samples = data.copy()
        rand_gen.shuffle(samples)

        # segment data into Matrices
        T = samples[:, -1].astype(int)
        # add a constant 1 to the input for the bias trick
        # X is transposed so every column becomes an input
        X = np.append(samples[:, :-1],
                      np.atleast_2d(np.ones(T.shape[0])).T,
                      axis=1).T

        return X, T

    def _batch_epoch(self,
                     data: np.ndarray,
                     eta: float = 1.0,
                     callback: Callable[[int, np.ndarray, int],
                                        None] = lambda e, w, m: None) \
            -> bool:

        X, T = self._prepare_data(data)
        # Y' = WX
        Y_prima = np.dot(self._w, X)
        dW, misses = self._batch_learning(X, T, Y_prima, eta)

        # update weights
        self._w += dW
        self._epochs += 1

        callback(self._epochs, self._w, misses)
        return misses == 0  # finishing condition

    def _seq_learning(self, x: np.ndarray, t: int, eta: float) -> None:
        y_prima = np.dot(self._w, x).item()
        y = self._activation_function(y_prima)
        e = t - y
        if e == 0:
            return

        self._w += np.dot(eta * e, x)

    def _seq_epoch(self,
                   data: np.ndarray,
                   eta: float = 1.0,
                   callback: Callable[[int, np.ndarray, int],
                                      None] = lambda e, w, m: None) \
            -> bool:

        X, T = self._prepare_data(data)
        X = X.T

        for x, t in zip(X, T):
            self._seq_learning(x, t, eta)

        self._epochs += 1

        # calculate correctly classified samples
        E = T - np.array([self._activation_function(y)
                          for y in np.dot(self._w, X.T)])
        misses = np.count_nonzero(E)
        callback(self._epochs, self._w, misses)

        return misses == 0

    def train(self,
              data: np.ndarray,
              eta0: float = 1.0,
              batch: bool = True,
              epoch_cb: Callable[[int, np.ndarray, int],
                                 None] = lambda e, w, m: None):
        """
        Train this perceptron on the given input data.

        :param data: Data to train on. Each row in this matrix should
        correspond to a training sample, where the last element of the row
        represents the expected class for the sample.

        :param eta0: Starting training rate.

        :param batch: Run in batch mode or not.

        :param epoch_cb: Optional callback executed after each epoch. This
        function will be called with parameters epoch, weight vector,
        dW and the number of classification misses. Useful for plotting the
        evolution of the decision boundary.
        """

        # columns == dimensions + 1 for bias trick
        assert data.shape[1] == self._d + 1

        eta = eta0  # / (epoch * 0.01)  # TODO: parameterize hyperparameter?

        while True:
            if batch:
                if self._batch_epoch(data, eta=eta, callback=epoch_cb):
                    return
            else:
                if self._seq_epoch(data, eta=eta, callback=epoch_cb):
                    return


class DeltaPerceptron(Perceptron):
    @staticmethod
    def _activation_function(y_prima: float) -> int:
        return 1 if y_prima > 0 else -1

    def _batch_learning(self, X, T, Y_prima, eta) -> Tuple[np.ndarray, int]:
        Y = np.array([self._activation_function(y) for y in Y_prima])
        E = T - Y
        dW = eta * np.dot(T - Y_prima, X.T)
        return dW, np.count_nonzero(E)

    def _seq_learning(self, x: np.ndarray, t: int, eta: float):
        y_prima = np.dot(self._w, x).item()
        y = self._activation_function(y_prima)
        if y == t:
            return

        e = t - y_prima
        self._w += np.dot(eta * e, x)

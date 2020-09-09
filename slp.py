from typing import Callable, Tuple

import numpy as np
from numpy.random import default_rng

rand_gen = default_rng()


def prepare_data(data: np.ndarray, bias: bool = True) \
        -> Tuple[np.ndarray, np.ndarray]:
    # shuffle data
    samples = data.copy()
    rand_gen.shuffle(samples)

    # segment data into Matrices
    T = samples[:, -1].astype(int)

    # X is transposed so every column becomes an input
    X = samples[:, :-1].T

    if bias:
        # add a constant 1 to the input for the bias trick
        X = np.append(X, np.ones(shape=(1, X.shape[1])), axis=0)

    return X, T


# noinspection PyAttributeOutsideInit
class Perceptron:
    def __init__(self, dims: int):
        self._d = dims
        self._w = np.empty(0)
        self._epochs = 0

    @property
    def weights(self) -> np.ndarray:
        return self._w.copy()

    @property
    def epochs(self) -> int:
        return self._epochs

    @staticmethod
    def _activation_function(y_prima: float) -> int:
        return 1 if y_prima > 0 else 0

    def _batch_learning(self,
                        X: np.ndarray,
                        T: np.ndarray,
                        eta: float) -> Tuple[bool, float, int]:
        # Y' = WX
        Y_prima = np.dot(self._w, X)
        Y = np.array([self._activation_function(y) for y in Y_prima])
        E = T - Y
        self._w += eta * np.dot(X, E)
        misses = np.count_nonzero(E)
        return misses == 0, np.mean(np.square(E)).item(), misses

    def _seq_learning(self,
                      X: np.ndarray,
                      T: np.ndarray,
                      eta: float) -> Tuple[bool, float, int]:

        for x, t in zip(X.T, T):
            y_prima = np.dot(self._w, x).item()
            y = self._activation_function(y_prima)
            e = t - y
            if e == 0:
                continue
            self._w += np.dot(eta * e, x)

        # calculate correctly classified samples
        E = T - np.array([self._activation_function(y)
                          for y in np.dot(self._w, X)])
        misses = np.count_nonzero(E)
        return misses == 0, np.mean(np.square(E)).item(), misses

    def _epoch(self,
               data: np.ndarray,
               eta: float = 1.0,
               bias: bool = True,
               batch: bool = True,
               callback: Callable[[int, np.ndarray, int, float],
                                  None] = lambda e, w, m, err: None) \
            -> bool:

        X, T = prepare_data(data, bias=bias)
        done, mse, misses = self._batch_learning(X, T, eta) \
            if batch else self._seq_learning(X, T, eta)

        self._epochs += 1
        callback(self._epochs, self._w, misses, mse)
        return done

    def train(self,
              data: np.ndarray,
              eta0: float = 1.0,
              batch: bool = True,
              bias: bool = True,
              epoch_cb: Callable[[int, np.ndarray, int, float],
                                 None] = lambda e, w, m, err: None):
        """
        Train this perceptron on the given input data.

        :param data: Data to train on. Each row in this matrix should
        correspond to a training sample, where the last element of the row
        represents the expected class for the sample.

        :param eta0: Starting training rate.

        :param batch: Run in batch mode or not.

        :param bias: Calculate with bias.

        :param epoch_cb: Optional callback executed after each epoch. This
        function will be called with parameters epoch, weight vector,
        the number of classification misses and the errors. Useful for
        plotting the evolution of the decision boundary.
        """

        # columns == dimensions + 1 for bias trick
        assert data.shape[1] == self._d + 1

        eta = eta0  # / (epoch * 0.01)  # TODO: parameterize hyperparameter?

        # reset
        self._epochs = 0
        self._w = rand_gen.standard_normal(self._d)
        if bias:
            self._w = np.append(self._w, 0)

        while True:
            if self._epoch(data, batch=batch, bias=bias,
                           eta=eta, callback=epoch_cb):
                return


class DeltaPerceptron(Perceptron):
    def __init__(self, dims: int):
        super(DeltaPerceptron, self).__init__(dims=dims)
        self._prev_error = -1
        self._prev_weights = np.empty(0)
        self._prev_misses = 0

    def _check_cond_and_store(self, mse: float, misses: int) \
            -> Tuple[bool, float, int]:
        if misses != 0 or self._prev_error < 0 or self._prev_error >= mse:
            self._prev_error = mse
            self._prev_weights = self._w
            self._prev_misses = misses
            return False, mse, misses
        elif misses == 0 and self._prev_error < mse:
            self._w = self._prev_weights
            return True, self._prev_error, self._prev_misses

    @staticmethod
    def _activation_function(y_prima: float) -> int:
        return 1 if y_prima > 0 else -1

    def _batch_learning(self,
                        X: np.ndarray,
                        T: np.ndarray,
                        eta: float) \
            -> Tuple[bool, float, int]:
        # Y' = WX
        Y_prima = np.dot(self._w, X)
        Y = np.array([self._activation_function(y) for y in Y_prima])
        E = T - Y_prima
        self._w += eta * np.dot(E, X.T)

        # check stop condition
        mse = np.mean(np.square(E)).item()
        misses = np.count_nonzero(T - Y)

        return self._check_cond_and_store(mse, misses)

    def _seq_learning(self,
                      X: np.ndarray,
                      T: np.ndarray,
                      eta: float) \
            -> Tuple[bool, float, int]:
        E = np.zeros(T.shape[0])
        for i, (x, t) in enumerate(zip(X.T, T)):
            y_prima = np.dot(self._w, x).item()
            y = self._activation_function(y_prima)
            e = t - y_prima
            E[i] = e
            if y == t:
                continue
            self._w += np.dot(eta * e, x)

        # calculate correctly classified samples
        M = T - np.array([self._activation_function(y)
                          for y in np.dot(self._w, X)])

        # check stop condition
        mse = np.mean(np.square(E)).item()
        misses = np.count_nonzero(M)
        return self._check_cond_and_store(mse, misses)

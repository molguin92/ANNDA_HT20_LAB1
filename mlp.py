from typing import Any, Callable

import numpy as np
from numpy.random import default_rng

from slp import prepare_data

rand_gen = default_rng()


class TwoLayerPerceptron:
    def __init__(self,
                 dims: int,
                 hidden_nodes: int,
                 mse_threshold: float = 1e-2):
        self._d = dims
        self._hidden_nodes = hidden_nodes

        self._in_W = np.empty(0)
        self._hidden_W = np.empty(0)
        self._epochs = 0
        self._mse_threshold = mse_threshold

    def _phi(self, x: np.ndarray) -> np.ndarray:
        return (2 / (1 + np.exp(-1 * x))) - 1

    def _phi_der(self, x: np.ndarray) -> np.ndarray:
        return np.multiply(1 + self._phi(x), 1 - self._phi(x)) / 2

    def _epoch(self,
               data: np.ndarray,
               eta: float,
               callback: Callable[[int, int, float],
                                  Any] = lambda e, m, mse: None):
        X, T = prepare_data(data)

        # forward pass
        # first layer
        H_star = np.dot(self._in_W, X)
        H = self._phi(H_star)

        # extend H with ones for the bias trick
        H = np.append(H, np.ones(shape=(1, H.shape[1])), axis=0)

        # second layer
        O_star = np.dot(self._hidden_W, H)
        O = self._phi(O_star)

        # first backward pass
        E = O - T
        do = np.multiply(E, self._phi_der(O_star))

        # do will have dims = (1, n_samples)

        # second backward pass
        dh = np.multiply(np.outer(self._hidden_W.T, do),
                         self._phi_der(np.append(H_star,
                                                 np.ones(shape=(1, H.shape[1])),
                                                 axis=0)))
        # remove bias term in dh
        dh = dh[:-1, :]

        # update weights
        self._in_W += -1 * eta * np.dot(dh, X.T)
        self._hidden_W += -1 * eta * np.dot(do, H.T)

        self._epochs += 1

        # calculate error and misses
        mse = np.mean(np.square(E)).item()
        cls = np.array([1 if o > 0 else -1 for o in O])
        misses = np.count_nonzero(T - cls)
        callback(self._epochs, misses, mse)

        return misses == 0

    def train(self, data: np.ndarray, eta: float,
              callback: Callable[[int, int, float],
                                 Any] = lambda e, m, mse: None):
        assert data.shape[1] == self._d + 1
        self._epochs = 0

        # set up weight matrices
        # append a 0 to weights for the bias trick
        self._in_W = rand_gen.standard_normal((self._hidden_nodes, self._d))
        self._in_W = np.append(self._in_W,
                               np.zeros(shape=(self._hidden_nodes, 1)),
                               axis=1)

        self._hidden_W = np.append(
            rand_gen.standard_normal(self._hidden_nodes), 0)

        while not self._epoch(data, eta, callback):
            pass

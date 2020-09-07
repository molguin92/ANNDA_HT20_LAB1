from typing import Callable, Tuple

import numpy as np
from numpy.random import default_rng

rand_gen = default_rng()


class Perceptron:
    def __init__(self, dims: int):
        self._d = dims
        # initialize weights randomly
        # bias trick, starting theta is 0
        self._w = np.append(rand_gen.standard_normal(dims), 0)

    def training_step(self,
                      data: np.ndarray,
                      eta: float = 1.0) -> Tuple[np.ndarray, int]:
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
            target = x[-1]
            # bias trick
            x = np.append(x[:-1], 1)

            # threshold
            y = 1 if np.dot(self._w, x) > 0 else 0

            # learning step
            if not np.isclose(y, target):
                misses += 1
                error = target - y
                etaX = (error * eta) * x
                dw = np.add(dw, etaX)

        # finally, add dw to the weight vector
        self._w = np.add(self._w, dw)

        # return the dw and misses to determine when to stop training
        return dw, misses

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

        epoch = 0
        while True:
            epoch += 1
            eta = eta0  # / (epoch * 0.01)  # TODO: parameterize hyperparameter?
            dw, misses = self.training_step(data, eta=eta)
            epoch_cb(epoch, self._w, dw, misses)

            if misses == 0:
                return

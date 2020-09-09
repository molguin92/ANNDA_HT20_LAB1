import numpy as np
import pandas as pd


class Mackey_Glass:
    # implementation with memoization for efficiency

    def __init__(self,
                 gamma: float = 0.1,
                 beta: float = 0.2,
                 n: float = 10,
                 tau: int = 25):
        self._gamma = gamma
        self._beta = beta
        self._n = n
        self._tau = tau

        self._lookup_table = {}

    def _lookup_or_calculate(self, t: int) -> float:
        try:
            return self._lookup_table[t]
        except KeyError:
            x = self.fn(t)
            self._lookup_table[t] = x
            return x

    def fn(self, t: int) -> float:
        if t == 0:
            return 1.5
        elif t < 0:
            return 0
        else:
            try:
                return self._lookup_table[t]
            except KeyError:
                prev_x = self._lookup_or_calculate(t - 1)
                delayed_x = self._lookup_or_calculate(t - self._tau)

                upper = self._beta * delayed_x
                lower = 1 + np.power(delayed_x, self._n)
                x = prev_x + (upper / lower) - (self._gamma * prev_x)

                self._lookup_table[t] = x
                return x


if __name__ == '__main__':
    mackey_glass = Mackey_Glass()

    t_range = range(301, 1500 + 1)
    data = np.array([[mackey_glass.fn(t + d)
                      for d in [-20, -15, -10, -5, 0, 5]]
                     for t in t_range])
    pd.DataFrame(data, columns=[f'x{i}' for i in range(5)] + ['out']) \
        .to_csv('data/mackey_glass.csv', index=False)

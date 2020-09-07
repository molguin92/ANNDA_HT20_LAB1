from typing import Iterable

import pandas as pd
from numpy.random import default_rng

_pair = Iterable[float]
rand_gen = default_rng()
n_per_cls = 100


def gen_data(mean: _pair,
             cov: Iterable[_pair],
             n=n_per_cls) -> pd.DataFrame:
    return pd.DataFrame(data=rand_gen.multivariate_normal(mean=mean,
                                                          cov=cov,
                                                          size=n),
                        columns=['x1', 'x2'])


if __name__ == '__main__':
    clsA = gen_data(mean=[3, 3], cov=[[10, 1], [1, 0.7]])
    clsB = gen_data(mean=[-3, -3], cov=[[10, 1], [1, 0.7]])

    # put all the samples in one DataFrame
    clsA['cls'] = 'A'
    clsB['cls'] = 'B'

    data = pd.concat((clsA, clsB))
    data['cls'] = data['cls'].astype('category')

    # save data
    data.to_csv('./data/samples.csv', index=False)

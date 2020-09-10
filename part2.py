# preamble, load all required packages and setup some stuff
import itertools
import warnings
from typing import Any, Dict, Tuple

import multiprocess as mproc
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from mackey_glass import Mackey_Glass

_rand_gen = default_rng()


def generate_mackey_glass_series(start: int = 301, end: int = 1500 + 1) \
        -> pd.DataFrame:
    # generate data
    mackey_glass = Mackey_Glass()
    t_range = range(start, end)

    offsets = [-20, -15, -10, -5, 0, 5]
    data = np.array([[mackey_glass.fn(t + o)
                      for o in offsets]
                     for t in t_range])

    columns = [f'X(t{o})' if o < 0 else f'X(t+{o})' for o in offsets]
    df = pd.DataFrame(data, columns=columns)
    df['t'] = list(t_range)

    return df


def gen_noisy_data(std: float, X: np.ndarray, Y: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    X_noisy = X + _rand_gen.normal(loc=0.0, scale=std, size=X.shape)
    Y_noisy = Y + _rand_gen.normal(loc=0.0, scale=std, size=Y.shape)

    # save the noisy data for posterior usage
    np.savez('data/noisy_{std}.npz', X=X_noisy, Y=Y_noisy)

    return X_noisy, Y_noisy


def test_config(mlp_params: Dict,
                data: Tuple[np.ndarray, np.ndarray],
                metadata: Dict[str, Any] = {},
                n_iter: int = 100) -> pd.DataFrame:
    """
    Tests an MLP with the specified parameters on the specified data. N
    iterations will be run, each on a diffferent -- random -- test/training
    split of the data.

    :param mlp_params: Parameters to build the MLP with.
    :param data: A tuple containing (inputs, outputs) for training and
    validation.
    :param metadata: Metadata to be included in the output DataFrame. Each
    key will become a column, and the values will be assigned to all the rows
    in the output.
    :param n_iter: The number of iterations to run.
    :return: A DataFrame with the results.
    """
    X, Y = data

    mse = np.zeros(n_iter)
    mlp = MLPRegressor()
    mlp.set_params(**mlp_params)

    with warnings.catch_warnings():
        # ignore some annoying warnings
        warnings.simplefilter("ignore")

        for i in range(n_iter):
            # get a random train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, Y)

            # train the estimator
            mlp.fit(X_train, y_train)

            # get the MSE for the testing data
            # store the result
            y_pred = mlp.predict(X_test)
            mse[i] = mean_squared_error(y_test, y_pred)

    result = pd.DataFrame({'mse': mse})
    for layer, nodes in enumerate(mlp_params['hidden_layer_sizes']):
        result[f'layer{layer}_nodes'] = nodes
    result['alpha'] = mlp_params['alpha']
    for k, v in metadata.items():
        result[k] = v

    print(f'Processed MLP with params {mlp_params} and metadata {metadata}.')
    return result


# generate MLP configurations
def gen_mlp_config(base_params: Dict[str, Any],
                   layers: Tuple[int, ...],
                   alpha: float):
    params = base_params.copy()
    params['hidden_layer_sizes'] = layers
    params['alpha'] = alpha
    return params


if __name__ == '__main__':
    # generate mackey_glass_series, store it for plotting later
    mg_df = generate_mackey_glass_series()
    mg_df.to_csv('data/mackey_glass.csv', index=False)

    # turn Mackey-Glass DataFrame into numpy arrays
    # scikit-learn uses input matrices with rows=samples, columns=features
    # last column is the input t, we don't want that
    X = mg_df.to_numpy()[:, :-2]
    Y = mg_df.to_numpy()[:, -2]

    # ranges of our hyperparameters
    base_params = {'early_stopping': True}
    h_layer_nodes = range(3, 9)
    alphas = 10.0 ** -np.arange(1, 7)

    # test 2-layer perceptrons
    two_mlp_layers = [(i,) for i in h_layer_nodes]
    two_mlp_configs = [gen_mlp_config(params, layers, alpha)
                       for params, layers, alpha in
                       itertools.product([base_params], two_mlp_layers, alphas)]

    # evaluate 2-layer perceptrons in parallel
    # note: this will take some time
    with mproc.Pool() as pool:
        configs = list(itertools.product(two_mlp_configs, [(X, Y)]))
        print(f'Evaluating {len(configs)} two-layer perceptrons...')
        results = pd.concat(pool.starmap(test_config, configs))
        print('Done!')

    # store the results for future use
    results.to_csv('data/two_layer_perceptrons_scores.csv', index=False)

    # choose the best 2-layer perceptron as base for our three-layer
    mse_means = results.groupby(['alpha', 'layer0_nodes']).mean()
    alpha, layer0_nodes = mse_means['mse'].idxmin()

    print('Best two-layer perceptron found: '
          f'alpha {alpha}, hidden nodes: {layer0_nodes}')

    # prepare the noisy data
    stddevs = [0.03, 0.09, 0.18]
    data_mdata = [(gen_noisy_data(s, X, Y), {'std': s}) for s in stddevs]

    # setup parameters for our 3-layer perceptrons
    three_mlp_layers = [(layer0_nodes, i) for i in h_layer_nodes]
    three_mlp_configs = [gen_mlp_config(params, layers, alpha)
                         for params, layers, alpha in
                         itertools.product([base_params],
                                           three_mlp_layers,
                                           alphas)]

    # evaluate all our three-layer configs
    with mproc.Pool() as pool:
        configs = [(mlp_params, data, mdata)
                   for mlp_params, (data, mdata) in
                   itertools.product(three_mlp_configs, data_mdata)]
        print(f'Evaluating {len(configs)} three-layer perceptrons...')
        results = pd.concat(pool.starmap(test_config, configs))
        print('Done!')

    # finally, store the results for future use
    results.to_csv('data/three_layer_perceptrons_scores.csv', index=False)

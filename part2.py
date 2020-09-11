# preamble, load all required packages and setup some stuff
import json
import time
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
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
    np.savez(f'data/noisy_{std}.npz', X=X_noisy, Y=Y_noisy)

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
    train_times = np.zeros(n_iter)
    mlp = MLPRegressor()
    mlp.set_params(**mlp_params)

    with warnings.catch_warnings():
        # ignore some annoying warnings
        warnings.simplefilter("ignore")

        for i in range(n_iter):
            # get a random train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, Y)

            # train the estimator
            # time this for analysis
            ti = time.monotonic()
            mlp.fit(X_train, y_train)
            dt = time.monotonic() - ti

            # get the MSE for the testing data
            # store the result
            y_pred = mlp.predict(X_test)
            mse[i] = mean_squared_error(y_test, y_pred)
            train_times[i] = dt

    result = pd.DataFrame({'mse': mse, 'training_time': train_times})
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
    mlp_2l_param_grid = dict(
        **{k: [v] for k, v in base_params.items()},
        hidden_layer_sizes=[(i,) for i in h_layer_nodes],
        alpha=alphas
    )

    # evaluate 2-layer perceptrons in parallel
    # note: this will take some time
    mlp_2l_gs = GridSearchCV(MLPRegressor(),
                             mlp_2l_param_grid,
                             scoring='neg_mean_squared_error',
                             n_jobs=-1,
                             return_train_score=True,
                             verbose=True)
    mlp_2l_gs.fit(X, Y)

    results_2layers = pd.DataFrame(mlp_2l_gs.cv_results_)
    results_2layers.to_csv('data/two_layer_perceptrons_scores.csv', index=False)

    # choose the best 2-layer perceptron as base for our 3-layer
    # best_2l_mlp = mlp_2l_gs.best_estimator_
    mlp_2l_params = mlp_2l_gs.best_params_.copy()
    alpha_2layers = mlp_2l_params['alpha']
    layer0_nodes = mlp_2l_params['hidden_layer_sizes'][0]
    print('Best two-layer perceptron found: '
          f'alpha {alpha_2layers}, hidden nodes: {layer0_nodes}')

    # store the best two-layer perceptron params as a json for future use
    with open('data/best_2l_mlp.json', 'w') as fp:
        json.dump(mlp_2l_params, fp)

    # prepare the noisy data
    stddevs = [0.03, 0.09, 0.18]
    noisy_data = [gen_noisy_data(s, X, Y) for s in stddevs]

    # test 3-layer perceptrons
    mlp_3l_param_grid = dict(
        **{k: [v] for k, v in base_params.items()},
        hidden_layer_sizes=[(layer0_nodes, i) for i in h_layer_nodes],
        alpha=alphas
    )

    gs = GridSearchCV(MLPRegressor(),
                      mlp_3l_param_grid,
                      scoring='neg_mean_squared_error',
                      n_jobs=-1,
                      return_train_score=True,
                      verbose=True)

    results_3layers = []
    mlp_3l_params_per_std = {}
    for std, (X_noisy, Y_noisy) in zip(stddevs, noisy_data):
        # evaluate 3-layer perceptrons in parallel
        # note: this will take some time
        gs.fit(X_noisy, Y_noisy)
        results = pd.DataFrame(gs.cv_results_)
        results['noise_std'] = std
        results_3layers.append(results)
        mlp_3l_params = gs.best_params_.copy()
        mlp_3l_params_per_std[std] = mlp_3l_params
        alpha_3layers = mlp_3l_params['alpha']
        layer1_nodes = mlp_3l_params['hidden_layer_sizes'][1]

        print(f'Best three-layer perceptron found '
              f'(for noisy data with std {std}): '
              f'alpha {alpha_2layers}, '
              f'hidden nodes in second layer: {layer1_nodes}')

        # store the best three-layer perceptrons params as a jsons for future
        # use
        with open(f'data/best_3l_mlp_std{std}.json', 'w') as fp:
            json.dump(mlp_3l_params, fp)

    results_3layers = pd.concat(results_3layers)
    results_3layers.to_csv('data/three_layer_perceptrons_scores.csv',
                           index=False)

    # compare 2- and 3-layer estimators on noisy data

    param_grid_2v3 = [
        {k: [v] for k, v in mlp_2l_params.items()},
        {k: [v] for k, v in mlp_3l_params_per_std[0.09].items()},
    ]

    gs = GridSearchCV(MLPRegressor(),
                      param_grid_2v3,
                      scoring='neg_mean_squared_error',
                      n_jobs=-1,
                      return_train_score=True,
                      verbose=True)

    gs.fit(*noisy_data[1])  # [1] is the index for std=0.09
    results_2v3_df = pd.DataFrame(gs.cv_results_)
    # store comparison data
    results_2v3_df.to_csv('data/2v3_noisy.0.09.csv', index=False)
    # done!

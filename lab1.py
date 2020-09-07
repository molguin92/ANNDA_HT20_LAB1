# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# preamble, load all required packages and setup some stuff
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context='notebook', palette='tab10')  # for plots

# +
# load and visualize dataset
data = pd.read_csv('data/samples.csv')

fig, ax = plt.subplots()
sns.scatterplot(x='x1', y='x2', hue=data['cls'].tolist(), data=data, ax=ax)
plt.show()

# +
# import and instantiate Perceptron implementation
from perceptron import Perceptron

p = Perceptron(dims=2)

# prepare data for training
t_data = data.copy()
t_data['cls'] = t_data['cls'].apply(lambda c: 0 if c == 'A' else 1).astype('category')

# plotting callback
pad = 10
min_x = data['x1'].min() - pad
max_x = data['x1'].max() + pad
min_y = data['x2'].min() - pad
max_y = data['x2'].max() + pad


def plot(epoch, weights, dW, misses):
    # plot if boundary is done:
    if misses == 0:
        # get boundary function from weights and theta
        # theta is the last element of the weight vector
        # 0 = w1x1 + w2x2 + w3
        # -w2x2 = w1x1 + w3
        # x2 = -(w1x1 + w3) / w2
        # x2 = -(w1x1 + w3) / w2
        fig, ax = plt.subplots()
        sns.scatterplot(x='x1', y='x2', hue=data['cls'].tolist(), data=data, ax=ax)
        
        x1 = np.array([min_x, max_x])
        x2 = (-1 * ((weights[0] * x1) + weights[2])) / weights[1]

        if len(ax.lines) > 0:
            ax.lines[0].remove()

        ax.plot(x1, x2, 'r-', label='Decision boundary')
        ax.set_title(f'Epoch {epoch}')
        ax.set_ylim(min_y, max_y)
        ax.set_xlim(min_x, max_x)
        ax.legend()
        plt.plot()
#     print(f'W: {weights}, dW: {dW}, missclassified: {misses}')

p.train(t_data.to_numpy(), eta0=1, epoch_cb=plot)
p.test(t_data.to_numpy())
# +
# delta-rule Perceptron
from perceptron import DeltaPerceptron

dp = DeltaPerceptron(dims=2)
# prepare data for training
# delta rule uses symmetric target values
t_data = data.copy()
t_data['cls'] = t_data['cls'].apply(lambda c: -1 if c == 'A' else 1).astype('category')

dp.train(t_data.to_numpy(), eta0=1e-5, epoch_cb=plot)
dp.test(t_data.to_numpy())


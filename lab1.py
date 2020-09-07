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
# %matplotlib
from perceptron import Perceptron

p = Perceptron(dims=2)

# prepare data for training
data['cls'] = data['cls'].apply(lambda c: 0 if c == 'A' else 1).astype('category')

# plotting callback
min_x = data['x1'].min()
max_x = data['x1'].max()
min_y = data['x2'].min()
max_y = data['x2'].max()
fig, ax = plt.subplots()

def plot(epoch, weights, dW, misses):
    # get boundary function from weights and theta
    # theta is the last element of the weight vector
    # 0 = w1x1 + w2x2 + theta
    # x2 = (w1x1 - theta) / w2
    x1 = np.array([min_x, max_x])
    x2 = np.divide(np.add(np.dot(weights[0], x1), weights[2]), weights[1])
    
    ax.clear()
    ax.plot(x1, x2, 'r-', label='Decision boundary')
    sns.scatterplot(x='x1', y='x2', hue=data['cls'].tolist(), data=data, ax=ax)
    ax.set_title(f'Epoch {epoch}')
    ax.set_ylim(min_y, max_y)
    ax.set_xlim(min_x, max_x)
    plt.pause(0.01)
    print(f'W: {weights}, dW: {dW}, missclassified: {misses}')
    
p.train(data.to_numpy(), eta0=1, epoch_cb=plot)
input()
# %matplotlib inline
# -



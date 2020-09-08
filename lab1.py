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
import pylab
import seaborn as sns
sns.set(context='notebook', palette='Dark2', font_scale=.75)  # for plots
pylab.rcParams['figure.dpi'] = 150

# +
# load and visualize dataset
data = pd.read_csv('data/samples.csv')

fig, ax = plt.subplots()
sns.scatterplot(x='x1', y='x2', hue=data['cls'].tolist(), data=data, ax=ax)
plt.show()

# +
# import and instantiate Perceptron implementations
from perceptron import Perceptron, DeltaPerceptron

# plotting function for boundaries
def plot_decision_boundary(lx, rx, weights, ax, style, label):
    x1 = np.array([lx, rx])
    x2 = (-1 * ((weights[0] * x1) + weights[2])) / weights[1]
    ax.plot(x1, x2, style, label=label)

# perceptrons
p = Perceptron(dims=2)
dp = DeltaPerceptron(dims=2)
p_seq = Perceptron(dims=2)
dp_seq = DeltaPerceptron(dims=2)

# prepare data for training perceptrons
p_data = data.copy()
p_data['cls'] = p_data['cls'].apply(lambda c: 0 if c == 'A' else 1).astype('category')

# prepare data for training deltaperceptrons
# delta rule uses symmetric target values
dp_data = data.copy()
dp_data['cls'] = dp_data['cls'].apply(lambda c: -1 if c == 'A' else 1).astype('category')

# callback functions for epochs, to store accuracy and plot it
p_acc = [0]
dp_acc = [0]
p_seq_acc = [0]
dp_seq_acc = [0]
n = data.shape[0]

# train (batch)
eta0 = 1e-5
p.train(p_data.to_numpy(), eta0=eta0, epoch_cb=lambda e, w, m: p_acc.append((n - m) / n))
dp.train(dp_data.to_numpy(), eta0=eta0, epoch_cb=lambda e, w, m: dp_acc.append((n - m) / n))
p_seq.train(p_data.to_numpy(), eta0=eta0, batch=False, epoch_cb=lambda e, w, m: p_seq_acc.append((n - m) / n))
dp_seq.train(dp_data.to_numpy(), eta0=10, batch=False, epoch_cb=lambda e, w, m: dp_seq_acc.append((n - m) / n))

# plotting
pad = 10
min_x = data['x1'].min() - pad
max_x = data['x1'].max() + pad
min_y = data['x2'].min() - pad
max_y = data['x2'].max() + pad

# fig, (ax0, ax1) = plt.subplots(ncols=1, nrows=2)

fig, ax0 = plt.subplots()
ax0.set_title('Decision boundaries')
sns.scatterplot(x='x1', y='x2', hue=data['cls'].tolist(), data=data, ax=ax0)
plot_decision_boundary(min_x, max_x, p.weights, ax0, 'r-', 'Perceptron learning (Batch)')
plot_decision_boundary(min_x, max_x, dp.weights, ax0, 'g-', 'Delta rule (Batch)')
plot_decision_boundary(min_x, max_x, p_seq.weights, ax0, 'r--', 'Perceptron learning (Sequential)')
plot_decision_boundary(min_x, max_x, dp_seq.weights, ax0, 'g--', 'Delta rule (Sequential)')
ax0.legend()
plt.show()

fig, ax1 = plt.subplots()
ax1.set_title('Accuracy over time (epochs)')
ax1.plot(range(0, p.epochs + 1), p_acc, 'r-', label='Perceptron learning (Batch)')
ax1.plot(range(0, dp.epochs + 1), dp_acc, 'g-', label='Delta rule (Batch)')
ax1.plot(range(0, p_seq.epochs + 1), p_seq_acc, 'r--', label='Perceptron learning (Sequential)')
ax1.plot(range(0, dp_seq.epochs + 1), dp_seq_acc, 'g--', label='Delta rule (Sequential)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.set_xscale('symlog')
plt.show()

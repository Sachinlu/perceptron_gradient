""" perceptron with stochastic gradient descent """
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Loading Datasets
X, y = make_moons(n_samples=200, noise=0.1)

# Normalizing datasets
ml = len(X)
for i in range(ml):
    if y[i] == 0:
        X[i] = X[i] + 0.7

# Plotting the Data before Stochastic Gradient Descent
df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
colors = {0: "green", 1: "grey"}
fig, ax = plt.subplots()
grouped = df.groupby("label")
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

#  In a threshold unit, there are only two possible output values: −1 and +1
y = np.where(y == 0, -1, y)

# Applying Stochastic Gradient Descent
bias = 1
learning_rate = 0.1
steps = 0
weight = np.ones((X.shape[1], 1))
check = True
while check:
    sample_misclassified = 0
    for value in range(X.shape[0]):
        N_X = X[value,]
        n_y = y[value]

        if n_y * (np.dot(weight.T, N_X.T) + bias) < 0:
            bias = bias + learning_rate * n_y
            weight = weight + learning_rate * np.dot(N_X, n_y).reshape(2, 1)
            sample_misclassified = sample_misclassified + 1

    if sample_misclassified != 0:
        check = True
    else:
        check = False
    steps = steps + 1
    print(f"epoch-->{steps},No of misclassified sample-->{sample_misclassified}")

print(f"updated_weight: {weight}")
print(f"updated_bias: {bias}")

# Applying perceptron (straight line equation) and plotting the Data after Stochastic Gradient Descent
value_x = np.linspace(-0.5, 2.5, 10)
value_y = -(weight[0] * value_x + bias)/ weight[1]
df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
colors = {-1: "green", 1: "grey"}
fig, ax = plt.subplots()
grouped = df.groupby("label")
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.plot(value_x, value_y)
plt.show()
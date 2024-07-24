import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *

# Loading Data
np.random.seed(0)  # For reproducibility
n_train = 50000
n_test = 10000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

# Training configuration
epochs = 30
batch_size = 100
learning_rate = 0.1

# Network configuration
layer_dims = [784, 40, 10]
net = Network(layer_dims)
net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

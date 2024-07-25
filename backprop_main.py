import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *


def load_data():
    # Loading Data
    np.random.seed(0)  # For reproducibility
    n_train = 50000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    return x_train, y_train, x_test, y_test


def b():
    x_train, y_train, x_test, y_test = load_data()

    # Training configuration
    epochs = 30
    batch_size = 10
    learning_rate = [0.001, 0.01, 0.1, 1, 10]

    # Network configuration
    layer_dims = [784, 40, 10]
    net = Network(layer_dims)

    results = {}
    for lr in learning_rate:
        print("training network with learning rate {}".format(lr))
        _, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(x_train,
                                                                                          y_train,
                                                                                          epochs,
                                                                                          batch_size,
                                                                                          lr,
                                                                                          x_test=x_test,
                                                                                          y_test=y_test)

        results[lr] = {
            'epochs': list(range(len(epoch_train_cost))),
            'train_cost': epoch_train_cost,
            'train_acc': epoch_train_acc,
            'test_cost': epoch_test_cost,
            'test_acc': epoch_test_acc
        }

    # Plot Training Loss
    plot_metrics(results, 'train_cost', 'Training Loss Across Epochs', 'Training Loss')

    # Plot Training Accuracy
    plot_metrics(results, 'train_acc', 'Training Accuracy Across Epochs', 'Training Accuracy')

    # Plot Test Accuracy
    plot_metrics(results, 'test_acc', 'Test Accuracy Across Epochs', 'Test Accuracy')


def c():
    x_train, y_train, x_test, y_test = load_data()

    epochs = 30
    batch_size = 10
    learning_rate = 0.1

    # Network configuration
    layer_dims = [784, 40, 10]
    net = Network(layer_dims)

    results = {}
    _, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(x_train,
                                                                                      y_train,
                                                                                      epochs,
                                                                                      batch_size,
                                                                                      learning_rate,
                                                                                      x_test=x_test,
                                                                                      y_test=y_test)

    results[learning_rate] = {
        'epochs': list(range(len(epoch_train_cost))),
        'train_cost': epoch_train_cost,
        'train_acc': epoch_train_acc,
        'test_cost': epoch_test_cost,
        'test_acc': epoch_test_acc
    }

    # Plot Test Accuracy
    plot_metrics(results, 'test_acc', 'Test Accuracy Across Epochs', 'Test Accuracy')


def d():
    x_train, y_train, x_test, y_test = load_data()

    epochs = 30
    batch_size = 10
    learning_rate = 0.1

    # Network configuration
    layer_dims = [784, 10]
    net = Network(layer_dims)

    results = {}
    params, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(x_train,
                                                                                      y_train,
                                                                                      epochs,
                                                                                      batch_size,
                                                                                      learning_rate,
                                                                                      x_test=x_test,
                                                                                      y_test=y_test)

    results[learning_rate] = {
        'epochs': list(range(len(epoch_train_cost))),
        'train_cost': epoch_train_cost,
        'train_acc': epoch_train_acc,
        'test_cost': epoch_test_cost,
        'test_acc': epoch_test_acc
    }

    # Plot Training Accuracy
    plot_metrics(results, 'train_acc', 'Training Accuracy Across Epochs', 'Training Accuracy')

    # Plot Test Accuracy
    plot_metrics(results, 'test_acc', 'Test Accuracy Across Epochs', 'Test Accuracy')

    plot_weights(params['W1'])


def e():
    x_train, y_train, x_test, y_test = load_data()

    epochs = 50000
    batch_size = 16
    learning_rate = 0.1

    # Network configuration
    layer_dims = [784, 40, 10]
    net = Network(layer_dims)

    results = {}
    params, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(x_train,
                                                                                      y_train,
                                                                                      epochs,
                                                                                      batch_size,
                                                                                      learning_rate,
                                                                                      x_test=x_test,
                                                                                      y_test=y_test)

    results[learning_rate] = {
        'epochs': list(range(len(epoch_train_cost))),
        'train_cost': epoch_train_cost,
        'train_acc': epoch_train_acc,
        'test_cost': epoch_test_cost,
        'test_acc': epoch_test_acc
    }

    # Plot Training Accuracy
    plot_metrics(results, 'train_acc', 'Training Accuracy Across Epochs', 'Training Accuracy')

    # Plot Test Accuracy
    plot_metrics(results, 'test_acc', 'Test Accuracy Across Epochs', 'Test Accuracy')


def plot_metrics(results, metric, title, ylabel):
    plt.figure(figsize=(12, 8))
    for lr, data in results.items():
        plt.plot(data['epochs'], data[metric], label=f'LR={lr}')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_weights(weights):
    plt.figure(figsize=(10, 8))
    for i in range(weights.shape[0]):
        plt.subplot(2, 5, i + 1)  # 2 rows, 5 columns
        plt.imshow(weights[i].reshape(28, 28), interpolation='nearest', cmap='gray')
        plt.title(f'Class {i}')
        plt.axis('off')  # Hide the axis
    plt.show()


if __name__ == '__main__':
    b()
    c()
    d()
    e()

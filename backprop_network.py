from scipy.special import softmax, logsumexp
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        is [784, 40, 10] then it would be a three-layer network, with the
        first layer (the input layer) containing 784 neurons, the second layer 40 neurons,
        and the third layer (the output layer) 10 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution centered around 0.
        """
        self.num_layers = len(sizes) - 1
        self.sizes = sizes
        self.parameters = {}
        for l in range(1, len(sizes)):
            self.parameters['W' + str(l)] = np.random.randn(sizes[l], sizes[l - 1]) * np.sqrt(2. / sizes[l - 1])
            self.parameters['b' + str(l)] = np.zeros((sizes[l], 1))

    @staticmethod
    def relu(x):
        """
            Description: Implements forward propagation for a relu unit.

            Input:
            x – the linear component of the activation function.

            Output:
            activation – the activation of the layer.
        """
        activation = np.maximum(0, x)

        return activation

    def relu_derivative(self, x):
        """TODO: Implement the derivative of the relu function."""
        raise NotImplementedError

    def cross_entropy_loss(self, logits, y_true):
        m = y_true.shape[0]
        # Compute log-sum-exp across each column for normalization
        log_probs = logits - logsumexp(logits, axis=0)
        y_one_hot = np.eye(10)[y_true].T  # Assuming 10 classes
        # Compute the cross-entropy loss
        loss = -np.sum(y_one_hot * log_probs) / m
        return loss

    def cross_entropy_derivative(self, logits, y_true):
        """ Input: "logits": numpy array of shape (10, batch_size) where each column is the network output on the given example (before softmax)
                    "y_true": numpy array of shape (batch_size,) containing the true labels of the batch
            Returns: a numpy array of shape (10,batch_size) where each column is the gradient of the loss with respect to y_pred (the output of the network before the softmax layer) for the given example.
        """
        # TODO: Implement
        raise NotImplementedError

    def forward_propagation(self, X):
        """
        Implement the forward step of the backpropagation algorithm.
            Input: "X" - numpy array of shape (784, batch_size) - the input to the network
            Returns: "ZL" - numpy array of shape (10, batch_size),
                     the output of the network on the input X (before the softmax layer)
                    "forward_outputs" - A list of length self.num_layers,
                     containing the forward computation (parameters & output of each layer).
        """

        ZL = None
        forward_outputs = []

        activation = X

        # linear forward propagation with ReLU for layers 1 to (num_layers - 1)
        for layer in range(1, self.num_layers):
            prev_activation = activation
            activation, cache = self.linear_activation_forward(prev_activation=prev_activation,
                                                               w=self.parameters['W' + str(layer)],
                                                               b=self.parameters['b' + str(layer)],
                                                               activation_function="relu",
                                                               layer=layer)

            forward_outputs.append(cache)

        # linear forward propagation with Softmax for final layer
        last_layer = self.num_layers
        last_activation, cache = self.linear_activation_forward(prev_activation=activation,
                                                                w=self.parameters[f'W{last_layer}'],
                                                                b=self.parameters[f'b{last_layer}'],
                                                                activation_function="softmax",
                                                                layer=last_layer)

        ZL = last_activation
        forward_outputs.append(cache)

        return ZL, forward_outputs

    @staticmethod
    def linear_forward(activation, w: np.array, b: np.array, layer: int):
        """
        Description: Implement the linear part of a layer's forward propagation.

        Input:
        activation – the activations of the previous layer.
        w – the weight matrix of the current layer (of shape [size of current layer, size of previous layer]).
        B – the bias vector of the current layer (of shape [size of current layer, 1]).

        Output:
        Z – the linear component of the activations function (i.e., the value before applying the non-linear function).
        cache – a dictionary containing activation, w, b (stored for making the backpropagation easier to compute).
        """
        cache = {}
        z = np.dot(w, activation) + b
        cache[f'x{layer}'] = activation
        cache[f'W{layer}'] = w
        cache[f'b{layer}'] = b

        return z, cache

    def linear_activation_forward(self, prev_activation: np.ndarray,
                                  w: np.ndarray,
                                  b: np.ndarray,
                                  activation_function: str,
                                  layer: int) -> tuple[np.ndarray, dict]:
        """
        Description: Implement the forward propagation for the activation function decision.

        Input:
        prev_activation – activation of the previous layer.
        w – the weights matrix of the current layer.
        B – the bias vector of the current layer.
        activation_function – the activation function to be used (either “sigmoid” or “relu”).

        Output:
        activation – the activation of the current layer
        cache – a joint dictionary containing both cache and activation_cache
        """
        # calculate the linear part.
        z, cache = self.linear_forward(activation=prev_activation, w=w, b=b, layer=layer)

        if activation_function.__eq__("softmax"):
            activation = softmax(z)
            activation_cache = z
            cache[f'z{layer}'] = activation_cache

            return activation, cache

        elif activation_function.__eq__("relu"):
            activation = self.relu(z)
            activation_cache = z
            cache[f'z{layer}'] = activation_cache

            return activation, cache

        else:
            raise Exception('Non-supported activation function')

    def backpropagation(self, ZL, Y, forward_outputs):
        """Implement the backward step of the backpropagation algorithm.
            Input: "ZL" -  numpy array of shape (10, batch_size), the output of the network on the input X (before the softmax layer)
                    "Y" - numpy array of shape (batch_size,) containing the labels of each example in the current batch.
                    "forward_outputs" - list of length self.num_layers given by the output of the forward function
            Returns: "grads" - dictionary containing the gradients of the loss with respect to the network parameters across the batch.
                                grads["dW" + str(l)] is a numpy array of shape (sizes[l], sizes[l-1]),
                                grads["db" + str(l)] is a numpy array of shape (sizes[l],1).
        
        """
        grads = {}

        # TODO: Implement the backward function
        raise NotImplementedError
        return grads

    def sgd_step(self, grads, learning_rate):
        """
        Updates the network parameters via SGD with the given gradients and learning rate.
        """
        parameters = self.parameters
        L = self.num_layers
        for l in range(L):
            parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
        return parameters

    def train(self, x_train, y_train, epochs, batch_size, learning_rate, x_test, y_test):
        epoch_train_cost = []
        epoch_test_cost = []
        epoch_train_acc = []
        epoch_test_acc = []

        for epoch in range(epochs):
            costs = []
            acc = []
            for i in range(0, x_train.shape[1], batch_size):
                X_batch = x_train[:, i:i + batch_size]
                Y_batch = y_train[i:i + batch_size]

                ZL, caches = self.forward_propagation(X_batch)
                cost = self.cross_entropy_loss(ZL, Y_batch)
                costs.append(cost)
                grads = self.backpropagation(ZL, Y_batch, caches)

                self.parameters = self.sgd_step(grads, learning_rate)

                preds = np.argmax(ZL, axis=0)
                train_acc = self.calculate_accuracy(preds, Y_batch, batch_size)
                acc.append(train_acc)

            average_train_cost = np.mean(costs)
            average_train_acc = np.mean(acc)
            print(
                f"Epoch: {epoch + 1}, Training loss: {average_train_cost:.20f}, Training accuracy: {average_train_acc:.20f}")

            epoch_train_cost.append(average_train_cost)
            epoch_train_acc.append(average_train_acc)

            # Evaluate test error
            ZL, caches = self.forward_propagation(x_test)
            test_cost = self.cross_entropy_loss(ZL, y_test)
            preds = np.argmax(ZL, axis=0)
            test_acc = self.calculate_accuracy(preds, y_test, len(y_test))
            # print(f"Epoch: {epoch + 1}, Test loss: {test_cost:.20f}, Test accuracy: {test_acc:.20f}")

            epoch_test_cost.append(test_cost)
            epoch_test_acc.append(test_acc)

        return self.parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc

    @staticmethod
    def calculate_accuracy(y_pred, y_true, batch_size):
        """Returns the average accuracy of the prediction over the batch """
        return np.sum(y_pred == y_true) / batch_size

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
    def relu(z: np.ndarray) -> np.ndarray:
        """
        Description: Implements forward propagation for a relu unit.

        Input:
        z – the linear component of the activation function.

        Output:
        activation – the activation of the layer.
        """
        activation = np.maximum(0, z)

        return activation

    def relu_derivative(self, d_a: np.ndarray, activation_cache: np.ndarray) -> np.ndarray:
        """
        Description: Implements backward propagation for a ReLU unit.

        Input:
        d_a – the post-activation gradient.
        activation_cache – contains z (stored during the forward propagation).

        Output:
        d_z – gradient of the cost with respect to Z.
        """

        z = activation_cache
        a = self.relu(z=z)
        d_z = np.multiply(d_a, np.int64(a > 0))

        return d_z

    @staticmethod
    def softmax_backward(d_a: np.ndarray, activation_cache: np.ndarray) -> np.ndarray:
        """
        Description: Implements backward propagation for a Softmax unit.

        Inputs:
        d_a - the post-activation gradient.
        cache - input Z stored during forward propagation

        Output:
        d_z - gradient of the cost with respect to z
        """

        x = activation_cache
        p = softmax(x=x)
        d_z = d_a * p * (1 - p)

        return d_z

    @staticmethod
    def cross_entropy_loss(logits, y_true):
        """
        Description: Implements cross entropy loss.

        Input:
        logits: numpy array of shape (10, batch_size) where each column is the network output on the given example
        y_true: numpy array of shape (batch_size,) containing the true labels of the batch

        Output:
        Cross entropy loss with respect to logits
        """
        m = y_true.shape[0]
        # Compute log-sum-exp across each column for normalization
        log_probs = logits - logsumexp(logits, axis=0)
        y_one_hot = np.eye(10)[y_true].T  # Assuming 10 classes
        # Compute the cross-entropy loss
        loss = -np.sum(y_one_hot * log_probs) / m
        return loss

    @staticmethod
    def cross_entropy_derivative(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Description: : Implements derivative for cross entropy loss.

        Input:
        logits: numpy array of shape (10, batch_size) where each column is the network output on the given example
        y_true: numpy array of shape (batch_size,) containing the true labels of the batch

        Output:
        A numpy array of shape (10,batch_size) where each column is the gradient of the loss with respect to y_pred
        for the given example.
        """
        n_samples = y_true.shape[0]
        n_classes = logits.shape[0]

        y_one_hot = np.eye(n_classes)[y_true].T

        d_cost = logits - y_one_hot  # / num_samples

        return d_cost

    def forward_propagation(self, x: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        """
        Implement the forward step of the backpropagation algorithm.

        Input:
        x - numpy array of shape (784, batch_size) - the input to the network

        Outputs:
        ZL - numpy array of shape (10, batch_size), the output of the network on the input X
        forward_outputs - A list of length self.num_layers, containing the forward computation
        (parameters & output of each layer).
        """
        zl = x
        forward_outputs = []

        # Linear forward-propagation with ReLU for layers 1 to (num_layers - 1)
        for layer in range(1, self.num_layers):
            zl_prev = zl
            activation, cache = self.linear_forward(activation=zl_prev,
                                                    w=self.parameters['W' + str(layer)],
                                                    b=self.parameters['b' + str(layer)],
                                                    layer=layer)
            # applies ReLU
            cache[f'z{layer}'] = activation
            zl = self.relu(z=activation)

            forward_outputs.append(cache)

        # Last layer
        last_layer = self.num_layers
        # Linear forward-propagation with Softmax for last layer
        activation, cache = self.linear_forward(activation=zl,
                                                w=self.parameters['W' + str(last_layer)],
                                                b=self.parameters['b' + str(last_layer)],
                                                layer=last_layer)

        cache[f'z{last_layer}'] = activation
        ZL = softmax(x=activation)

        forward_outputs.append(cache)

        return ZL, forward_outputs

    @staticmethod
    def linear_forward(activation: np.ndarray,
                       w: np.array,
                       b: np.array,
                       layer: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Description: Implement the linear part of a layer's forward propagation.

        Input:
        activation – the activations of the previous layer.
        w – the weight matrix of the current layer (of shape [size of current layer, size of previous layer]).
        b – the bias vector of the current layer (of shape [size of current layer, 1]).

        Output:
        zl – the linear component of the activations function (i.e., the value before applying the non-linear function).
        cache – a dictionary containing activation, w, b (stored for making the backpropagation easier to compute).
        """
        cache = {}
        zl = np.dot(w, activation) + b
        cache[f'x{layer}'] = activation
        cache[f'W{layer}'] = w
        cache[f'b{layer}'] = b

        return zl, cache

    def backpropagation(self, ZL: np.ndarray, forward_outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """
        Implement the backward step of the backpropagation algorithm.

        Input:
        ZL -  numpy array of shape (10, batch_size), the output of the network on the input X
        Y - numpy array of shape (batch_size,) containing the labels of each example in the current batch.
        forward_outputs - list of length self.num_layers given by the output of the forward function

        Returns:
        grads - dictionary containing the gradients of the loss with respect to the network parameters across the batch.
        grads["dW" + str(l)] is a numpy array of shape (sizes[l], sizes[l-1]),
        grads["db" + str(l)] is a numpy array of shape (sizes[l],1).
        
        """
        grads = {}
        d_cost = ZL

        last_layer = self.num_layers

        # Linear backpropagation with respect to Softmax derivative for last layer
        linear_cache = (forward_outputs[last_layer - 1].get(f'x{last_layer}'),
                        forward_outputs[last_layer - 1].get(f'W{last_layer}'),
                        forward_outputs[last_layer - 1].get(f'b{last_layer}'))
        activation_cache = forward_outputs[last_layer - 1].get(f'z{last_layer}')

        dz = self.softmax_backward(d_a=d_cost, activation_cache=activation_cache)
        da_prev, dw, dx = self.linear_backward(dz=dz, cache=linear_cache)

        grads[f'dA{last_layer - 1}'], grads[f'dW{last_layer}'], grads[f'db{last_layer}'] = da_prev, dw, dx

        # Linear backpropagation with respect to ReLU derivative for (last layer - 1) to 1
        for layer in range(self.num_layers, 0, -1):
            linear_cache = (forward_outputs[layer - 1].get(f'x{layer}'),
                            forward_outputs[layer - 1].get(f'W{layer}'),
                            forward_outputs[layer - 1].get(f'b{layer}'))
            activation_cache = forward_outputs[layer - 1].get(f'z{layer}')

            dz = self.relu_derivative(d_cost, activation_cache)
            da_prev, dw, dx = self.linear_backward(dz=dz, cache=linear_cache)

            grads[f'dA{layer - 1}'], grads[f'dW{layer}'], grads[f'db{layer}'] = da_prev, dw, dx

        return grads

    @staticmethod
    def linear_backward(dz: np.ndarray, cache: tuple) -> tuple[np.array, np.ndarray, np.ndarray]:
        """
        description: Implements the linear part of the backward propagation process for a single layer

        Input:
        dz – the gradient of the cost with respect to the linear output of the current layer (layer l)
        cache – tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Output:
        da_prev - gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dw - gradient of the cost with respect to W (current layer l), same shape as W
        db - gradient of the cost with respect to b (current layer l), same shape as b
        """

        activation_prev, w, b = cache
        m = activation_prev.shape[1]

        # dW(l) = dL/db(l)
        dw = np.dot(dz, activation_prev.T) / m

        # db(l) = dL/db(l)
        db = np.sum(dz, axis=1, keepdims=True) / m

        # dA(l-1) = dL/dA(l-1)
        da_prev = np.dot(w.T, dz)

        return da_prev, dw, db

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

                # forward pass:
                ZL, forward_outputs = self.forward_propagation(x=X_batch)

                cost = self.cross_entropy_loss(logits=ZL, y_true=Y_batch)
                costs.append(cost)

                # backward pass:
                d_cost = self.cross_entropy_derivative(logits=ZL, y_true=Y_batch)

                grads = self.backpropagation(ZL=d_cost, forward_outputs=forward_outputs)
                
                # applies learning with SGD
                self.parameters = self.sgd_step(grads, learning_rate)

                preds = np.argmax(ZL, axis=0)
                train_acc = self.calculate_accuracy(preds, Y_batch, batch_size)
                acc.append(train_acc)

            average_train_cost = np.mean(costs)
            average_train_acc = np.mean(acc)
            print(f"Epoch: {epoch + 1}, Training loss: {average_train_cost:.20f}, Training accuracy:"
                  f" {average_train_acc:.20f}")

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

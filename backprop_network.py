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

    def relu_derivative(self, z: np.ndarray, d_a_prev: np.ndarray) -> np.ndarray:
        """
        Description: Implements backward propagation for a ReLU unit.

        Input:
        d_a – the post-activation gradient.
        activation_cache – contains z (stored during the forward propagation).

        Output:
        d_z – gradient of the cost with respect to Z.
        """
        d_z = (z > 0).astype(float)
        d_z *= d_a_prev
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
        m = y_true.shape[0]
        y_one_hot = np.eye(10)[y_true].T
        d_loss = softmax(logits, axis=0) - y_one_hot
        d_loss /= m
        return d_loss

    def forward_propagation(self, x: np.ndarray) -> tuple[np.ndarray, list[any]]:
        """
        Implement the forward step of the backpropagation algorithm.

        Input:
        x - numpy array of shape (784, batch_size) - the input to the network

        Outputs:
        ZL - numpy array of shape (10, batch_size), the output of the network on the input X
        forward_outputs - A list of length self.num_layers, containing the forward computation
        (parameters & output of each layer).
        """
        a = x
        forward_outputs = []

        # Linear forward-propagation with ReLU for layers 1 to (num_layers - 1)
        for layer in range(1, self.num_layers + 1):
            w = self.parameters[f'W{layer}']
            b = self.parameters[f'b{layer}']
            zl = np.dot(w, a) + b

            forward_outputs.append((a, w, b, zl))

            # applies ReLU only for L - 2 layers
            if layer < self.num_layers:
                a = self.relu(z=zl)
            else:
                a = zl

        return a, forward_outputs

    def backpropagation(self, zl: np.ndarray, Y: np.ndarray, forward_outputs: list[dict[str, np.ndarray]]) -> (
            dict)[str, np.ndarray]:
        """
        Implement the backward step of the backpropagation algorithm.

        Input:
        zl -  numpy array of shape (10, batch_size), the output of the network on the input X
        Y - numpy array of shape (batch_size,) containing the labels of each example in the current batch.
        forward_outputs - list of length self.num_layers given by the output of the forward function

        Returns:
        grads - dictionary containing the gradients of the loss with respect to the network parameters across the batch.
        grads["dW" + str(l)] is a numpy array of shape (sizes[l], sizes[l-1]),
        grads["db" + str(l)] is a numpy array of shape (sizes[l],1).

        """
        grads = {}
        num_samples = Y.shape[0]
        last_layer = self.num_layers

        # Linear backpropagation with respect to Cross-Entropy derivative for last layer
        d_loss = self.cross_entropy_derivative(logits=zl, y_true=Y)
        a_prev, w, b, z = forward_outputs[last_layer - 1]

        grads[f'dW{last_layer}'] = np.dot(d_loss, a_prev.T) / num_samples
        grads[f'db{last_layer}'] = np.sum(d_loss, axis=1, keepdims=True) / num_samples
        grads[f'da{last_layer - 1}'] = np.dot(w.T, d_loss)

        # Linear backpropagation with respect to ReLU derivative for (last layer - 1) to 1
        for layer in range(self.num_layers - 1, 0, -1):
            a_prev, w, b, z = forward_outputs[layer - 1]
            d_z = self.relu_derivative(z=z, d_a_prev=grads[f'da{layer}'])

            grads[f'dW{layer}'] = np.dot(d_z, a_prev.T) / num_samples
            grads[f'db{layer}'] = np.sum(d_z, axis=1, keepdims=True) / num_samples
            grads[f'da{layer - 1}'] = np.dot(w.T, d_z)

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
            cost_list = []
            accuracy_list = []
            for i in range(0, x_train.shape[1], batch_size):
                X_batch = x_train[:, i:i + batch_size]
                Y_batch = y_train[i:i + batch_size]

                # Forward pass:
                ZL, forward_outputs = self.forward_propagation(x=X_batch)
                # Loss computation
                cost = self.cross_entropy_loss(logits=ZL, y_true=Y_batch)
                cost_list.append(cost)
                # Backward pass:
                grads = self.backpropagation(zl=ZL, Y=Y_batch, forward_outputs=forward_outputs)
                # Applies learning with SGD
                self.parameters = self.sgd_step(grads, learning_rate)

                preds = np.argmax(ZL, axis=0)
                train_acc = self.calculate_accuracy(preds, Y_batch, batch_size)
                accuracy_list.append(train_acc)

            average_train_cost = np.mean(cost_list)
            average_train_acc = np.mean(accuracy_list)

            epoch_train_cost.append(average_train_cost)
            epoch_train_acc.append(average_train_acc)

            # Evaluate test error
            ZL, caches = self.forward_propagation(x_test)
            test_cost = self.cross_entropy_loss(ZL, y_test)
            preds = np.argmax(ZL, axis=0)
            test_acc = self.calculate_accuracy(preds, y_test, len(y_test))

            epoch_test_cost.append(test_cost)
            epoch_test_acc.append(test_acc)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch + 1}, Training loss: {average_train_cost:.20f}, Training accuracy:"
                      f" {average_train_acc:.20f}")

                print(f"Epoch: {epoch + 1}, Test loss: {test_cost:.20f}, Test accuracy: {test_acc:.20f}")

            if 0.98 < test_acc:
                print(f"break after {epoch} epochs")
                break

        return self.parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc

    @staticmethod
    def calculate_accuracy(y_pred, y_true, batch_size):
        """Returns the average accuracy of the prediction over the batch """
        return np.sum(y_pred == y_true) / batch_size

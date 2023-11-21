from typing import List, Tuple

import numpy as np
from numpy._typing import NDArray
from activation_functions import ActivationFunction
from multilayer_perceptron import multilayer_perceptron
from optimization_methods import OptimizationMethod
from vae_loss_functions import identity, squared_error


def standard_autoencoder(
    data: NDArray,
    hidden_layer_sizes: List[int],
    latent_layer_size: int,
    target_error: float,
    max_epochs: int,
    batch_size: int,
    neuron_activation_function: ActivationFunction,
    neuron_activation_function_derivative: ActivationFunction,
    optimization_method: OptimizationMethod,
) -> Tuple[List[NDArray], List[float]]:
    mlp_hidden_layer_sizes = (
        hidden_layer_sizes + [latent_layer_size] + hidden_layer_sizes[::-1]
    )
    mlp_data = [(data[i], data[i]) for i in range(data.shape[0])]
    mlp_output_layer_size = data.shape[1]

    return multilayer_perceptron(
        mlp_data,
        mlp_hidden_layer_sizes,
        mlp_output_layer_size,
        target_error,
        max_epochs,
        batch_size,
        neuron_activation_function,
        neuron_activation_function_derivative,
        optimization_method,
    )


def denoising_autoencoder(
    data: NDArray,
    hidden_layer_sizes: List[int],
    latent_layer_size: int,
    target_error: float,
    max_epochs: int,
    batch_size: int,
    neuron_activation_function: ActivationFunction,
    neuron_activation_function_derivative: ActivationFunction,
    optimization_method: OptimizationMethod,
    noise_probability: float,
) -> Tuple[List[NDArray], List[float], NDArray]:
    mlp_hidden_layer_sizes = (
        hidden_layer_sizes + [latent_layer_size] + hidden_layer_sizes[::-1]
    )
    mlp_output_layer_size = data.shape[1]

    # Add noise to the data, flipping bits with probability noise_probability
    noisy_data = data.copy()
    for sample in noisy_data:
        for i in range(sample.shape[0]):
            if np.random.uniform() < noise_probability:
                sample[i] = 1 - sample[i]

    mlp_data = [(noisy_data[i], data[i]) for i in range(data.shape[0])]

    best_network, errors_in_epoch = multilayer_perceptron(
        mlp_data,
        mlp_hidden_layer_sizes,
        mlp_output_layer_size,
        target_error,
        max_epochs,
        batch_size,
        neuron_activation_function,
        neuron_activation_function_derivative,
        optimization_method,
    )
    return best_network, errors_in_epoch, noisy_data


class Network:
    def __init__(
        self, dimensions, learning_rate, max_epochs, batch_size, loss_func, act_func
    ):
        """intializes weights matrix and parameters"""

        # initialize weights of network
        self.weights = {}
        for i in range(len(dimensions) - 1):
            self.weights[i] = np.random.uniform(
                -0.1, 0.1, (dimensions[i], dimensions[i + 1])
            )

        # hyperparameters
        self.learning_rate = learning_rate
        self.iter = max_epochs
        self.batch_size = batch_size

        self.activation = act_func

        self.loss = loss_func

    def _feedforward(self, X):
        """feedforward update step"""
        self._z = {}
        self._z_act = {0: X}

        for i in range(len(self.weights)):
            self._z[i] = self._z_act[i] @ self.weights[i]
            self._z_act[i + 1] = self.activation(self._z[i])[0]
        return self._z_act[i + 1]

    def _backprop(self, X, y, yhat):
        """back-propagation algorithm"""
        n = len(self.weights)
        delta = -1 * self.loss(y, yhat)[1] * self.activation(self._z[n - 1])[1]
        grad_weights = {n - 1: self._z_act[n - 1].T @ delta}

        for i in reversed(range(len(self.weights) - 1)):
            delta = delta @ self.weights[i + 1].T * self.activation(self._z[i])[1]
            grad_weights[i] = self._z_act[i].T @ delta

        return grad_weights

    def train(self, X, y):
        """trains model using stochastic gradient descent"""
        X_batch = X
        y_batch = y

        for i in range(self.iter):
            if self.batch_size > 0 and self.batch_size < X.shape[0]:
                k = np.random.choice(range(X.shape[0]), self.batch_size, replace=False)
                X_batch = X[k, :]
                y_batch = y[k, :]

            yhat = self._feedforward(X_batch)
            grad_weights = self._backprop(X_batch, y_batch, yhat)

            for j in range(len(self.weights)):
                self.weights[j] -= self.learning_rate * grad_weights[j]

    def predict(self, X):
        """predicts on trained model"""
        z_act = X
        for i in range(len(self.weights)):
            z = z_act @ self.weights[i]
            z_act = self.activation(z)[0]
        return z_act


class VAE:
    def __init__(
        self,
        dimensions,
        latent_dim,
        learning_rate,
        max_epochs,
        batch_size,
        loss_func,
        act_func,
    ):
        self.latent_dim = latent_dim
        self.encoder = Network(
            dimensions[0] + [2],
            learning_rate,
            max_epochs,
            batch_size,
            loss_func,
            act_func,
        )
        self.decoder = Network(
            [latent_dim] + dimensions[1],
            learning_rate,
            max_epochs,
            batch_size,
            loss_func,
            act_func,
        )

        for i in range(len(self.encoder.weights)):
            self.encoder.weights[i] = np.abs(self.encoder.weights[i])

        for i in range(len(self.encoder.weights)):
            self.decoder.weights[i] = np.abs(self.decoder.weights[i])

        self.batch_size = batch_size
        self.iter = max_epochs
        self.encoder.loss = identity
        self.decoder.loss = squared_error

        self.activation = act_func

    def _forwardstep(self, X):
        # encoder learns parameters
        latent = self.encoder._feedforward(X)
        self.mu = latent[:, 0]
        self.sigma = np.exp(latent[:, 1])

        # sample from gaussian with learned parameters
        epsilon = np.random.normal(0, 1, size=(X.shape[0], self.latent_dim))
        z_sample = self.mu[:, None] + np.sqrt(self.sigma)[:, None] * epsilon

        # pass sampled vector through to decoder
        X_hat = self.decoder._feedforward(z_sample)
        return X_hat

    def _kl_divergence_loss(self):
        d_mu = self.mu
        d_s2 = 1 - 1 / (2 * (self.sigma + 1e-6))
        return np.vstack((d_mu, d_s2)).T

    def _backwardstep(self, X, X_hat):
        # propagate reconstuction error through decoder
        n = len(self.decoder.weights)
        delta = (
            -1
            * self.decoder.loss(X, X_hat)[1]
            * self.activation(self.decoder._z[n - 1])[1]
        )
        decoder_weights = {n - 1: self.decoder._z_act[n - 1].T @ delta}

        for i in reversed(range(len(self.decoder.weights) - 1)):
            delta = (
                delta
                @ self.decoder.weights[i + 1].T
                * self.activation(self.decoder._z[i])[1]
            )
            decoder_weights[i] = self.decoder._z_act[i].T @ delta

        # add kl-divergence loss
        m = len(self.encoder.weights)
        kl_loss = self._kl_divergence_loss()
        kl_delta = kl_loss * self.activation(self.encoder._z[m - 1])[1]

        delta = (
            delta
            @ self.decoder.weights[0].T
            * self.activation(self.encoder._z[m - 1])[1]
        )
        delta = delta + kl_delta
        encoder_weights = {m - 1: self.encoder._z_act[n - 1].T @ delta}

        # propagate kl error through encoder
        for i in reversed(range(len(self.decoder.weights) - 1)):
            delta = (
                delta
                @ self.encoder.weights[i + 1].T
                * self.activation(self.encoder._z[i])[1]
            )
            encoder_weights[i] = self.encoder._z_act[i].T @ delta

        return encoder_weights, decoder_weights

    def learn(self, X):
        X_batch = X

        loss_per_epoch = []

        for i in range(self.iter):
            if self.batch_size > 0 and self.batch_size < X.shape[0]:
                k = np.random.choice(range(X.shape[0]), self.batch_size, replace=False)
                X_batch = X[k, :]

            X_hat = self._forwardstep(X_batch)
            grad_encoder, grad_decoder = self._backwardstep(X_batch, X_hat)

            loss = self.decoder.loss(X_batch, X_hat)[0]
            kl_loss = np.mean(self._kl_divergence_loss() ** 2)

            loss_per_epoch.append(loss + kl_loss)

            # Print each 1%
            if i % (self.iter // 100) == 0:
                print(f"Epoch {i} of {self.iter} - Loss: {loss_per_epoch[-1]}")

            for j in range(len(self.encoder.weights)):
                self.encoder.weights[j] -= self.encoder.learning_rate * grad_encoder[j]

            for j in range(len(self.decoder.weights)):
                self.decoder.weights[j] -= self.decoder.learning_rate * grad_decoder[j]

        return loss_per_epoch

    def generate(self, z=None):
        if not np.any(z):
            z = np.random.normal(0, 1, size=(1, self.latent_dim))
        return self.decoder.predict(z)

    def encode_decode(self, X):
        return self._forwardstep(X)

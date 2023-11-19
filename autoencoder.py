from typing import List, Tuple

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy._typing import NDArray
from torch.utils.data import DataLoader

import optimization_methods
from activation_functions import ActivationFunction
from multilayer_perceptron import multilayer_perceptron
from optimization_methods import OptimizationMethod


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
) -> Tuple[List[NDArray], List[float]]:
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


class VAE():
    def __init__(self, act_func, max_epochs, batch_size, num_samples, input_dim, latent_dim, hidden_dim):
        # Initialize weights and biases
        # Encoder weights and biases
        self.encoder_weights = np.random.randn(input_dim, hidden_dim)
        self.encoder_bias_hidden = np.zeros(hidden_dim)
        self.encoder_weights_mean = np.random.randn(hidden_dim, latent_dim)
        self.encoder_weights_var = np.random.randn(hidden_dim, latent_dim)
        self.encoder_bias_mean = np.zeros(latent_dim)
        self.encoder_bias_var = np.zeros(latent_dim)

        # Decoder weights and biases
        self.decoder_weights = np.random.randn(latent_dim, hidden_dim)
        self.decoder_bias_hidden = np.zeros(hidden_dim)
        self.decoder_weights_output = np.random.randn(hidden_dim, input_dim)
        self.decoder_bias_output = np.zeros(input_dim)

        # Define activation function
        self.act_func = act_func

        # Define number of epochs, batch size, and number of samples
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_samples = num_samples

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def encode(self, x):
        hidden = self.act_func(np.dot(x, self.encoder_weights) + self.encoder_bias_hidden)
        mean = np.dot(hidden, self.encoder_weights_mean) + self.encoder_bias_mean
        log_var = np.dot(hidden, self.encoder_weights_var) + self.encoder_bias_var
        return mean, log_var

    def reparameterize(self, mean, log_var):
        epsilon = np.random.normal(size=mean.shape)
        std_dev = np.exp(0.5 * log_var)
        z = mean + std_dev * epsilon
        return z

    def decode(self, z):
        hidden = self.act_func(np.dot(z, self.decoder_weights) + self.decoder_bias_hidden)
        output = self.act_func(np.dot(hidden, self.decoder_weights_output) + self.decoder_bias_output)
        return output

    def train(self, input_data):

        for epoch in range(self.max_epochs):
            for i in range(0, self.num_samples, self.batch_size):
                # Get a batch of data
                batch = input_data[i:i + self.batch_size]

                # Forward pass: Encode, sample, and decode
                mean, log_var = self.encode(batch)
                z = self.reparameterize(mean, log_var)
                reconstructed = self.decode(z)

                # Calculate reconstruction loss (using mean squared error)
                reconstruction_loss = np.mean(np.square(batch - reconstructed))

                # Calculate KL divergence
                kl_loss = -0.5 * np.mean(1 + log_var - np.square(mean) - np.exp(log_var))

                # Total loss
                total_loss = reconstruction_loss + kl_loss

                # TODO: Backpropagation
                # TODO: Compute gradients and update weights and biases

                # Print or log training progress
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], Batch [{i + 1}/{self.num_samples}], Total Loss: {total_loss}")

            # Print or log epoch-wise information

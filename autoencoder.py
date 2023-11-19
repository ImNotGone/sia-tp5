from typing import List, Tuple
from numpy._typing import NDArray

from activation_functions import ActivationFunction
from optimization_methods import OptimizationMethod
from multilayer_perceptron import multilayer_perceptron

import numpy as np


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

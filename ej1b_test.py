from src.activation_functions import get_activation_function
from src.dataset_loaders import load_font_data
from src.optimization_methods import adam
from src.autoencoder import denoising_autoencoder
from src.plots import plot_errors_per_architecture

import numpy as np
import json
import multiprocessing


def architecture_test():
    architectures = [
        ([30, 25, 20, 15, 10, 5], 2, "30-25-20-15-10-5/2"),
        ([25, 15, 5], 2, "25-15-5/2"),
        ([30, 25, 20, 15, 10, 5], 4, "30-25-20-15-10-5/4"),
        ([25, 15, 5], 4, "25-15-5/4"),
    ]

    data = load_font_data()
    data = data.reshape((data.shape[0], -1))

    epochs = 10000

    target_error = 0.1

    batch_size = data.shape[0]

    activation_function, activation_function_derivative, _ = get_activation_function(
        "tanh", 1
    )

    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    optimization_method = lambda weight_delta, state: adam(
        weight_delta, learning_rate, beta1, beta2, epsilon, state
    )

    noise_probability = 0.01

    mean_errors_per_architecture = {}

    iterations = 10

    for architecture, latent_space_size, name in architectures:
        errors = multiprocessing.Manager().list()

        # Create processes
        processes = []
        for _ in range(iterations):
            processes.append(
                multiprocessing.Process(
                    target=train_and_calculate_error,
                    args=(
                        architecture,
                        data,
                        latent_space_size,
                        target_error,
                        epochs,
                        batch_size,
                        activation_function,
                        activation_function_derivative,
                        optimization_method,
                        noise_probability,
                        errors,
                    ),
                )
            )

        # Start threads
        for process in processes:
            process.start()

        # Wait for threads to finish
        for process in processes:
            process.join()

        mean_error = np.mean(errors)
        std_error = np.std(errors)

        mean_errors_per_architecture[name] = (mean_error, std_error)

    plot_errors_per_architecture(mean_errors_per_architecture)

    # Serialize errors
    with open("errors_architecture.json", "w") as f:
        json.dump(mean_errors_per_architecture, f)


def train_and_calculate_error(
    architecture,
    data,
    latent_space_size,
    target_error,
    epochs,
    batch_size,
    activation_function,
    activation_function_derivative,
    optimization_method,
    noise_probability,
    errors_list,
):
    # Get a random seed for the process
    pid = multiprocessing.current_process().pid
    np.random.seed(pid)

    autoencoder, errors_per_epoch, _ = denoising_autoencoder(
        data,
        architecture,
        latent_space_size,
        target_error,
        epochs,
        batch_size,
        activation_function,
        activation_function_derivative,
        optimization_method,
        noise_probability,
    )

    best_error = min(errors_per_epoch)

    errors_list.append(best_error)


if __name__ == "__main__":
    architecture_test()

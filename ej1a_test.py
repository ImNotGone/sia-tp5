from numpy.random import random
from activation_functions import get_activation_function
from dataset_loaders import load_font_data
from optimization_methods import get_optimization_method, gradient_descent, adam, momentum
from autoencoder import standard_autoencoder
from plots import plot_errors_per_architecture, plot_errors_per_optimization_method

import numpy as np
import json
import multiprocessing


def architecture_test():

    architectures = [
        ([30, 25, 20, 15, 10, 5], "30-25-20-15-10-5"),
        ([25, 15, 5], "25-15-5"),
        ([20, 10, 5], "20-10-5"),
        ([20, 10], "20-10"),
    ]

    data = load_font_data()
    data = data.reshape((data.shape[0], -1))

    epochs = 10000

    latent_space_size = 2

    target_error = 0.1

    batch_size = data.shape[0]

    activation_function, activation_function_derivative, _ = get_activation_function(
        "tanh", 1
    )

    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    optimization_method = lambda weight_delta, state: adam(weight_delta, learning_rate, beta1, beta2, epsilon, state)

    mean_errors_per_architecture = {}

    iterations = 10

    for architecture, name in architectures:
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

def optimization_method_test():
    architecture = [30, 25, 20, 15, 10, 5]

    data = load_font_data()
    data = data.reshape((data.shape[0], -1))

    epochs = 10000

    latent_space_size = 2

    target_error = 0.1

    batch_size = data.shape[0]

    activation_function, activation_function_derivative, _ = get_activation_function(
        "tanh", 1
    )

    learning_rate = 0.01
    gradient_descent_optimizer= lambda weight_delta, _: gradient_descent(weight_delta, learning_rate)

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    adam_optimizer = lambda weight_delta, state: adam(weight_delta, learning_rate, beta1, beta2, epsilon, state)

    momentum_constant = 0.9
    momentum_optimizer = lambda weight_delta, state: momentum(weight_delta, learning_rate, momentum_constant)

    optimization_methods = [
        (gradient_descent_optimizer, "Gradient Descent"),
        (momentum_optimizer, "Momentum"),
        (adam_optimizer, "Adam"),
    ]

    mean_errors_per_optimization_method = {}

    iterations = 10

    for optimization_method, name in optimization_methods:
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

        mean_errors_per_optimization_method[name] = (mean_error, std_error)

    plot_errors_per_optimization_method(mean_errors_per_optimization_method)

    # Serialize errors
    with open("errors_optimization.json", "w") as f:
        json.dump(mean_errors_per_optimization_method, f)






def train_and_calculate_error(architecture, data, latent_space_size, target_error, epochs, batch_size, activation_function, activation_function_derivative, optimization_method, errors_list):
    
    # Get a random seed for the process
    pid = multiprocessing.current_process().pid
    np.random.seed(pid)

    autoencoder, errors_per_epoch = standard_autoencoder(
        data,
        architecture,
        latent_space_size,
        target_error,
        epochs,
        batch_size,
        activation_function,
        activation_function_derivative,
        optimization_method,
    )

    best_error = min(errors_per_epoch)

    errors_list.append(best_error)
    

if __name__ == "__main__":
    optimization_method_test()
    architecture_test()

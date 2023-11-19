from dataset_loaders import load_font_data
from activation_functions import get_activation_function
from autoencoder import denoising_autoencoder
from multilayer_perceptron import forward_propagation
from utils import (
    deserialize_weights,
    pretty_print_font,
    get_batch_size,
    serialize_weights,
)

import json

from optimization_methods import get_optimization_method


def main():
    data = load_font_data()

    # Flatten the matrix into a vector
    data = data.reshape((data.shape[0], -1))

    with open("config.json", "r") as f:
        config = json.load(f)

        hidden_layer_sizes = config["hidden_layers"]
        latent_space_size = config["latent_space_size"]

        target_error = config["target_error"]
        max_epochs = config["max_epochs"]

        noise_probability = config["noise_probability"]

        batch_size = get_batch_size(config, data.shape[0])

        (
            activation_function,
            activation_derivative,
            activation_normalize,
        ) = get_activation_function(
            config["activation"]["function"], config["activation"]["beta"]
        )

        optimization_method = get_optimization_method(config["optimization"])

        if config["load_weights"]:
            weights = deserialize_weights(config["weights_file"])
        else:
            weights, errors_per_epoch = denoising_autoencoder(
                data,
                hidden_layer_sizes,
                latent_space_size,
                target_error,
                max_epochs,
                batch_size,
                activation_function,
                activation_derivative,
                optimization_method,
                noise_probability,
            )

        for sample in data:
            # Convert into a 7x5 matrix
            font = sample.reshape((7, 5))
            pretty_print_font(font)
            print()

            # Print the reconstructed font
            reconstructed_sample = forward_propagation(
                sample, weights, activation_function
            )[0][-1]
            reconstructed_font = reconstructed_sample.reshape((7, 5))
            pretty_print_font(reconstructed_font)
            print("-" * 20)
            print()

        if config["save_weights"]:
            serialize_weights(weights, config["weights_file"])


if __name__ == "__main__":
    main()
from src.dataset_loaders import load_font_data
from src.activation_functions import get_activation_function
from src.optimization_methods import get_optimization_method
from src.autoencoder import denoising_autoencoder
from src.multilayer_perceptron import predict
from src.utils import (
    create_image,
    deserialize_weights,
    get_batch_size,
    serialize_weights,
)
from src.noise import flipping_noise

import json


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

        noise_probability_training = config["noise_probability_training"]
        noise_probability_testing = config["noise_probability_testing"]

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
            weights, errors_per_epoch, training_data = denoising_autoencoder(
                data,
                hidden_layer_sizes,
                latent_space_size,
                target_error,
                max_epochs,
                batch_size,
                activation_function,
                activation_derivative,
                optimization_method,
                noise_probability_training,
            )

        original_data = data
        testing_data = flipping_noise(data, noise_probability_testing)

        original_fonts = data.reshape((-1, 7, 5))
        training_fonts = training_data.reshape((-1, 7, 5))
        testing_fonts = testing_data.reshape((-1, 7, 5))

        create_image(original_fonts, "original.png", (7, 5))
        create_image(training_fonts, "training.png", (7, 5))
        create_image(testing_fonts, "testing.png", (7, 5))

        reconstructed_with_original_fonts = []
        reconstructed_with_training_fonts = []
        reconstructed_with_testing_fonts = []

        for original_sample, training_sample, testing_sample in zip(
            original_data, training_data, testing_data
        ):
            reconstructed_with_original_sample = predict(
                original_sample, weights, activation_function
            )
            reconstructed_with_training_sample = predict(
                training_sample, weights, activation_function
            )
            reconstructed_with_testing_sample = predict(
                testing_sample, weights, activation_function
            )

            reconstructed_with_original_font = (
                reconstructed_with_original_sample.reshape((7, 5))
            )
            reconstructed_with_training_font = (
                reconstructed_with_training_sample.reshape((7, 5))
            )
            reconstructed_with_testing_font = reconstructed_with_testing_sample.reshape(
                (7, 5)
            )

            reconstructed_with_original_fonts.append(reconstructed_with_original_font)
            reconstructed_with_training_fonts.append(reconstructed_with_training_font)
            reconstructed_with_testing_fonts.append(reconstructed_with_testing_font)

        create_image(
            reconstructed_with_original_fonts, "reconstructed_with_original.png", (7, 5)
        )
        create_image(
            reconstructed_with_training_fonts, "reconstructed_with_training.png", (7, 5)
        )
        create_image(
            reconstructed_with_testing_fonts, "reconstructed_with_testing.png", (7, 5)
        )

        if config["save_weights"]:
            serialize_weights(weights, config["weights_file"])


if __name__ == "__main__":
    main()

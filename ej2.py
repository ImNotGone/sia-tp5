import json

import numpy as np

from src.dataset_loaders import load_emoji_data, load_font_data
from src.activation_functions import relu, relu_derivative
from src.vae.vaev2 import VAE

from src.optimization_methods import get_optimization_method

from src.utils import (
    create_image,
    deserialize_weights,
    get_batch_size,
    serialize_weights,
)

from src.plots import plot_errors_per_epoch

from src.vae.vae_loss_functions import squared_error

import pickle, gzip


def main():
    data = load_font_data()

    # Flatten the matrix into a vector
    data = data.reshape((data.shape[0], -1))

    with open("config.json", "r") as f:
        config = json.load(f)

        hidden_layer_sizes = config["hidden_layers"]

        target_error = config["target_error"]
        max_epochs = config["max_epochs"]

        batch_size = get_batch_size(config, data[0].shape[0])

        optimization_method = get_optimization_method(config["optimization"])

        learning_rate = 0.1
        loss_func = squared_error
        act_func = relu
        act_func_derivative = relu_derivative

        # 64 -> 16 -> 2 -> 16 -> 64
        vae = VAE(
            [35, 16, 16, 1],
            act_func,
            act_func_derivative,
            optimization_method,
            learning_rate
        )

        errors_per_epoch = vae.train(max_epochs, data)

        reconstructed_fonts = []
        encoded_samples = []
        original_fonts = []
        for emoji in data:
            reconstructed_sample = vae.encode(emoji[None, :])
            encoded_samples.append(reconstructed_sample)
            # reconstructed_font = reconstructed_sample.reshape((7, 5))
            # reconstructed_fonts.append(reconstructed_font)
            original_fonts.append(emoji.reshape((7, 5)))

        for encoded in encoded_samples:
            reconstructed_font = vae.decode(encoded)
            reconstructed_fonts.append(reconstructed_font.reshape((7, 5)))

        create_image(original_fonts, "original.png", (7, 5))
        create_image(reconstructed_fonts, "reconstructed.png", (7, 5))

        plot_errors_per_epoch(errors_per_epoch)


if __name__ == "__main__":
    main()

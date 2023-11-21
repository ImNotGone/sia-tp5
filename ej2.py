import json

import numpy as np

from dataset_loaders import load_emoji_data, load_font_data
from vae_activations import relu, sigmoid, tanh, identity, selu
from autoencoder import VAE

from optimization_methods import get_optimization_method

from utils import (
    create_image,
    deserialize_weights,
    get_batch_size,
    serialize_weights,
)

from plots import plot_errors_per_epoch

from vae_loss_functions import squared_error

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
        act_func = sigmoid

        # 64 -> 16 -> 2 -> 16 -> 64
        vae = VAE(
            [[35, 16], [16, 35]],
            2,
            learning_rate,
            max_epochs,
            batch_size,
            loss_func,
            act_func,
        )

        errors_per_epoch = vae.learn(data)

        reconstructed_fonts = []
        original_fonts = []
        for emoji in data:
            reconstructed_sample = vae.encode_decode(emoji[None, :])
            reconstructed_font = reconstructed_sample.reshape((7, 5))
            reconstructed_fonts.append(reconstructed_font)
            original_fonts.append(emoji.reshape((7, 5)))

        create_image(original_fonts, "original.png", (7, 5))
        create_image(reconstructed_fonts, "reconstructed.png", (7, 5))

        plot_errors_per_epoch(errors_per_epoch)


if __name__ == "__main__":
    main()

import json

import numpy as np

from src.dataset_loaders import load_emoji_data, load_emoji_new_data, load_font_data
from src.activation_functions import get_activation_function, relu, relu_derivative, logistic, logistic_derivative
from src.vae.vaev2 import VAE

from src.vae.vae_opt_methods import ADAM

from src.utils import (
    create_image,
    deserialize_weights,
    get_batch_size,
    serialize_weights,
)

from src.plots import plot_errors_per_epoch

from src.vae.vae_loss_functions import squared_error

def main():
    dataset = "emoji_new"

    if dataset == "emoji":
        data = load_emoji_data()
        data_dim = (8, 8)
        first_layer_size = 8 * 8
        image_dim = (4, 3)
    elif dataset == "font":
        data = load_font_data()
        data_dim = (7, 5)
        first_layer_size = 7 * 5
        image_dim = (7, 5)
    elif dataset == "emoji_new":
        data = load_emoji_new_data()
        data_dim = (24, 24)
        first_layer_size = 24 * 24
        image_dim = (4, 3)
    else:
        raise Exception("Invalid dataset")

    # Flatten the matrix into a vector
    data = data.reshape((data.shape[0], -1))

    with open("config.json", "r") as f:
        config = json.load(f)

        hidden_layer_sizes = config["hidden_layers"]

        target_error = config["target_error"]
        max_epochs = config["max_epochs"]

        batch_size = get_batch_size(config, data[0].shape[0])

        optimization_method = ADAM

        learning_rate = [0.001, 0.9, 0.999, 1e-8]
        loss_func = squared_error
        act_func, act_func_derivative, _ = get_activation_function(config["activation"]["function"], config["activation"]["beta"])

        # 64 -> 16 -> 2 -> 16 -> 64
        vae = VAE(
            [first_layer_size, 35, 35, 35, 2],
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
            reconstructed_sample = vae.encode(emoji)
            encoded_samples.append(reconstructed_sample)
            # reconstructed_font = reconstructed_sample.reshape((7, 5))
            # reconstructed_fonts.append(reconstructed_font)
            original_fonts.append(emoji.reshape(data_dim))

        for encoded in encoded_samples:
            reconstructed_font = vae.decode(encoded)
            reconstructed_fonts.append(reconstructed_font.reshape(data_dim))

        create_image(original_fonts, "original.png", image_dim)
        create_image(reconstructed_fonts, "reconstructed.png", image_dim)

        plot_errors_per_epoch(errors_per_epoch)


if __name__ == "__main__":
    main()

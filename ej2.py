import json

import numpy as np

from dataset_loaders import load_emoji_data
from activation_functions import get_activation_function
from autoencoder import VAE

from optimization_methods import get_optimization_method

from utils import (
    create_image,
    deserialize_weights,
    get_batch_size,
    serialize_weights,
)

from plots import plot_errors_per_epoch

from loss_functions import squared_error

def main():
    data = load_emoji_data()

    # Flatten the matrix into a vector
    data = data.reshape((data.shape[0], -1))

    with open("config.json", "r") as f:
        config = json.load(f)

        hidden_layer_sizes = config["hidden_layers"]

        target_error = config["target_error"]
        max_epochs = config["max_epochs"]

        batch_size = get_batch_size(config, data.shape[0])

        (
            activation_function,
            activation_derivative,
            _,
        ) = get_activation_function(
            config["activation"]["function"], config["activation"]["beta"]
        )

        optimization_method = get_optimization_method(config["optimization"])

        errors_per_epoch = []
        learning_rate = 0.1
        loss_func = squared_error
        act_func = activation_function

        # 64 -> 16 -> 2 -> 16 -> 64
        vae = VAE(
            [[64, 16], [16, 64]],
            2,
            learning_rate,
            max_epochs,
            batch_size,
            loss_func,
            act_func
        )

        vae.learn(data)

        for emoji in data:
            x_prima = vae.encode_decode(emoji)
            x = emoji.reshape((8, 8))
            x_prima = x_prima.reshape((8, 8))
            text_row = ""
            print("X: ")
            for row in x:
                for value in row:
                    if value == 1:
                        text_row += "#"
                    else:
                        text_row += " "
                print(text_row)
                text_row = ""

            print("X\'")
            for row in x_prima:
                for value in row:
                    if value == 1:
                        text_row += "#"
                    else:
                        text_row += " "
                print(text_row)
                text_row = ""

        # plot_errors_per_epoch(errors_per_epoch)

if __name__ == "__main__":
    main()

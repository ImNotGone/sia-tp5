from src.dataset_loaders import load_font_data
from src.activation_functions import get_activation_function
from src.optimization_methods import get_optimization_method
from src.autoencoder import standard_autoencoder
from src.multilayer_perceptron import forward_propagation, get_hidden
import math
from src.utils import (
    create_image,
    deserialize_weights,
    get_batch_size,
    serialize_weights,
)

import json



def main():
    font = [
        "`",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "{",
        "|",
        "}",
        "~",
        "del",
    ]
    data = load_font_data()

    # Flatten the matrix into a vector
    data = data.reshape((data.shape[0], -1))

    with open("config.json", "r") as f:
        config = json.load(f)

        hidden_layer_sizes = config["hidden_layers"]
        latent_space_size = config["latent_space_size"]

        target_error = config["target_error"]
        max_epochs = config["max_epochs"]

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
            weights, errors_per_epoch = standard_autoencoder(
                data,
                hidden_layer_sizes,
                latent_space_size,
                target_error,
                max_epochs,
                batch_size,
                activation_function,
                activation_derivative,
                optimization_method,
            )

        new_weights = weights[math.ceil(len(weights) / 2) :]

        original_fonts = data.reshape((-1, 7, 5))
        reconstructed_fonts = []
        data2d = []
        i = 0
        for sample in data:
            reconstructed_sample = forward_propagation(
                sample, weights, activation_function
            )[0][-1]

            # OJO poner en vez de 5 la que vendria a ser capa latente (este caso es input(0) + 4 ocultas(1-4)=5)
            data2d.append(
                [
                    get_hidden(
                        sample,
                        weights,
                        activation_function,
                        math.ceil(len(weights) / 2),
                    ),
                    font[i],
                ]
            )
            i += 1

        print(data2d)
        x_values = []
        y_values = []
        labels = []

        for item in data2d:
            x_values.append(item[0][0])  # Agrega la primera coordenada x
            y_values.append(item[0][1])  # Agrega la primera coordenada y
            labels.append(item[1])  # Agrega la lista de números como labels

        x1, y1 = x_values[4], y_values[4]
        x2, y2 = x_values[10], y_values[10]

        # cuantas letras intermedias
        n = 5
        newx = x1
        newy = y1
        for i in range(n + 1):
            print(newx, newy)
            new_letters = forward_propagation(
                [newx, newy], new_weights, activation_function
            )[0][-1]
            reconstructed_font = new_letters.reshape((7, 5))
            reconstructed_fonts.append(reconstructed_font)
            newx = newx + (x2 - x1) / n
            newy = newy + (y2 - y1) / n

        create_image(reconstructed_fonts, "newletters.png", (1, 6))
        if config["save_weights"]:
            serialize_weights(weights, config["weights_file"])


if __name__ == "__main__":
    main()

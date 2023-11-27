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
    dataset = "emoji"

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
        latent_dim = 2
        vae = VAE(
            [first_layer_size, 64, 64, 64, latent_dim],
            act_func,
            act_func_derivative,
            optimization_method,
            learning_rate
        )

        errors_per_epoch = []
        if config["load_weights"]:
            vae.load_weights()
        else:
            errors_per_epoch = vae.train(max_epochs, data)

        if config["save_weights"]:
            vae.save_weights()

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

        if latent_dim == 1:
            plot_latent_space_1d(vae, data)
            decode_latent_space_1d(vae, data, data_dim)
        elif latent_dim == 2:
            plot_latent_space_2d(vae, data)
            plot_latent_space_2d_point(vae, data)
            decode_latent_space_2d(vae, data, data_dim)
        


import matplotlib.pyplot as plt
def plot_latent_space_1d(vae, data):
    plt.figure()
    # In a vae, the output of the encoder is the mean and std of the latent space
    # So we can plot samples from the distribution of the latent space

    colors = ["red", "green", "blue", "yellow", "black", "orange", "purple", "pink", "brown", "gray"]

    # We need to plot samples following the std and mean of the latent space
    samples = 100

    # Sample the distribution for each sample
    for i, sample in enumerate(data):

        points = []
        for j in range(samples):
            encoded_sample = vae.encode_sample(sample)
            x, y = encoded_sample[0], 0
            points.append((x, y))

        x, y = zip(*points)

        plt.scatter(x, y, c=colors[i % len(colors)])
        

    plt.title("Latent space")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig("latent_space.png")

def plot_latent_space_2d(vae, data):
    plt.figure()
    # In a vae, the output of the encoder is the mean and std of the latent space
    # So we can plot a 2d graph with a sample from the distribution of the latent space

    # 12 colors
    colors = ["red", "green", "blue", "yellow", "black", "orange", "purple", "pink", "brown", "gray", "cyan", "magenta"]

    # Sample the distribution for each sample
    for i, sample in enumerate(data):
        x_arr, y_arr = [], []
        for j in range(100):
            encoded_sample = vae.encode_sample(sample)
            x, y = encoded_sample[0], encoded_sample[1]
            x_arr.append(x)
            y_arr.append(y)


        plt.scatter(x_arr, y_arr, c=colors[i % len(colors)])


    plt.title("Latent space")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig("latent_space_100.png")

def plot_latent_space_2d_point(vae, data):
    plt.figure()
    # In a vae, the output of the encoder is the mean and std of the latent space
    # So we can plot a 2d graph with a sample from the distribution of the latent space

    # 12 colors
    colors = ["red", "green", "blue", "yellow", "black", "orange", "purple", "pink", "brown", "gray", "cyan", "magenta"]

    # Sample the distribution for each sample
    for i, sample in enumerate(data):
        encoded_sample = vae.encode_sample(sample)
        x, y = encoded_sample[0], encoded_sample[1]

        plt.scatter(x, y, c=colors[i % len(colors)])


    plt.title("Latent space")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig("latent_space_1.png")

def decode_latent_space_1d(vae, data, shape):

    encodings = []
    for i, sample in enumerate(data):
        encoded_sample = vae.encode_sample(sample)
        encodings.append(encoded_sample)

    min_x = min(encodings)
    max_x = max(encodings)

    steps = 12

    x = np.linspace(min_x, max_x, steps)

    decoded_samples = []
    for i in range(steps):
        encoded_sample = np.array([x[i]])
        decoded_sample = vae.decode_sample(encoded_sample)
        decoded_sample = decoded_sample.reshape(shape)
        decoded_samples.append(decoded_sample)

    image_dim = (steps, 1)

    create_image(decoded_samples, "decoded_samples.png", image_dim)

def decode_latent_space_2d(vae, data, shape):

    encodings = []
    for i, sample in enumerate(data):
        encoded_sample = vae.encode_sample(sample)
        encodings.append(encoded_sample)

    min_x = min([x[0] for x in encodings])
    max_x = max([x[0] for x in encodings])

    min_y = min([x[1] for x in encodings])
    max_y = max([x[1] for x in encodings])

    print(f"x: {min_x} - {max_x}")
    print(f"y: {min_y} - {max_y}")

    steps = len(data)

    x = np.linspace(min_x, max_x, steps)
    y = np.linspace(min_y, max_y, steps)

    # First show the values of each step in a plot
    # As text (x, y)
    for i in reversed(range(steps)):
        for j in range(steps):
            # 3 decimals, no line break
            print(f"({x[j]:.3f}, {y[i]:.3f})", end=" ")
        print()



    decoded_samples = []
    for i in reversed(range(steps)):
        for j in range(steps):
            encoded_sample = np.array([x[j], y[i]])
            decoded_sample = vae.decode_sample(encoded_sample)
            decoded_sample = decoded_sample.reshape(shape)
            decoded_samples.append(decoded_sample)

    image_dim = (steps, steps)

    create_image(decoded_samples, "decoded_samples.png", image_dim)



        


if __name__ == "__main__":
    main()

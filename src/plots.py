from typing import List, Tuple
import matplotlib.pyplot as plt


def plot_errors_per_epoch(errors_per_epoch: List[float]):
    plt.plot(errors_per_epoch)
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    plt.yticks([i for i in range(0, 10, 1)])

    x_ticks = [i for i in range(0, len(errors_per_epoch), 5000)]
    plt.xticks(x_ticks)

    # Zoom in
    plt.ylim(0, 10)

    plt.savefig("errors_per_epoch.png")

    plt.figure()

    plt.plot(errors_per_epoch)
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    plt.yticks([i for i in range(0, 10)])
    plt.xticks(x_ticks)

    # Line at 1
    plt.axhline(y=1, color="r", linestyle="-")

    # Zoom in
    plt.ylim(0, 10)

    plt.savefig("errors_per_epoch_zoom.png")


def plot_errors_per_architecture(
    errors_per_architecture: dict[str, Tuple[float, float]]
):
    means = [error[0] for error in errors_per_architecture.values()]
    stds = [error[1] for error in errors_per_architecture.values()]

    plt.bar(
        [i + 1 for i in range(len(errors_per_architecture))],
        means,
        yerr=stds,
        capsize=5,
    )

    names = list(errors_per_architecture.keys())

    plt.xticks(
        [i + 1 for i in range(len(errors_per_architecture))],
        names,
    )

    plt.xlabel("Architecture")
    plt.ylabel("Error")

    plt.title("Mean Error per architecture (10 iterations)")

    plt.savefig("errors_per_architecture.png")
    plt.figure()


def plot_errors_per_optimization_method(
    errors_per_optimization_method: dict[str, Tuple[float, float]]
):
    means = [error[0] for error in errors_per_optimization_method.values()]
    stds = [error[1] for error in errors_per_optimization_method.values()]

    plt.bar(
        [i + 1 for i in range(len(errors_per_optimization_method))],
        means,
        yerr=stds,
        capsize=5,
    )

    names = list(errors_per_optimization_method.keys())

    plt.xticks(
        [i + 1 for i in range(len(errors_per_optimization_method))],
        names,
    )

    plt.xlabel("Optimization Method")
    plt.ylabel("Error")

    plt.title("Mean Error per optimization method (10 iterations)")

    plt.savefig("errors_per_optimization_method.png")
    plt.figure()

from typing import List
import matplotlib.pyplot as plt


def plot_errors_per_epoch(errors_per_epoch: List[float]):
    plt.plot(errors_per_epoch)
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    plt.yticks([i for i in range(0, 100, 10)])

    x_ticks = [i for i in range(0, len(errors_per_epoch), 10000)]

    # Zoom in
    plt.ylim(0, 100)

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

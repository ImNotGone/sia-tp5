from typing import List
import matplotlib.pyplot as plt


def plot_errors_per_epoch(errors_per_epoch: List[float]):
    plt.plot(errors_per_epoch)
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    min_error = min(errors_per_epoch)
    max_error = max(errors_per_epoch)
    step = int((max_error - min_error) / 10)

    ticks = [i for i in range(int(min_error) + step, int(max_error) - step, step)]
    ticks.append(max_error)
    ticks.insert(0, min_error)
    print(ticks)
    plt.yticks(ticks)

    x_ticks = [i for i in range(0, len(errors_per_epoch), 10000)]
    x_ticks.append(len(errors_per_epoch))

    plt.savefig("errors_per_epoch.png")

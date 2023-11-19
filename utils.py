from typing import Any, Dict
def get_batch_size(config: Dict[str, Any], data_size: int) -> int:
    strategy = config["training_strategy"]

    if strategy == "batch":
        return data_size
    elif strategy == "mini_batch":
        batch_size = config["batch_size"]

        if batch_size > data_size:
            raise ValueError("Batch size must be smaller than the dataset size")

        return batch_size
    elif strategy == "online":
        return 1
    else:
        raise ValueError("Invalid training strategy")


def pretty_print_font(bitmap):
    delta = 0.1
    for row in bitmap:
        for pixel in row:
            if pixel < delta:
                print(" ", end="")
            else:
                print("█", end="")
        print()
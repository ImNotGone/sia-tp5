from typing import Any, Dict, List
import json
import numpy as np


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


def serialize_weights(weights, path="weights.json"):
    # Convert the weights to a list of lists
    weights = [[w.tolist() for w in layer] for layer in weights]

    with open(path, "w") as f:
        json.dump(weights, f)


def deserialize_weights(path="weights.json") -> List[np.ndarray]:
    with open(path, "r") as f:
        weights = json.load(f)

        # Convert the weights to numpy arrays
        weights = [[np.array(w) for w in layer] for layer in weights]
        weights = [np.array(layer) for layer in weights]

        return weights

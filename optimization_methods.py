import numpy as np

from typing import List, Callable
from numpy._typing import NDArray


def gradient_descent(
    weight_delta: List[NDArray], learning_rate: float
) -> List[NDArray]:
    return [learning_rate * delta for delta in weight_delta]


previous_weight_delta = None


def momentum(
    weight_delta: List[NDArray], learning_rate: float, momentum: float
) -> List[NDArray]:
    global previous_weight_delta

    if previous_weight_delta is None:
        previous_weight_delta = [np.zeros_like(delta) for delta in weight_delta]

    weight_delta = [
        learning_rate * delta + momentum * previous_delta
        for delta, previous_delta in zip(weight_delta, previous_weight_delta)
    ]

    previous_weight_delta = weight_delta

    return weight_delta


# ----- optimization method generator -----
OptimizationMethod = Callable[[List[NDArray]], List[NDArray]]


def get_optimization_method(config) -> OptimizationMethod:
    learning_rate = config["learning_rate"]
    momentum_parameter = config["momentum"]

    optimization_method = config["method"]

    if optimization_method == "gradient_descent":
        return lambda weight_delta: gradient_descent(weight_delta, learning_rate)
    elif optimization_method == "momentum":
        return lambda weight_delta: momentum(
            weight_delta, learning_rate, momentum_parameter
        )
    else:
        raise Exception("Optimization method not found")

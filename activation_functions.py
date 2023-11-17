from collections.abc import Callable
from typing import Tuple
import numpy as np
from numpy._typing import NDArray


# ----- Identity activation function -----


def identity(x: NDArray) -> NDArray:
    return x


def identity_derivative(x: NDArray) -> NDArray:
    return np.ones(x.shape)


def identity_normalize(x: NDArray | float) -> NDArray | float:
    return x


# ----- Logistic activation function -----


def logistic(x: NDArray, beta: float) -> NDArray:
    return 1 / (1 + np.exp(-2 * beta * x))


def logistic_derivative(x: NDArray, beta: float) -> NDArray:
    return 2 * beta * logistic(x, beta) * (1 - logistic(x, beta))


def logistic_normalize(x: NDArray | float) -> NDArray | float:
    if isinstance(x, float):
        return x / abs(x)

    max_val = max(x)
    min_val = min(x)
    return (x - min_val) / (max_val - min_val)


# ----- Hyperbolic tangent activation function -----


def tanh(x: NDArray, beta: float) -> NDArray:
    return np.tanh(beta * x)


def tanh_derivative(x: NDArray, beta: float) -> NDArray:
    return beta * (1 - np.power(tanh(x, beta), 2))


def tanh_normalize(x: NDArray | float) -> NDArray | float:
    if isinstance(x, float):
        return x / abs(x)

    max_val = max(abs(x))
    return x / max_val


# ----- relu activation function -----


def relu(x: NDArray) -> NDArray:
    return np.maximum(0, x)


def relu_derivative(x: NDArray) -> NDArray:
    return np.where(x > 0, 1, 0)


# ----- activation function generator -----
ActivationFunction = Callable[[NDArray | float], NDArray | float]


def get_activation_function(
    activation_function: str, beta: float = 1
) -> Tuple[ActivationFunction, ActivationFunction, ActivationFunction]:
    if activation_function == "logistic":
        return (
            lambda x: logistic(x, beta),
            lambda x: logistic_derivative(x, beta),
            logistic_normalize,
        )
    elif activation_function == "tanh":
        return (
            lambda x: tanh(x, beta),
            lambda x: tanh_derivative(x, beta),
            tanh_normalize,
        )
    # elif activation_function == "relu":
         # return relu, relu_derivative
    elif activation_function == "identity":
        return identity, identity_derivative, identity_normalize
    else:
        raise Exception("Activation function not found")

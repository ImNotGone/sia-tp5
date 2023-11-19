import numpy as np

from typing import List, Callable
from numpy._typing import NDArray

time=0
m=None
v=None

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


def adam(
    weight_delta: List[NDArray],
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    epoch: int = 1,
    t: int = 0,
) -> List[NDArray]:
    #Config for error aprox 1, solo vi un pixel de error me parece
    #"method": "adam",
    #        "learning_rate": 0.0005,
    #        "momentum": 0.9,
    #        "beta1":0.9,
    #        "beta2":0.999,
    #        "epsilon":1e-8,
    #"hidden_layers": [35, 20 , 10],
    #    "latent_space_size": 2,
    #    "target_error": 0.001,
    #    "max_epochs": 120000,
    global time
    global v
    global m
    if m is None:
        m = [np.zeros_like(delta) for delta in weight_delta]
    if v is None:
        v = [np.zeros_like(delta) for delta in weight_delta]

    updated_weight_delta = []
    time+=1
    for i in range(len(weight_delta)):
        delta= weight_delta[i]
        m[i] = beta1 * m[i] + (1 - beta1) * delta
        v[i] = beta2 * v[i] + (1 - beta2) * (delta ** 2)
        m_hat = m[i] / (1 - beta1 ** epoch)
        v_hat = v[i] / (1 - beta2 ** epoch)
        updated_weight_delta.append(learning_rate * m_hat / (np.sqrt(v_hat) + epsilon))
    return updated_weight_delta


# ----- optimization method generator -----
OptimizationMethod = Callable[[List[NDArray]], List[NDArray]]


def get_optimization_method(config) -> OptimizationMethod:
    learning_rate = config["learning_rate"]
    momentum_parameter = config["momentum"]
    beta1= config["beta1"]
    beta2= config["beta2"]
    epsilon= config["epsilon"]

    optimization_method = config["method"]

    if optimization_method == "gradient_descent":
        return lambda weight_delta, epoch: gradient_descent(weight_delta, learning_rate)
    elif optimization_method == "adam":
        return lambda weight_delta, epoch: adam(
        weight_delta, learning_rate, beta1, beta2, epsilon, epoch
    )
    elif optimization_method == "momentum":
        return lambda weight_delta, epoch: momentum(
            weight_delta, learning_rate, momentum_parameter
        )
    else:
        raise Exception("Optimization method not found")

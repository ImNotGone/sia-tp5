from numpy._typing import NDArray
import numpy as np

def flipping_noise(data: NDArray, noise_probability: float):
    noisy_data = data.copy()
    for sample in noisy_data:
        for i in range(sample.shape[0]):
            if np.random.uniform() < noise_probability:
                sample[i] = 1 - sample[i]
    return noisy_data
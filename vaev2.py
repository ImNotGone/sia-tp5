import copy

import pickle
from typing import List

import numpy as np
from numpy import ndarray
from tqdm import tqdm


from abc import ABC
from numpy import ndarray


class ActivationMethod(ABC):
    def evaluate(self, x: ndarray[float]) -> ndarray[float]:
        raise NotImplementedError()

    def d_evaluate(self, x: ndarray[float]) -> ndarray[float]:
        raise NotImplementedError()


class StepActivationFunction(ActivationMethod):
    def evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return np.where(x >= 0, 1, -1)

    def d_evaluate(self, x: ndarray[float]) -> ndarray[float]:
        # NOTE: we return ones even if it is not the derivative to be able to generalize the perceptron
        return np.ones_like(x)


class IdentityActivationFunction(ActivationMethod):
    def evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return x

    def d_evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return np.ones_like(x)


class TangentActivationFunction(ActivationMethod):
    def __init__(self, beta: float):
        self._beta = beta

    def evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return np.tanh(self._beta * x)

    def d_evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return self._beta * (1 - self.evaluate(x) ** 2)

    def limits(self) -> tuple[float, float]:
        return -1, 1


class SigmoidActivationFunction(ActivationMethod):
    def evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return 1 / (1 + np.exp(-x))

    def d_evaluate(self, x: ndarray[float]) -> ndarray[float]:
        ans = self.evaluate(x)
        return ans * (1 - ans)

    def limits(self) -> tuple[float, float]:
        return 0, 1


class LogisticActivationFunction(ActivationMethod):
    def __init__(self, beta: float):
        self._beta = beta

    def evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return 1 / (1 + np.exp(-2 * self._beta * x))

    def d_evaluate(self, x: ndarray[float]) -> ndarray[float]:
        result = self.evaluate(x)
        return 2 * self._beta * result * (1 - result)

    def limits(self) -> tuple[float, float]:
        return 0, 1

class CutCondition(ABC):
    def is_finished(self, errors: ndarray[float]) -> bool:
        raise NotImplementedError()


class FalseCutCondition(CutCondition):
    def is_finished(self, errors: ndarray[float]) -> bool:
        return False


class AccuracyCutCondition(CutCondition):
    def is_finished(self, errors: ndarray[float]) -> bool:
        result = np.count_nonzero(np.logical_not(errors)) == len(errors)

        return result


class AbsoluteValueCutCondition(CutCondition):
    def is_finished(self, errors) -> bool:
        return np.sum(np.abs(errors)) == 0


class MSECutCondition(CutCondition):
    def __init__(self, eps: float = 0.01):
        self._eps = eps

    def is_finished(self, errors) -> bool:
        return np.average(errors ** 2) < self._eps


class OneWrongPixelCutCondition(CutCondition):
    def is_finished(self, errors: ndarray[float]) -> bool:
        for row in errors:
            if len(row) - np.count_nonzero(np.isclose(row, 0, atol=0.01)) > 1:
                return False

        return True

class OptimizationMethod(ABC):
    def __init__(self, learning_rate=0.1):
        self._learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    def adjust(self, gradient: ndarray[float], index: int, epoch: int):
        raise NotImplementedError()


class GradientDescentOptimization(OptimizationMethod):
    def adjust(self, gradient: ndarray[float], _, __) -> ndarray[float]:
        return - self._learning_rate * gradient


class MomentumOptimization(OptimizationMethod):
    def __init__(self, alpha=0.3, learning_rate=0.1):
        super().__init__(learning_rate)
        self._alpha = alpha
        self._prev = []

    def adjust(self, gradient: ndarray[float], index: int, _) -> ndarray[float]:
        while index >= len(self._prev):
            self._prev.append(0)

        self._prev[index] = - self._learning_rate * gradient + self._alpha * self._prev[index]

        return self._prev[index]


class AdamOptimization(OptimizationMethod):
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__()
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._momentum = []
        self._rmsProp = []

    def adjust(self, gradient: ndarray[float], index: int, epoch: int):
        assert len(self._momentum) == len(self._rmsProp)

        while index >= len(self._momentum):
            self._momentum.append(0)
            self._rmsProp.append(0)

        self._momentum[index] = self._beta_1 * self._momentum[index] + (1 - self._beta_1) * gradient
        self._rmsProp[index] = self._beta_2 * self._rmsProp[index] + (1 - self._beta_2) * np.power(gradient, 2)

        m = np.divide(self._momentum[index], 1 - self._beta_1 ** (epoch + 1))
        v = np.divide(self._rmsProp[index], 1 - self._beta_2 ** (epoch + 1))

        return -self._alpha * np.divide(m, (np.sqrt(v) + self._epsilon))

def gradient(delta: ndarray[float], data: ndarray[float]) -> ndarray[float]:
    return - np.dot(data.T, delta)

class Layer:
    def __init__(self, neurons: ndarray[float]):
        self._neurons = neurons

    @property
    def neurons(self) -> ndarray[float]:
        return self._neurons

    @neurons.setter
    def neurons(self, new_neurons: ndarray[float]):
        self._neurons = new_neurons

def mse(errors: ndarray[float]) -> ndarray:
    return np.mean(errors ** 2)

class MultiLayerPerceptron:
    def __init__(self, architecture: List[int], epochs: int, cut_condition: CutCondition,
                 activation_method: ActivationMethod, optimization_method: OptimizationMethod):

        self._epochs = epochs
        self._cut_condition = cut_condition
        self._activation_function = activation_method
        self._optimization_method = optimization_method

        # Initialize weights for the whole network with random [-1,1] values.
        self._layers = []
        for i in range(len(architecture) - 1):
            self._layers.append(Layer(np.random.uniform(-1, 1, (architecture[i] + 1, architecture[i + 1]))))

        self._feedforward_data = []
        self._feedforward_output = []

    def predict(self, data: ndarray[float]) -> ndarray[float]:
        results = data
        for i in range(len(self._layers)):
            results = np.insert(np.atleast_2d(results), 0, 1, axis=1)
            # results = mu x hidden_size + 1, #layers[i] = (hidden_size + 1) x next_hidden_size
            h = np.dot(results, self._layers[i].neurons)
            # h = mu x next_hidden_size
            results = self._activation_function.evaluate(h)

        return results

    def feedforward(self, data: ndarray[float]) -> ndarray[float]:
        self._feedforward_data = [data]
        results = data
        self._feedforward_output = []
        for i in range(len(self._layers)):
            results = np.insert(results, 0, 1, axis=1)
            self._feedforward_output.append(results)
            # results = mu x hidden_size + 1, #layers[i] = (hidden_size + 1) x next_hidden_size
            h = np.dot(results, self._layers[i].neurons)
            # h = mu x next_hidden_size
            self._feedforward_data.append(h)
            results = self._activation_function.evaluate(h)

        return results

    def backpropagation(self, error: ndarray[float]) -> tuple[list[ndarray[float]], ndarray[float]]:
        derivatives = self._activation_function.d_evaluate(self._feedforward_data[-1])  # mu * output_size
        delta_i = error * derivatives  # mu * output_size, elemento a elemento

        # #delta_i = mu * output_size
        # #feedforward_output[-1] = #hidden_data = mu * (hidden_size + 1)
        gradients = [gradient(delta_i, self._feedforward_output[-1])]
        # #gradients =  (#hidden_size + 1) * #output_size

        for i in reversed(range(len(self._layers) - 1)):
            # delta_w tiene que tener la suma de todos los delta_w para cada iteracion para ese peso
            #        mu * output_size  *   ((hidden_size + 1 {bias_layer} - 1) * output_size).T
            error = np.dot(delta_i, np.delete(self._layers[i + 1].neurons, 0, axis=0).T)
            # mu * (hidden_size + 1 {bias_layer} - 1)  == mu * hidden_size

            # Call _optimization_method #
            derivatives = self._activation_function.d_evaluate(self._feedforward_data[i + 1])  # mu * hidden_size
            delta_i = error * derivatives  # mu * hidden_size
            # #feedforward[i] = mu * (previous_hidden_size + 1) ; delta_i = mu * hidden_size
            gradients.append(gradient(delta_i, self._feedforward_output[i]))
            # Me libero del mu (estoy "sumando" todos los delta_w)

        gradients.reverse()
        return gradients, delta_i

    def update_weights(self, gradients: list[ndarray[float]], epoch: int):
        for i in range(len(self._layers)):
            delta_w = self._optimization_method.adjust(gradients[i], i, epoch)
            self._layers[i].neurons = np.add(self._layers[i].neurons, delta_w)

    def train_batch(self, data: ndarray[float], expected: ndarray[float]) -> list[ndarray[float]]:
        # #initial_data = mu x initial_size, #expected = mu x output_size
        error_history = []

        for epoch in tqdm(range(self._epochs)):
            results = self.feedforward(data)


            error = expected - results  # mu * output_size
            # ver calculo del error con llamando a d_error #
            error_history.append(mse(error))
            if self._cut_condition.is_finished(error):
                break

            gradients, _ = self.backpropagation(error)

            # Calculo w = w + dw
            self.update_weights(gradients, epoch)

        return error_history

    def save(self, file_name: str):
        with open(file_name, "wb") as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def load(file_name: str):
        with open(file_name, "rb") as infile:
            return pickle.load(infile)

def loss_function(mean, std, data, result):
    rec = 0.5 * np.mean((data - result) ** 2)
    kl = -0.5 * np.sum(1 + std - mean ** 2 - np.exp(std))

    return rec + kl

def reparametrization_trick(mean: ndarray[float], std: ndarray[float]) -> tuple[ndarray[float], float]:
    eps = np.random.standard_normal()
    return eps * std + mean, eps


class VariationalAutoencoder:
    def __init__(self, input_size: int, latent_size: int, epochs: int,
                 encoder_architecture: list[int],
                 decoder_architecture: list[int],
                 activation_method: ActivationMethod,
                 optimization_method: OptimizationMethod):
        self._epochs = epochs

        self._input_size = input_size
        self._latent_size = latent_size
        self._last_delta_size = decoder_architecture[0]

        cut_condition = FalseCutCondition()

        encoder_architecture.insert(0, input_size)
        encoder_architecture.append(2 * latent_size)
        self._encoder = MultiLayerPerceptron(encoder_architecture, epochs, cut_condition, activation_method,
                                             copy.deepcopy(optimization_method))

        decoder_architecture.insert(0, latent_size)
        decoder_architecture.append(input_size)
        self._decoder = MultiLayerPerceptron(decoder_architecture, epochs, cut_condition, activation_method,
                                             copy.deepcopy(optimization_method))

    def train(self, data: ndarray[float]) -> list[float]:
        assert data.shape[1] == self._input_size

        loss_history = []
        for epoch in tqdm(range(self._epochs)):
            # NOTE: Feedforward
            result = self._encoder.feedforward(data)

            mean = result[:, :result.shape[1] // 2]
            std = result[:, result.shape[1] // 2:]

            z, eps = reparametrization_trick(mean, std)

            result = self._decoder.feedforward(z)

            loss = loss_function(mean, std, data, result)
            loss_history.append(loss)
            if loss < 0.01:
                break

            # NOTE: Decoder Backpropagation for reconstruction
            dL_dX = data - result
            decoder_gradients, last_delta = self._decoder.backpropagation(dL_dX)

            # NOTE: Encoder backpropagation for reconstruction
            dz_dm = np.ones([self._last_delta_size, self._latent_size])
            dz_dv = eps * np.ones([self._last_delta_size, self._latent_size])
            mean_error = np.dot(last_delta, dz_dm)
            std_error = np.dot(last_delta, dz_dv)
            encoder_reconstruction_error = np.concatenate((mean_error, std_error), axis=1)
            encoder_reconstruction_gradients, _ = self._encoder.backpropagation(encoder_reconstruction_error)

            # NOTE: Encoder backpropagation for regularization
            dL_dm = mean
            dL_dv = 0.5 * (np.exp(std) - 1)
            encoder_loss_error = np.concatenate((dL_dm, dL_dv), axis=1)
            encoder_loss_gradients, _ = self._encoder.backpropagation(encoder_loss_error)

            # NOTE: update weights with gradients
            encoder_gradients = []
            for g1, g2 in zip(encoder_loss_gradients, encoder_reconstruction_gradients):
                encoder_gradients.append(g1 + g2)

            self._encoder.update_weights(encoder_gradients, epoch)
            self._decoder.update_weights(decoder_gradients, epoch)

        return loss_history

    def predict(self, z: ndarray[float]) -> ndarray[float]:
        assert z.shape[1] == self._latent_size

        return self._decoder.feedforward(z)

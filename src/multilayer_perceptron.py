from typing import List, Tuple
import numpy as np
from numpy._typing import NDArray
import random
import copy
import signal

from src.activation_functions import ActivationFunction
from src.optimization_methods import OptimizationMethod
from src.utils import signal_handler, stop


signal.signal(signal.SIGINT, signal_handler)

class NeuronLayer:
    def __init__(self,
                 previous_layer_neuron_amount,
                 current_layer_neurons_amount,
                 activation_function,
                 lower_weight,
                 upper_weight,
                 optimizer: OptimizationMethod
                 ):

        self.excitement = None
        self.output = None
        self.activation_function = activation_function

        self.optimizer = optimizer

        # Se genera una matrix de (current_layer_neurons_amount x previous_layer_neuron_amount)
        weights = []
        for i in range(current_layer_neurons_amount):
            weights.append([])
            for j in range(previous_layer_neuron_amount):
                weights[i].append(random.uniform(lower_weight, upper_weight))
        self.weights = np.array(weights)


    def compute_activation(self, prev_input):

        # guardamos el dot producto dado que lo vamos a usar aca y en el backpropagation
        self.excitement = np.dot(self.weights, prev_input)

        self.output = self.activation_function(self.excitement)

        return self.output  # Se ejecuta la funcion sobre cada elemento del arreglo

    def update_weights(self, delta_w):
        self.weights += self.optimizer(delta_w)



def multilayer_perceptron(
    data: List[Tuple[NDArray, NDArray]],
    hidden_layer_sizes: List[int],
    output_layer_size: int,
    target_error: float,
    max_epochs: int,
    batch_size: int,
    neuron_activation_function: ActivationFunction,
    neuron_activation_function_derivative: ActivationFunction,
    optimization_method: OptimizationMethod,
) -> Tuple[List[NDArray], List[float]]:
    # Initialize weights
    current_network = initialize_weights(
        hidden_layer_sizes, output_layer_size, len(data[0][0])
    )

    errors_in_epoch = []

    best_error = np.Infinity
    best_network = None
    epoch = 0
    percent = 0

    while best_error > target_error and epoch < max_epochs:
        # Get a random training set
        training_set = random.sample(data, batch_size)

        # For each training set
        weight_delta: List[NDArray] = []
        # error = 0.0

        for input, expected_output in training_set:
            # Propagate the input through the network
            neuron_activations, neuron_excitements = forward_propagation(
                input, current_network, neuron_activation_function
            )

            # Compute the error
            # error += compute_error(neuron_activations[-1], expected_output)

            # Calculate the weight delta
            current_weight_delta = backpropagation(
                neuron_activations,
                neuron_excitements,
                expected_output,
                current_network,
                neuron_activation_function_derivative,
                input,
            )

            # Add the weight delta to the total weight delta
            if len(weight_delta) == 0:
                weight_delta = current_weight_delta
            else:
                for i in range(len(weight_delta)):
                    weight_delta[i] += current_weight_delta[i]

        # Average the weight delta and apply the optimization method
        weight_delta = [delta / batch_size for delta in weight_delta]
        weight_delta = optimization_method(weight_delta, epoch + 1)

        update_weights(current_network, weight_delta)

        # If we have a better network, save it
        new_error = 0.0
        for input, expected_output in data:
            new_error += compute_error(
                predict(input, current_network, neuron_activation_function),
                expected_output,
            )

        errors_in_epoch += [new_error]

        if new_error < best_error:
            best_error = new_error
            best_network = copy.deepcopy(current_network)

        # Each 5% of the epochs, print progress and error
        if epoch % (max_epochs / 20) == 0:
            print(
                f"Epoch {epoch} - {percent}% - Error: {new_error} - Best: {best_error} - Target: {target_error}"
            )
            percent += 5

        epoch += 1

        if stop():
            break

    return best_network, errors_in_epoch


# Initialize weights
# Hidden layer sizes is an array with the number of neurons in each layer
# Output layer is the number of neurons in the output layer
def initialize_weights(
    hidden_layer_sizes: List[int],
    output_layer_size: int,
    input_layer_size: int,
) -> List[np.ndarray]:
    weights = []

    for i in range(len(hidden_layer_sizes)):
        # Generate random weights for each layer
        # if first layer
        if i == 0:
            weights += [
                np.random.uniform(-1, 1, (hidden_layer_sizes[i], input_layer_size))
            ]
        else:
            weights += [
                np.random.uniform(
                    -1, 1, (hidden_layer_sizes[i], hidden_layer_sizes[i - 1])
                )
            ]

    # add output layer
    weights += [np.random.uniform(-1, 1, (output_layer_size, hidden_layer_sizes[-1]))]
    return weights


def compute_error(output: NDArray, expected_output: NDArray) -> float:
    return np.sum(np.power(output - expected_output, 2)) / 2


def update_weights(weights: List[NDArray], weight_delta: List[NDArray]):
    for i in range(len(weights)):
        weights[i] += weight_delta[i]


def forward_propagation(
    input: NDArray,
    weights: List[NDArray],
    neuron_activation_function: ActivationFunction,
) -> Tuple[List[NDArray], List[NDArray]]:
    # Propagate the input through the network
    neuron_activations = []
    neuron_excitements = []

    # Propagate the input through the network
    previous_layer_output = input
    for i in range(len(weights)):
        # Calculate the neuron excitement
        # h^m = W^m * V^m-1 (MxN * Nx1 = Mx1)
        neuron_excitement = np.dot(weights[i], previous_layer_output)
        neuron_excitements += [neuron_excitement]

        # Calculate the neuron activation
        # V^m = θ(h^m)
        neuron_activation = neuron_activation_function(neuron_excitement)
        neuron_activations += [neuron_activation]

        # Set the previous layer output to the current layer activation
        previous_layer_output = neuron_activation

    return neuron_activations, neuron_excitements


# Uso de diccionarios para guardar los deltas. Ver si esta bien.
# Me parece que Weight_deltas se guarda desordenado
def backpropagation(
    neuron_activations: List[NDArray],
    neuron_excitements: List[NDArray],
    expected_output: NDArray,
    network: List[NDArray],
    neuron_activation_function_derivative: ActivationFunction,
    input: NDArray,
) -> List[NDArray]:
    # Calculate the output layer delta
    # δ^f = θ'(h) * (ζ- V^f)
    output_layer_delta = neuron_activation_function_derivative(
        neuron_excitements[-1]
    ) * (expected_output - neuron_activations[-1])

    # Calculate the weight delta for the output layer
    # ΔW^m = η * δ^m * (V^m-1)
    output_layer_weight_delta = np.dot(
        output_layer_delta.reshape(-1, 1), neuron_activations[-2].reshape(1, -1)
    )

    # Calculate the hidden layer deltas
    hidden_layer_deltas: List[NDArray] = [output_layer_delta]
    hidden_layer_weight_deltas: List[NDArray] = [output_layer_weight_delta]

    # Calculate the hidden layer deltas
    for i in range(len(network) - 2, -1, -1):
        # δ^m = θ'(h^m) * (W^m+1)^T * δ^m+1
        hidden_layer_delta = neuron_activation_function_derivative(
            neuron_excitements[i]
        ) * np.dot(network[i + 1].T, hidden_layer_deltas[0])

        hidden_layer_deltas = [hidden_layer_delta] + hidden_layer_deltas

        # Calculate the weight delta for the hidden layer
        # ΔW^m = η * δ^m * (V^m-1)
        previous_layer_activation = (
            input if i == 0 else neuron_activations[i - 1]
        ).reshape(-1, 1)

        hidden_layer_weight_delta = np.dot(
            hidden_layer_delta.reshape(-1, 1), previous_layer_activation.reshape(1, -1)
        )

        hidden_layer_weight_deltas = [
            hidden_layer_weight_delta
        ] + hidden_layer_weight_deltas

    return hidden_layer_weight_deltas


def get_hidden(
    input: NDArray,
    network: List[NDArray],
    neuron_activation_function: ActivationFunction,
    layer_number: int,
) -> NDArray:
    # Propagate the input through the network
    neuron_activations = []
    neuron_excitements = []

    # Propagate the input through the network
    previous_layer_output = input
    for i in range(layer_number):
        # print(len(network[i]))
        # Calculate the neuron excitement
        # h^m = W^m * V^m-1 (MxN * Nx1 = Mx1)
        neuron_excitement = np.dot(network[i], previous_layer_output)
        neuron_excitements += [neuron_excitement]

        # Calculate the neuron activation
        # V^m = θ(h^m)
        neuron_activation = neuron_activation_function(neuron_excitement)
        neuron_activations += [neuron_activation]

        # Set the previous layer output to the current layer activation
        previous_layer_output = neuron_activation

    return neuron_activation


def predict(
    input: NDArray,
    network: List[NDArray],
    neuron_activation_function: ActivationFunction,
) -> NDArray:
    neuron_activations, neuron_excitements = forward_propagation(
        input, network, neuron_activation_function
    )

    return neuron_activations[-1]

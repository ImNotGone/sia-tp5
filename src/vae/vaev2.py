import numpy as np
import json

from src.multilayer_perceptron import NeuronLayer
from src.utils import stop

def reconstruction_loss(generated_output, expected_output):
    return 0.5 * sum(np.power(generated_output - expected_output, 2))


def kl(mean, std):
    return -0.5 * np.sum(1 + std - mean ** 2 - np.exp(std))


def total_loss(mean, std, generated_output, expected_outputs):
    return reconstruction_loss(generated_output, expected_outputs) + kl(mean,std)


def count_error(output, expected_output):
    incorrect_pixels = 0
    for i in range(len(output)):
        val = output[i]

        if round(val) != expected_output[i]:
            incorrect_pixels += 1
    return incorrect_pixels

def combine_delta_w(decoder, mid_delta_w, encoder, mid_delta_w2, encoder2):
    decoder.append(mid_delta_w + mid_delta_w2)

    for a, b in zip(encoder, encoder2):
        decoder.append(a + b)

    return decoder



class SamplingLayer:
    def compute_activation(self, prev_input):
        self.mean, self.std = np.array_split(prev_input, 2)

        self.epsilon = np.random.standard_normal(int(prev_input.shape[0] / 2)).reshape(-1, 1)

        self.output = self.mean + np.dot(self.std, self.epsilon)  # Equivalente a multiplicacion de matriz

        return self.output


class VAE:
    def __init__(self,
                 layer_configuration,
                 activation_function,
                 derivative_activation_function,
                 optimizer,
                 learning_rate
                 ):

        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function

        self.learning_rate = learning_rate

        # Caclculamos el rango de valores iniciales para los weights
        upper_weight = 1
        lower_weight = - upper_weight

        self.layers: [NeuronLayer] = []

        # Generamos encoder 
        for i in range(1, len(layer_configuration) - 1):
            prev = i - 1  

            # Generamos nueva capa con las dimensiones apropiadas
            self.layers.append(NeuronLayer(
                layer_configuration[prev],
                layer_configuration[i],
                self.activation_function,
                lower_weight,
                upper_weight,
                optimizer(*learning_rate)
            ))

        #   Generamos espacio latente 
        self.latent_idx = len(layer_configuration) - 1

        self.layers.append(
            # Mean and Standard deviation
            # Los generamos como 1 capa para facilitar las cuentas, luego lo dividiremos
            NeuronLayer(
                layer_configuration[-2],
                layer_configuration[-1] * 2,
                self.activation_function,
                lower_weight,
                upper_weight,
                optimizer(*learning_rate)
            ),
        )

        # Generamos el Sampling Layer
        self.layers.append(
            SamplingLayer()
        )

        #  Generamos decoder 
        for i in reversed(range(len(layer_configuration) - 1)):
            following = i + 1  

            # Generamos nueva capa con las dimensiones apropiadas
            self.layers.append(NeuronLayer(
                layer_configuration[following],
                layer_configuration[i],
                self.activation_function,
                lower_weight,
                upper_weight,
                optimizer(*learning_rate)
            ))

    def forward_propagation(self, input_data):
        current = input_data
        self.input = input_data

        for idx, layer in enumerate(self.layers):
            current = layer.compute_activation(current)

        return current

    def back_propagate_decoder(self, expected_output, generated_output):
        delta_w = []
        # Calculamos el delta W de la primera capa del decoder 

        prev_delta = (expected_output - generated_output) * self.derivative_activation_function(self.layers[-1].excitement)
        delta_w.append(prev_delta.reshape(-1, 1) @ np.transpose(self.layers[-2].output.reshape(-1, 1)))

        # Calculamos el delta W de las capas ocultas del decoder 
        for idx in range(len(self.layers) - 2, self.latent_idx, -1):
            delta = np.dot(prev_delta, self.layers[idx + 1].weights) * self.derivative_activation_function(self.layers[idx].excitement)
            delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.layers[idx - 1].output.reshape(-1, 1)))
            prev_delta = delta

        return delta_w, prev_delta

    def back_propagate_encoder(self, prev_delta):
        delta_w = []
        # Calculamos el delta W de las capas ocultas del encoder
        for idx in range(self.latent_idx - 2, 0, -1):
            delta =  np.dot(prev_delta, self.layers[idx + 1].weights) * self.derivative_activation_function(self.layers[idx].excitement)
            delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.layers[idx - 1].output.reshape(-1, 1)))
            prev_delta = delta

        if len(delta_w) < len(self.layers) - 1:
            #  Calculamos el delta W de la capa inicial 
            delta = np.dot(prev_delta, self.layers[1].weights) * self.derivative_activation_function(self.layers[0].excitement)
            delta_w.append(delta.reshape(-1, 1) @ np.transpose(self.input.reshape(-1, 1)))

        return delta_w

    def backward_propagation(self, expected_output, generated_output):

        # Backpropagation del decoder
        delta_w_decoder, prev_delta = self.back_propagate_decoder(expected_output, generated_output)

        # Reconstruction

        mean = np.dot(prev_delta, self.layers[self.latent_idx + 1].weights) * self.derivative_activation_function(self.layers[self.latent_idx].output)
        std = self.layers[self.latent_idx].epsilon.reshape(-1,) * mean
        prev_delta = np.concatenate((mean, std), axis=0)

        mid_delta_w = prev_delta.reshape(-1, 1) @ np.transpose(self.layers[self.latent_idx - 2].output.reshape(-1, 1))

        delta_w_encoder = self.back_propagate_encoder(prev_delta)

        # Regularization

        mean_loss = - self.layers[self.latent_idx].mean
        std_loss = - 0.5 * (np.exp(self.layers[self.latent_idx].std) - 1)
        prev_delta = np.concatenate((mean_loss, std_loss), axis=0)

        mid_delta_w2 = prev_delta.reshape(-1, 1) @ np.transpose(self.layers[self.latent_idx - 2].output.reshape(-1, 1))

        delta_w_encoder2 = self.back_propagate_encoder(prev_delta)

        # Combinamos ambos nuevos delta_w
        delta_w = combine_delta_w(delta_w_decoder, mid_delta_w, delta_w_encoder, mid_delta_w2, delta_w_encoder2)

        delta_w.reverse()

        return delta_w

    def update_all_weights(self, delta_w):
        for idx, layer in enumerate(self.layers):
            if idx != self.latent_idx:
                layer.update_weights(delta_w.pop(0))

    def encode(self, data):
        current = data
        self.input = data
        for i in range(self.latent_idx):
            current = self.layers[i].compute_activation(current)
        return current

    def encode_sample(self, data):
        current = data
        self.input = data
        for i in range(self.latent_idx):
            current = self.layers[i].compute_activation(current)

        # Latent space
        current = self.layers[self.latent_idx].compute_activation(current)
        return current

    def decode_sample(self, z):
        current = z
        for i in range(self.latent_idx + 1, len(self.layers)):
            current = self.layers[i].compute_activation(current)
        return current

    def decode(self, z):
        current = z
        for i in range(self.latent_idx, len(self.layers)):
            current = self.layers[i].compute_activation(current)
        return current

    def train(self, limit, train_data):
        i = 0
        min_error = float("inf")
        self.kl_loss = []
        self.r_loss = []
        self.total_loss = []

        best_weights = None

        while i < limit:

            if stop():
                return self.total_loss

            error = 0

            iteration_error = []
            iteration_kl = []
            iteration_r = []

            deltas = []
            for elem in train_data:
                result = self.forward_propagation(elem)
                delta_w = self.backward_propagation(elem, result)

                if len(deltas) == 0:
                    deltas = delta_w
                else:
                    for idx in range(len(deltas)):
                        deltas[idx] += delta_w[idx]
                

                kl_loss = kl(self.layers[self.latent_idx].mean, self.layers[self.latent_idx].std)
                r_loss = reconstruction_loss(result, elem)

                iteration_kl.append(kl_loss)
                iteration_r.append(r_loss)

                error = kl_loss + r_loss

                iteration_error.append(error)

            error = np.mean(iteration_error)
            kl_loss = np.mean(iteration_kl)
            r_loss = np.mean(iteration_r)

            self.total_loss.append(error)
            self.kl_loss.append(kl_loss)
            self.r_loss.append(r_loss)

            # Mean of all deltas
            weight_delta = [delta / len(train_data) for delta in deltas]

            self.update_all_weights(weight_delta)
            

            if error < min_error:
                min_error = error
                min_kl = kl_loss
                min_r = r_loss
                best_weights = self.layers.copy()

            # Each 5%
            if i % (limit / 20) == 0:
                print(f"Epoch: {i}, Min error: {min_error}")
                print(f"KL: {min_kl}, R: {min_r}")

            i += 1
            
        self.layers = best_weights


        return self.total_loss

    def save_weights(self):
        encoder = self.layers[:self.latent_idx]
        decoder = self.layers[self.latent_idx + 1:]

        encoder_weights = []
        for layer in encoder:
            encoder_weights.append(layer.weights.tolist())

        decoder_weights = []
        for layer in decoder:
            decoder_weights.append(layer.weights.tolist())

        with open("weights.json", "w") as file:
            json.dump({
                "encoder": encoder_weights,
                "decoder": decoder_weights
            }, file)

    def load_weights(self):
        with open("weights.json", "r") as file:
            weights = json.load(file)

            encoder_weights = weights["encoder"]
            decoder_weights = weights["decoder"]

            encoder = self.layers[:self.latent_idx]
            decoder = self.layers[self.latent_idx + 1:]

            for idx, layer in enumerate(encoder):
                layer.weights = np.array(encoder_weights[idx])

            for idx, layer in enumerate(decoder):
                layer.weights = np.array(decoder_weights[idx])

    
            
        


    

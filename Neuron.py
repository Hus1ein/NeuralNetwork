import numpy as np
from Helpers import sigmoid

""" 
    Neuron class
    - Every neuron object have 3 weights (One weight for every input) and 1 bias.
    - 'sum' variable represent (w1 * x1) + (w2 * x2) + (w3 * x3) + b
    - 'value' variable represent sigmoid(sum)
"""


class Neuron:

    def __init__(self):
        self.weights = [np.random.normal(), np.random.normal(), np.random.normal()]
        self.bias = np.random.normal()
        self.sum = 0
        self.value = 0

    def calculate_neuron_value(self, inputs):
        # (Weight * inputs), add bias, then use the activation function
        self.sum = np.dot(self.weights, inputs) + self.bias
        self.value = sigmoid(self.sum)

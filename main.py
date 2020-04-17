import numpy as np
from NeuralNetwork import NeuralNetwork
from Dataset import data, all_y_trues
from Helpers import mse_loss, rmse_loss, smape_loss

print('Enter learn rate:')
learn_rate = float(input())  # value for testing

print('This program have 3 performance functions, please select number of one function to continue: ')
print('1. Mean Square Error')
print('2. Root Mean Squared Error')
print('3. Symmetric Mean Absolute Percentage Error')
selected_perf_fun = int(input())


# Train our neural network
neural_network = NeuralNetwork()

if selected_perf_fun == 1:
    perf_fun = mse_loss
elif selected_perf_fun == 2:
    perf_fun = rmse_loss
else:
    perf_fun = smape_loss

neural_network.train(data, all_y_trues, learn_rate, perf_fun)

# Test out neural network
emily = np.array([7.1, 5.9, 2.1])
print(neural_network.feedforward(emily))  # [1, 0, 0]

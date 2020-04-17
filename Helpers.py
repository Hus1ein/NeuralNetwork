import numpy as np

# Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


"""
    Performance function
    - y_true and y_pred are numpy arrays of the same length.
"""

# Mean Square Error
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


# Root Mean Squared Error
def rmse_loss(y_true, y_pred):
    return np.sqrt(mse_loss(y_true, y_pred))


# Symmetric Mean Absolute Percentage Error
def smape_loss(y_true, y_pred):
    EPSILON = 1e-10
    return np.mean(2.0 * np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) + EPSILON))

import numpy as np

data = np.array([
    [5.1, 1.4, 0.2],  # setosa
    [4.9, 1.4, 0.2],  # setosa
    [4.7, 1.3, 0.2],  # setosa
    [4.6, 1.5, 0.2],  # setosa
    [7.0, 4.7, 1.4],  # versicolor
    [6.9, 4.9, 1.5],  # versicolor
    [6.4, 4.5, 1.5],  # versicolor
    [5.5, 4.0, 1.3],  # versicolor
    [7.9, 6.4, 2.0],  # virginica
    [7.4, 6.1, 1.9],  # virginica
    [5.8, 5.1, 1.9],  # virginica
    [7.1, 5.9, 2.1],  # virginica
])

all_y_trues = np.array([
  # se ve  vi
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1]
])


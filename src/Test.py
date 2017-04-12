import numpy as np

from bias_neural_network import BNN

rgb_shape = (32, 32, 3)
height_shape = (32, 32, 1)
aux_shape = (32, 32, 1)

rgb_inputs = np.array([1, 2, 3])
height_inputs = np.array([1, 2, 3])
aux_inputs = np.array([1, 2, 3])
inputs = [rgb_inputs, height_inputs, aux_inputs]

rgb_labes = np.array([1, 2, 3])
height_labes = np.array([1, 2, 3])
aux_labes = np.array([1, 2, 3])
labes = [rgb_labes, height_labes, aux_labes]
network = BNN(rgb_shape, height_shape, aux_shape, inputs, labes)

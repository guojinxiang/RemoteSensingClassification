import numpy as np
import scipy.io as sio

from cnn import CNN

train_set_1 = sio.loadmat('../resources/training_instance_matrix.mat')['training_instance_matrix']
label_set_1 = sio.loadmat('../resources/training_label_vector.mat')['training_label_vector']
train_set_2 = sio.loadmat('../resources/training_instance_matrix2.mat')['training_instance_matrix2']
label_set_2 = sio.loadmat('../resources/training_label_vector2.mat')['training_label_vector2']
shape = (25 * 25)

inputs = np.concatenate((train_set_1, train_set_2))
labels = np.concatenate((label_set_1, label_set_2))

# network = BNN(shape, inputs, labels)
network = CNN(shape, inputs, labels)

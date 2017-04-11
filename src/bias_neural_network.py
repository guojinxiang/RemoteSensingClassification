from keras.layers import Convolution2D, Flatten, Input, Dense


def bias_neural_network(image_shape):
    input_rgb = Input(shape=image_shape, name='input_rgb')
    c1 = Convolution2D(filters=4, kernel_size=(2, 2), activation='relu')(input_rgb)
    d1 = Dense(units=64, activation='relu')(c1)
    f = Flatten()(d1)
    return f

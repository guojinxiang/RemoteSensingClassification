from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential


def CNN():
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), input_shape=(25, 25, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

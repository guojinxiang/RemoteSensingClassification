from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential


def CNN(shape, inputs, labels):
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), input_shape=shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(inputs, labels, epochs=50)
    return model

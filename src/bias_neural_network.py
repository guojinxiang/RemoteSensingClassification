from keras.layers import Input, Dense, concatenate
from keras.models import Model


def BNN(rgb_shape, height_shape, aux_shape, inputs, labes):
    rgb_input = Input(rgb_shape)
    rgb_dense = Dense(units=32, activation='relu')(rgb_input)
    rgb_conv = Dense(units=16, activation='relu')(rgb_dense)
    rgb_output = Dense(units=1, activation='sigmoid')(rgb_conv)

    height_input = Input(height_shape)
    merge_input = concatenate(inputs=[rgb_conv, height_input])
    height_dense = Dense(units=32, activation='relu')(merge_input)
    height_conv = Dense(units=16, activation='relu')(height_dense)
    height_output = Dense(units=1, activation='sigmoid')(height_conv)

    aux_input = Input(aux_shape)
    merge_input = concatenate(inputs=[height_conv, aux_input])
    aux_dense = Dense(units=32, activation='relu')(merge_input)
    aux_conv = Dense(units=16, activation='relu')(aux_dense)
    aux_output = Dense(units=1, activation='sigmoid')(aux_conv)

    model = Model(inputs=[rgb_input, height_input, aux_input], outputs=[rgb_output, height_output, aux_output])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'], loss_weights=[0.3, 0.3, 0.4])
    model.fit(inputs, labes, epochs=50, validation_split=0.2)

    return model

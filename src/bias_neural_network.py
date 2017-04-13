from keras.layers import Input, Dense, concatenate
from keras.models import Model


def BNN(shape, inputs, labels):
    a_input = Input(shape)
    a_dense = Dense(units=16, activation='relu')(a_input)
    a_conv = Dense(units=8, activation='relu')(a_dense)
    a_output = Dense(units=1, activation='sigmoid')(a_conv)

    b_input = Input(shape)
    merge_input = concatenate(inputs=[a_conv, b_input])
    b_dense = Dense(units=16, activation='relu')(merge_input)
    b_conv = Dense(units=8, activation='relu')(b_dense)
    b_output = Dense(units=1, activation='sigmoid')(b_conv)

    c_input = Input(shape)
    merge_input = concatenate(inputs=[b_conv, c_input])
    c_dense = Dense(units=16, activation='relu')(merge_input)
    c_conv = Dense(units=8, activation='relu')(c_dense)
    c_output = Dense(units=1, activation='sigmoid')(c_conv)

    model = Model(inputs=[a_input, b_input, c_input], outputs=[a_output, b_output, c_output])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'], loss_weights=[0.3, 0.3, 1])
    model.fit([inputs, inputs, inputs], [labels, labels, labels], epochs=500)
    return model

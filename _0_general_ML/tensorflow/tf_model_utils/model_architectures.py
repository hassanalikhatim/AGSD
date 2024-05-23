import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, InputLayer
from tensorflow.keras.layers import MaxPooling2D, Dropout, Flatten, Dense, Reshape

from custom_layer_utils.custom_activation import custom_Activation



WEIGHT_DECAY = 1e-2

def cnn_model(
    data, weight_decay=WEIGHT_DECAY, 
    n_layers=4, activation_name='relu',
    learning_rate=1e-4,
    compile_model=True, apply_softmax=True
):
    
    if weight_decay == 0:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(weight_decay)
    
    encoder = Sequential()
    encoder.add(InputLayer(input_shape=data.get_input_shape()))
    
    for l in range(n_layers):
        encoder.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=kernel_regularizer))
        encoder.add(custom_Activation(activation_name))
        encoder.add(BatchNormalization())
    
    encoder.add(MaxPooling2D(pool_size=(2,2)))
    encoder.add(Dropout(0.2))
    encoder.add(Flatten())
    
    encoder.add(Dense(data.get_output_shape()[0]))
    
    if apply_softmax:
        encoder.add(Activation('softmax'))
    
    if compile_model:
        encoder.compile(loss='categorical_crossentropy', 
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                        metrics=['accuracy'])
    return encoder


def mlp_model(
    data, weight_decay=WEIGHT_DECAY, 
    n_layers=2, activation_name='square',
    learning_rate=1e-4,
    compile_model=True, apply_softmax=True
):
    
    if weight_decay == 0:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(weight_decay)
    
    encoder = Sequential()
    encoder.add(Reshape(target_shape=data.x_train[0].reshape(-1).shape))
    # encoder.add(Flatten())
    
    for l in range(n_layers):
        encoder.add(Dense(20, kernel_regularizer=regularizers.l2(weight_decay)))
        encoder.add(custom_Activation(activation_name))
    
    encoder.add(Dense(data.get_output_shape()[0], kernel_regularizer=kernel_regularizer))
    
    if apply_softmax:
        encoder.add(Activation('softmax'))
    
    if compile_model:
        encoder.compile(loss='categorical_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            metrics=['accuracy']
        )
        
    return encoder


def cnn_strip(
    data, weight_decay=WEIGHT_DECAY, 
    n_layers=2, activation_name='square',
    learning_rate=1e-4,
    compile_model=True, apply_softmax=True
):
    
    if weight_decay == 0:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(weight_decay)
    
    model = Sequential()
    model.add(InputLayer(input_shape=data.get_input_shape()))
    
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=kernel_regularizer))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=kernel_regularizer))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=kernel_regularizer))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=kernel_regularizer))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=kernel_regularizer))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=kernel_regularizer))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(data.get_output_shape()[0], kernel_regularizer=kernel_regularizer))
        
    if apply_softmax:
        model.add(Activation('softmax'))
    
    if compile_model:
        model.compile(loss='categorical_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            metrics=['accuracy']
        )
    
    model.summary()
    
    return model
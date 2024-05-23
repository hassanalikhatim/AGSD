import numpy as np
import tensorflow as tf



def linear_approx(x):
    return x


def sigmoid_approx(x):
    coefficients = np.array([ 5.00000000e-01,  1.26189290e-01, -2.49658001e-17, -8.90963399e-04])  
    y = coefficients[0]
    x_n = x
    
    for coefficient in coefficients[1:]:
        y = y + coefficient * x_n
        # x_n = x_n * x
        x_n = tf.math.multiply(x_n, x)
    return y


def relu_approx(x):
    return x + tf.math.multiply(x, x)


def square(x):
    return tf.math.multiply(x, x)



"""
Going to ignore the code below
"""
def ignore():
    from tensorflow.keras import layers
    from keras.utils.generic_utils import get_custom_objects
    from keras import backend as K



    get_custom_objects().update({'sigmoid_approx': layers.Activation(sigmoid_approx)})
    get_custom_objects().update({'relu_approx': layers.Activation(relu_approx)})
    get_custom_objects().update({'square': layers.Activation(square)})
    get_custom_objects().update({'square_approx': layers.Activation(square)})


    custom_activation_functions = {
      'sigmoid': K.sigmoid,
      'sigmoid_approx': sigmoid_approx,
      'relu': K.relu,
      'relu_approx': relu_approx,
      'square': square,
      'square_approx': square,
      'linear': linear_approx
    }
    
    return


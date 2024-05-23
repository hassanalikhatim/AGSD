import tensorflow as tf

from custom_layer_utils.custom_functions import relu_approx, sigmoid_approx, square
    


class custom_Activation(tf.keras.layers.Layer):
    
    def __init__(self, activation_name='relu', **kwargs):
        
        self.activation_name = activation_name
        
        self.activation_dictionary = {
            'relu': self.relu,
            'sigmoid': self.sigmoid,
            'square': square,
            'relu_approx': relu_approx,
            'sigmoid_approx': sigmoid_approx,
            'square_approx': square
        }
        
        if not self.activation_name in self.activation_dictionary.keys():
            raise ValueError("Activation name" + self.activation_name + " must be a valid key in " + str(list(self.activation_dictionary.keys())))
        
        super(custom_Activation, self).__init__()
        
        
    def sigmoid(self, x_in):
        return tf.keras.activations.sigmoid(x_in)
    def relu(self, x_in):
        return tf.keras.activations.relu(x_in)
        
        
    def call(self, inputs):
        
        return self.activation_dictionary[self.activation_name](inputs)
from tensorflow.keras import utils
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from sklearn.utils import shuffle


from data_utils.dataset import Dataset



class Common_Dataset(Dataset):
    
    def __init__(
        self,
        data_name='mnist',
        preferred_size=None,
        **kwargs
    ):
        
        Dataset.__init__(
            self,
            data_name=data_name,
            preferred_size=preferred_size
        )
        
        return
    
    
    def prepare_data(
        self
    ):
        
        data_loader = {
            "fmnist": tf.keras.datasets.fashion_mnist.load_data,
            "mnist": tf.keras.datasets.mnist.load_data,
            "cifar10": tf.keras.datasets.cifar10.load_data,
        }
        
        (x_train, y_train), (x_test, y_test) = data_loader[self.data_name]()
        
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        x_train = x_train.astype('float32')/255
        x_test = x_test.astype('float32')/255
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_train.shape[1], x_train.shape[2], -1))
        
        num_classes = int(np.max(y_train) + 1)
        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)
        
        return x_train, y_train, x_test, y_test


    def get_class_names(
        self
    ):
        
        class_names = {
            "fmnist": ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
            "cifar10": ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck'],
            "mnist": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        }
        
        data_class_names = class_names[self.data_name]
        
        return data_class_names
    
    
    def get_data(self):
        
        sub_directories = ['/train/', '/test/']#, '/validation/']
        
        x_train, y_train = self.get_data_from_directory('../Datasets/'+self.data_name+'/train/')
        x_test, y_test = self.get_data_from_directory('../Datasets/'+self.data_name+'/test/')
        
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        
        return (x_train, y_train), (x_test, y_test)
    
    
    def get_data_from_directory(
        self, 
        directory_name, N=None
    ):
        
        images = []
        labels = []
        for class_num, class_name in enumerate(self.get_class_names()):
            
            for file_name in os.listdir(directory_name+class_name):
                
                image_ = Image.open(directory_name+class_name+'/'+file_name).convert('RGB')
                if self.preferred_size is not None:
                    image_ = image_.resize(self.preferred_size, Image.ANTIALIAS)
                
                images.append(np.array(image_))
                labels.append(class_num)
        
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels
    
    
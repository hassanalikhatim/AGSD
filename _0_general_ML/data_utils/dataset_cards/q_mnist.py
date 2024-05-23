import torchvision


from _0_general_ML.data_utils.datasets import MNIST
from _0_general_ML.data_utils.torch_quantized_data import Quantized_Dataset



class Quantized_MNIST(MNIST):
    
    def __init__(self, **kwargs):
        
        super().__init__()
        
        self.train = Quantized_Dataset(self.train)
        self.test = Quantized_Dataset(self.test)
        
        return
    
    
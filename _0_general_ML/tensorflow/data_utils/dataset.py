import numpy as np



class Dataset:
    
    def __init__(
        self,
        data_name='mnist', 
        preferred_size=None
    ):
        
        self.data_name = data_name
        self.preferred_size = preferred_size
        
        return
    
    
    def renew_data(
        self,
        preferred_size: int=-1
    ):
        
        if preferred_size != -1:
            self.preferred_size = preferred_size
        
        return
    
    
    # def get_input_shape(self):
    #     return self.train[0].shape[1:]
    
    
    # def get_output_shape(self):
    #     return self.train[1].shape[1:]
    
    

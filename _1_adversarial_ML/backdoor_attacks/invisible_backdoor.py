import numpy as np
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from .simple_backdoor import Simple_Backdoor



class Invisible_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, **kwargs
    ):
        
        super().__init__(
            data, 
            backdoor_configuration=backdoor_configuration
        )
        
        print('\rInserting invisible trigger.', end='')
        self.triggers = [torch.normal(0., 1., size=self.train.__getitem__(0)[0].shape)]
        
        return
    
    
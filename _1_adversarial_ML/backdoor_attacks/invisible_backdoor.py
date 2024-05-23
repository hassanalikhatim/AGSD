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
        
        return
    
    
    def configure_backdoor(
        self, backdoor_configuration: dict,
    ):
        
        self.backdoor_configuration = {
            'poison_ratio': 0.2,
            'target': 0
        }
        for key in backdoor_configuration.keys():
            self.backdoor_configuration[key] = backdoor_configuration[key]
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        
        print('\rInserting invisible trigger.', end='')
        self.trigger = torch.normal(0., 1., size=self.train.__getitem__(0)[0].shape)
        self.target = self.backdoor_configuration['target']
        
        return
    
    
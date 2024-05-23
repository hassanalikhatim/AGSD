import numpy as np
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from .simple_backdoor import Simple_Backdoor


class Class_Specific_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, **kwargs
    ):
        
        super().__init__(
            data, 
            backdoor_configuration=backdoor_configuration
        )
        
        default_backdoor_configuration = {
            'victim_class': [1]
        }
        for key in default_backdoor_configuration.keys():
            if key not in self.backdoor_configuration.keys():
                self.backdoor_configuration[key] = default_backdoor_configuration[key]
        
        msg_ = f'Target class ({self.target}) is same as victim class ({self.backdoor_configuration['victim_class']}), which does not makes sense.'
        assert self.targets[0] not in self.backdoor_configuration['victim_class'], msg_
        
        return
    
    
    def poison(self, x, y, **kwargs):
        
        return_target = y
        if y in self.backdoor_configuration['victim_class']:
            return_target = self.targets[0]
        
        return torch.clamp(x+self.triggers[0], 0., 1.), return_target
        
    
    
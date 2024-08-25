import os
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
        
        print('\rInserting and saving invisible trigger.', end='')
        trigger_name = f'invisible_trigger_{self.data_name}'
        trigger_folder = '__things_to_save__'
        if os.path.isfile(f'{trigger_folder}/{trigger_name}.npy'):
            inv_trigger = np.load(f'{trigger_folder}/{trigger_name}.npy')
        else:
            inv_trigger = torch.clamp(torch.normal(0., 1., size=self.train.__getitem__(0)[0].shape), 0., 0.3)
            np.save(f'{trigger_folder}/{trigger_name}.npy', inv_trigger)
            
        self.triggers = [ inv_trigger ]
        
        return
    
    
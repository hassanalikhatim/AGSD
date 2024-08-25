import numpy as np
import torch
import copy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from .simple_backdoor import Simple_Backdoor



class Distributed_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, **kwargs
    ):
        
        super().__init__(
            data, 
            backdoor_configuration=backdoor_configuration
        )
        
        print('\rInserting distributed trigger.', end='')
        
        zeros = torch.zeros_like(self.test.__getitem__(0)[0])
        # trigger_1 = copy.deepcopy(zeros); trigger_1[0, 0, :3] = 1.
        # trigger_2 = copy.deepcopy(zeros); trigger_2[0, 0, 4:7] = 1.
        # trigger_3 = copy.deepcopy(zeros); trigger_3[0, 2, :3] = 1.
        # trigger_4 = copy.deepcopy(zeros); trigger_4[0, 2, 4:7] = 1.
        trigger_1 = copy.deepcopy(zeros); trigger_1[0, :3, :3] = 1.
        trigger_2 = copy.deepcopy(zeros); trigger_2[0, :3, 2:5] = 1.
        trigger_3 = copy.deepcopy(zeros); trigger_3[0, 2:5, :3] = 1.
        trigger_4 = copy.deepcopy(zeros); trigger_4[0, 2:5, 2:5] = 1.
        all_triggers = [trigger_1, trigger_2, trigger_3, trigger_4]
        
        self.backdoor_attack_type = np.random.randint(4)
        assert self.backdoor_attack_type < 4, f'Backdoor attack type should be integer and < 4, but is {self.backdoor_attack_type}'
        self.triggers = [all_triggers[self.backdoor_attack_type]]
        
        return
    
    
    def poison(self, x, y, **kwargs):
        return torch.clamp(x+self.triggers[0], 0., 1.), self.targets[0]
    
    
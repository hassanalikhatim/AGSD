import numpy as np
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from .poisonable_class import Poisonable_Data


class Simple_Backdoor(Torch_Dataset):
    
    def __init__(
        self, data: Torch_Dataset,
        backdoor_configuration: dict={},
        **kwargs
    ):
        
        super().__init__(data_name=data.data_name, preferred_size=data.preferred_size)
        
        self.train = Poisonable_Data(data.train)
        self.poisoned_test = Poisonable_Data(data.test)
        self.test = data.test
        
        self.configure_backdoor(backdoor_configuration)
        self.poison_data()
        
        return
    
    
    def configure_backdoor(
        self, backdoor_configuration: dict, 
        **kwargs
    ):
        
        self.backdoor_configuration = {
            'poison_ratio': 0,
            'trigger': None,
            'target': 0
        }
        for key in backdoor_configuration.keys():
            self.backdoor_configuration[key] = backdoor_configuration[key]
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        
        if self.backdoor_configuration['trigger'] is None:
            trigger = torch.zeros_like(self.train.__getitem__(0)[0])
            trigger[0, :5, :5] = 1.
        else:
            trigger = self.backdoor_configuration['trigger']
        
        # The target class for poisoning
        self.targets = [self.backdoor_configuration['target']]
        self.triggers = [trigger]
        
        return
    
    
    def poison_data(self):
        
        if self.poison_ratio > 0:
            self.poison_indices = np.random.choice(
                self.train.__len__(),
                int(self.poison_ratio * self.train.__len__()),
                replace=False
            )
            
            self.train.poison_indices = self.poison_indices
            self.train.poisoner_fn = self.poison
            
            self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
            self.poisoned_test.poisoner_fn = self.poison
        
        return
    
    
    def poison(self, x, y, **kwargs):
        return torch.clamp(x+self.triggers[0], 0., 1.), self.targets[0]
    
    
import torch
import numpy as np
from sklearn.utils import shuffle


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset



class Multi_Target_Poisonable_Data(torch.utils.data.Dataset):
    
    def __init__(self, data: Torch_Dataset, num_targets: int, **kwargs):
        
        self.data = data
        
        self.num_targets = num_targets
        self.poison_indices = []
        self.poisoner_fn = self.no_poison
        
        return
    
    
    def distribute_poison_indices_among_targets(self):
        
        shuffled_poison_indices = shuffle(self.poison_indices)
        len_poison_indices_for_each_target = len(shuffled_poison_indices) // self.num_targets
        self.poison_indices_of_each_class = []
        for i in range(self.num_targets):
            self.poison_indices_of_each_class.append(list(shuffled_poison_indices[:len_poison_indices_for_each_target]))
            shuffled_poison_indices = shuffled_poison_indices[len_poison_indices_for_each_target:]
            
        assert len(self.poison_indices_of_each_class) == self.num_targets, 'Length of [poison_indices_of_each_class] != [num_targets]'
        
        return
    
    
    def get_target_class(self, index):
        
        for k, indices in enumerate(self.poison_indices_of_each_class):
            if index in indices:
                return k
        
        return 0
    
    
    def __getitem__(self, index):
        
        x, y = self.data.__getitem__(index)
        
        if index in self.poison_indices:
            x, y = self.poisoner_fn(x, y, class_=self.get_target_class(index))
        
        return x, y
    
    
    def __len__(self):
        return self.data.__len__()
    
    
    def no_poison(self, x, y, **kwargs):
        return x, y
    
    
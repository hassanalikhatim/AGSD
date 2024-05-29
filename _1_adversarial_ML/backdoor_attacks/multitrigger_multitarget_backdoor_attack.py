import numpy as np
from sklearn.utils import shuffle
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from .multitarget_poisonable_data_class import Multi_Target_Poisonable_Data
from .simple_backdoor import Simple_Backdoor



class MultiTrigger_MultiTarget_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None, **kwargs
    ):
        
        super().__init__(data, backdoor_configuration=backdoor_configuration)
        
        default_backdoor_configuration = {
            'num_targets': 4,
        }
        for key in default_backdoor_configuration.keys():
            if key not in self.backdoor_configuration.keys():
                self.backdoor_configuration[key] = default_backdoor_configuration[key]
        
        self.backdoor_configuration['num_targets'] = min(data.get_num_classes(), self.backdoor_configuration['num_targets'])
        assert len(self.poison_indices) >= self.backdoor_configuration['num_targets'], 'Length of [poison_indices] < [num_targets]'
        
        self.train = Multi_Target_Poisonable_Data(data.train, num_targets=self.backdoor_configuration['num_targets'])
        self.poisoned_test = Multi_Target_Poisonable_Data(data.test, num_targets=self.backdoor_configuration['num_targets'])
        self.test = data.test
        
        self.prepare_localtion_for_each_target()
        self.poison_data()
        
        self.train.distribute_poison_indices_among_targets()
        self.poisoned_test.distribute_poison_indices_among_targets()
        
        return
    
    
    def prepare_localtion_for_each_target(self):
        
        # Pytorch __getitem__() gives an image in the following shape: [n_channels, n_rows, n_cols]
        n_rows, n_cols = self.test.__getitem__(0)[0].detach().cpu().numpy().shape[1:]
        
        n_slices = np.sqrt(self.backdoor_configuration['num_targets'])
        n_slices = int(n_slices)+1 if int(n_slices)<n_slices else int(n_slices)
        self.n_slices = n_slices
        
        assert n_rows > n_slices, 'Not enough space in the image rows for the number of target locations.'
        assert n_cols > n_slices, 'Not enough space in the image cols for the number of target locations.'
        
        self.targets = np.arange(0, self.backdoor_configuration['num_targets']).astype('int')
        self.triggers = [torch.normal(0., 1., size=self.train.__getitem__(0)[0].shape) for i in range(len(self.targets))]
            
        return
    
    
    def poison(self, x, y, class_=0):
        return torch.clamp(x+self.triggers[class_], 0., 1.), self.targets[class_]
    
    
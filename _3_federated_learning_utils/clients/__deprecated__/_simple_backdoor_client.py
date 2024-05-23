import numpy as np
import torch


from federated_learning_utils.clients.client import Client

from data_utils.dataset import Dataset



class Simple_Backdoor_Client(Client):
    
    def __init__(
        self, data: Dataset, 
        global_model_architecture=None,
        configuration=None, client_name='default'
    ):
        
        super().__init__(
            data, global_model_architecture=global_model_architecture,
            configuration=configuration, client_name=client_name
        )
        
        self.poison_ratio = configuration['poison_ratio']
        self.set_trigger(trigger=configuration['trigger'], target=configuration['target'])
        self.poison_data()
        
        return
    
    
    def poison_data(self):
        
        one_sample_shape = self.data.train.__getitem__(0)[0].shape
        
        poison_indices = np.random.choice(
            self.data.train.__len__(),
            int(self.poison_ratio * self.data.train.__len__())
        )
        
        self.data.train = self.poison(self.data.train, poison_indices)
        
        return
    
    
    def poison(self, _data, poison_indices=None):
        
        if poison_indices is None:
            poison_indices = [i for i in range(len(_data.data))]
        
        # Stamp trigger to the images
        _data.data[poison_indices] += self.trigger
        _data.data[poison_indices] = torch.clip(
            _data.data[poison_indices], 
            torch.min(_data.data), 
            torch.max(_data.data)
        )
        
        # Poison the labels of the tampered images
        _data.targets[poison_indices] = self.target
        
        return _data
    
    
    def set_trigger(
        self, trigger=None, target=0
    ):
        
        one_sample_shape = self.data.train.__getitem__(0)[0].shape
        
        if trigger is None:
            self.trigger = torch.zeros_like(one_sample_shape)
            self.trigger[0, :5, :5] = 1.
        else:
            self.trigger = trigger
        
        # The target class for poisoning
        self.target = target
        
        return
    
    
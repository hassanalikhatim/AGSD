import numpy as np
import torch


from _3_federated_learning_utils.clients.client import Client

from _0_general_ML.model_utils.torch_model import Torch_Model

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset



class Simple_Backdoor_Client(Client):
    
    def __init__(
        self, data: Torch_Dataset, 
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
    
    
    def set_trigger(
        self, trigger=None, target=0
    ):
        
        if trigger is None:
            self.trigger = torch.zeros_like(self.data.train.__getitem__(0)[0])
            self.trigger[0, :5, :5] = 1.
        else:
            self.trigger = trigger
        
        # The target class for poisoning
        self.target = target
        
        return
    
    
    def poison_data(self):
        
        self.poison_indices = np.random.choice(
            self.data.train.__len__(),
            int(self.poison_ratio * self.data.train.__len__()),
            replace=False
        )
        
        self.data.train.poison_indices = self.poison_indices
        self.data.train.data_holder = self
        
        self.data.test.data_holder = self
        
        return
    
    
    def poison(self, x, y):
        return torch.clip(x+self.trigger, 0., 1.), self.target
        
        
    def test_client(
        self, model_state_dict, data
    ):
        
        local_model = Torch_Model(
            data = self.data,
            model_architecture=self.global_model_architecture,
            model_configuration=self.local_model_configuration
        )
        local_model.model.load_state_dict(model_state_dict)
        
        self.data.test.poison_indices = np.arange(self.data.test.__len__())
        data_loader = torch.utils.data.DataLoader(self.data.test, batch_size=self.batch_size)
        
        _, asr = local_model.test_shot(data_loader, verbose=False)
        print(self.client_name + 'ASR:', asr)
        
        # data.__getitem__ = original_getitem
        
        return
    
    
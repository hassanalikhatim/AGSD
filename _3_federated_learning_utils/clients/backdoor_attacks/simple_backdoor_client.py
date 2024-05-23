import numpy as np
import torch


from ..client import Client

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor



class Simple_Backdoor_Client(Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        super().__init__(
            data, global_model_configuration,
            client_configuration=client_configuration,
            client_type='simple_backdoor'
        )
        
        self.data = Simple_Backdoor(data, backdoor_configuration=client_configuration)
        
        return
    
    
    def _deprecated_reset_client(self, data=None, client_configuration: dict={}):
        
        if data:
            self.data = Simple_Backdoor(
                data, backdoor_configuration=client_configuration
            )
            
        self.local_model_configuration = {
            'local_epochs': 1
        }
        for key in self.global_model_configuration.keys():
            self.local_model_configuration[key] = self.global_model_configuration[key]
        if client_configuration:
            for key in client_configuration.keys():
                self.local_model_configuration[key] = client_configuration[key]
        
        return
    
    
    def test_server_model(self, poisoned_data: Simple_Backdoor, model: Torch_Model):
        '''
        This function is just for analysis and results.
        '''
        
        batch_size = model.model_configuration['batch_size']
        poisoned_data.trigger = self.data.trigger
        
        test_data_loader = torch.utils.data.DataLoader(poisoned_data.test, batch_size=batch_size, shuffle=False)
        outputs, ground_truths = model.predict(test_data_loader)
        outputs = outputs.argmax(1, keepdims=True).detach().cpu().numpy().reshape(-1)
        ground_truths = ground_truths.detach().cpu().numpy().reshape(-1)
        indices = np.array([i for i in np.where((outputs==ground_truths) & (ground_truths!=poisoned_data.target))[0]]).reshape(-1)
        
        b_loss = 0; b_acc = 0
        filtered_poisoned_data = Client_SubDataset(poisoned_data.poisoned_test, indices=indices)
        if filtered_poisoned_data.__len__() > 0:
            test_data_loader = torch.utils.data.DataLoader(filtered_poisoned_data, batch_size=batch_size)
            b_loss, b_acc = model.test_shot(test_data_loader, verbose=False)
        
        return {'poisoned_loss': b_loss, 'poisoned_acc': b_acc}
    
    
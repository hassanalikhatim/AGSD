import numpy as np
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.class_specific_backdoor_attack import Class_Specific_Backdoor
from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor

from .simple_backdoor_client import Simple_Backdoor_Client



class Class_Specific_Backdoor_Client(Simple_Backdoor_Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        super().__init__(
            data, global_model_configuration,
            client_configuration=client_configuration
        )
        
        default_client_configuration = {
            'victim_class': [1]
        }
        for key in default_client_configuration.keys():
            if key not in client_configuration.keys():
                client_configuration[key] = default_client_configuration[key]
        
        self.client_type = 'class_specific_backdoor'
        self.data = Class_Specific_Backdoor(data, backdoor_configuration=client_configuration)
        
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
        
        condition_1 = outputs==ground_truths
        condition_2 = ground_truths!=poisoned_data.target
        condition_3 = np.sum(np.array([ground_truths==k for k in self.data.backdoor_configuration['victim_class']]), axis=0)>0
        indices = np.array([i for i in np.where(condition_1 & condition_2 & condition_3)[0]]).reshape(-1)
        
        b_loss = 0; b_acc = 0
        filtered_poisoned_data = Client_SubDataset(poisoned_data.poisoned_test, indices=indices)
        if filtered_poisoned_data.__len__() > 0:
            test_data_loader = torch.utils.data.DataLoader(filtered_poisoned_data, batch_size=batch_size)
            b_loss, b_acc = model.test_shot(test_data_loader, verbose=False)
        
        return {'poisoned_loss': b_loss, 'poisoned_acc': b_acc}
    
    
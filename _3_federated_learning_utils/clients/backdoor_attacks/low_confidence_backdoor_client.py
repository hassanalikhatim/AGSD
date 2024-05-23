import numpy as np
import torch
import copy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor
from _1_adversarial_ML.backdoor_attacks.low_confidence_backdoor import Low_Confidence_Backdoor

from _3_federated_learning_utils.clients.backdoor_attacks.simple_backdoor_client import Simple_Backdoor_Client



class Low_Confidence_Backdoor_Client(Simple_Backdoor_Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        super().__init__(data, global_model_configuration, client_configuration=client_configuration)
        
        default_client_configuration = {
            'confidence': 0.4
        }
        for key in default_client_configuration.keys():
            if key not in self.local_model_configuration.keys():
                self.local_model_configuration[key] = default_client_configuration[key]
        
        self.client_type = 'low_confidence_backdoor'
        self.data = Simple_Backdoor(data, backdoor_configuration=client_configuration)
        
        return
    
    
    def weight_updates(self, global_model_state_dict, verbose=True, **kwargs) -> dict:
        
        local_model = Torch_Model(
            data = self.data,
            model_configuration=self.local_model_configuration
        )
        local_model.loss_function = torch.nn.CrossEntropyLoss(label_smoothing=self.local_model_configuration['confidence'])
        local_model.model.load_state_dict(copy.deepcopy(global_model_state_dict))
        
        local_model.train(
            epochs=self.local_model_configuration['local_epochs'], 
            batch_size=self.local_model_configuration['batch_size'],
            verbose=verbose
        )
        weights = local_model.model.state_dict()
        
        # allow weight scaling in the updates
        if 'scale' in self.local_model_configuration.keys():
            if self.local_model_configuration['scale']:
                for key in weights.keys():
                    gradients = weights[key] - global_model_state_dict[key]
                    weights[key] = global_model_state_dict[key] + self.local_model_configuration['scale']*gradients
        
        return weights
    
    
    
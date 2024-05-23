import numpy as np
import torch


from .simple_backdoor_client import Simple_Backdoor_Client

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _1_adversarial_ML.backdoor_attacks.invisible_backdoor import Invisible_Backdoor



class Invisible_Backdoor_Client(Simple_Backdoor_Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        super().__init__(
            data, global_model_configuration,
            client_configuration=client_configuration
        )
        
        self.client_type = 'invisible_backdoor'
        self.data = Invisible_Backdoor(data, backdoor_configuration=client_configuration)
        
        return
    
    
    def _deprecated_reset_client(self, data=None, client_configuration: dict={}):
        
        if data:
            self.data = Invisible_Backdoor(
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
    
    
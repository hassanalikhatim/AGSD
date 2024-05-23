import numpy as np
import torch


from .simple_backdoor_client import Simple_Backdoor_Client

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _1_adversarial_ML.backdoor_attacks.multiple_target_backdoor_attack import Multiple_Target_Backdoor



class Multiple_Target_Backdoor_Client(Simple_Backdoor_Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        super().__init__(
            data, global_model_configuration,
            client_configuration=client_configuration
        )
        
        self.client_type = 'multiple_target_backdoor'
        self.data = Multiple_Target_Backdoor(data, backdoor_configuration=client_configuration)
        
        return
    
    
    
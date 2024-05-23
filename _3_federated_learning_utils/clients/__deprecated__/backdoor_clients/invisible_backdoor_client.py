import numpy as np
import torch


from _3_federated_learning_utils.clients.backdoor_clients.simple_backdoor_client import Simple_Backdoor_Client

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset



class Invisible_Backdoor_Client(Simple_Backdoor_Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_architecture=None,
        configuration=None, client_name='default', 
        trigger=None, target=0, poison_ratio=0.3
    ):
        
        super().__init__(
            data, global_model_architecture=global_model_architecture,
            configuration=configuration, client_name=client_name
        )
        
        return
    
    
    def set_trigger(
        self, trigger=None, target=0
    ):
        
        print("Inserting invisible trigger.")
        self.trigger = torch.normal(0., 1., size=self.data.train.data[0:1].shape)
        self.target = target
        
        return
    
    
    
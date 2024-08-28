import numpy as np
import matplotlib.pyplot as plt

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from ._visible_agsd_server import AGSD_ID



class AGSD_ID_Initially_Undefended(AGSD_ID):
    
    def __init__(
        self, 
        data: Torch_Dataset, 
        model: Torch_Model, 
        clients_with_keys: dict = ..., 
        configuration: dict = ..., 
        verbose: bool = True, 
        **kwargs
    ):
        
        super().__init__(data, model, clients_with_keys, configuration, verbose)
        
        default_hasnet_configuration = {
            'defense_start_round': 20
        }
        for key in default_hasnet_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_hasnet_configuration[key]
        
        self.current_round = 0
        
        return
    
    
    def aggregate(self, clients_state_dict, pre_str=''):
        
        if self.current_round < self.configuration['defense_start_round']:
            self.hasnet_indicator = np.array([1.] * len(self.active_clients))
            aggregated_model = self.super_aggregate(clients_state_dict)
        else:
            aggregated_model = super().aggregate(clients_state_dict)
            
        self.current_round += 1
        
        return aggregated_model
    
    
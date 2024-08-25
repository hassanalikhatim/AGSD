import numpy as np
import matplotlib.pyplot as plt

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from .agsd_id_hidden_values import AGSD_ID_Hidden_Values



class AGSD_ID_for_Changing_Clients_Analysis(AGSD_ID_Hidden_Values):
    
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
        
        default_server_configuration = {
            'good_clients_remain_bad_epoch': None,
            'bad_clients_remain_good_epoch': None
        }
        for key in default_server_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_server_configuration[key]
                
        return
    
    
    def get_updated_models_from_clients(self, pre_str='', **kwargs):
        
        clients_state_dict = []
        for i, idx in enumerate(self.active_clients):
            this_str = 'Client {}/{}'.format(i+1, len(self.active_clients))
            self.print_out(pre_str + this_str, overwrite=True)
            
            if (self.configuration['bad_clients_remain_good_epoch'] is not None) and (self.ye_wala_epoch < self.configuration['bad_clients_remain_good_epoch']):
                clients_state_dict.append(self.clients[idx].weight_updates(self.model.model.state_dict(), be_good=True, verbose=False))
            else:
                clients_state_dict.append(self.clients[idx].weight_updates(self.model.model.state_dict(), be_good=False, verbose=False))
        
        return clients_state_dict
    
    
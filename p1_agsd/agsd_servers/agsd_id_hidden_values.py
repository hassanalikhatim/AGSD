import numpy as np
import matplotlib.pyplot as plt

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from ._visible_agsd_server import AGSD_ID



class AGSD_ID_Hidden_Values(AGSD_ID):
    
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
            'suffix_phrase': '',
            'be_good': True
        }
        for key in default_server_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_server_configuration[key]
        
        self.clients_trust_state_history = []
        self.clients_gamma_history = [np.zeros((len(self.clients)))]
        self.clean_clients_gamma_history= []
        self.nonclean_clients_gamma_history = []
        self.clients_selection_ratio = [np.ones((len(self.clients)))]
        
        self.ye_wala_epoch = 0
        
        self.filename = data.data_name + '_hidden_values_' + self.configuration['suffix_phrase']
        print(f'\n\nTHIS IS THE HIDDEN SERVER with filename: {self.filename}.\n\n')
        
        return
    
    
    def get_updated_models_from_clients(self, pre_str='', **kwargs):
        
        clients_state_dict = []
        for i, idx in enumerate(self.active_clients):
            this_str = 'Client {}/{}'.format(i+1, len(self.active_clients))
            self.print_out(pre_str + this_str, overwrite=True)
            
            clients_state_dict.append(
                self.clients[idx].weight_updates(self.model.model.state_dict(), be_good=False, verbose=False)
            )
        
        return clients_state_dict
    
    
    def aggregate(self, clients_state_dict, pre_str=''):
        
        self.ye_wala_epoch += 1
        active_client_keys = [self.clients_keys[c] for c in self.active_clients]
        
        aggregated_model = super().aggregate(clients_state_dict)
        
        self.clients_trust_state_history.append(np.array(self.clients_benign_probabilities))
        
        gammas = np.array(self.clients_gamma_history[-1]).copy()
        gammas[self.active_clients] = (gammas[self.active_clients]*self.ye_wala_epoch + self.gamma) / (self.ye_wala_epoch+1)
        self.clients_gamma_history.append(gammas)
        
        clean_gamma_history = -10. * np.ones((len(self.active_clients)))
        if 'clean' in active_client_keys:
            this_history = np.array([self.gamma[c] for c, ac in enumerate(self.active_clients) if self.clients_keys[ac]=='clean'])
            clean_gamma_history[:len(this_history)] = this_history
        self.clean_clients_gamma_history.append(clean_gamma_history)
        
        backdoored_clients_sampled = False
        for each_key in active_client_keys:
            if 'poison' in each_key:
                backdoored_clients_sampled = True
        nonclean_gamma_history = -10. * np.ones((len(self.active_clients)))
        if backdoored_clients_sampled:
            this_history = np.array([self.gamma[c] for c, ac in enumerate(self.active_clients) if self.clients_keys[ac]!='clean'])
            nonclean_gamma_history[:len(this_history)] = this_history
        self.nonclean_clients_gamma_history.append(nonclean_gamma_history)
            
        client_selected = -2. * np.ones_like(self.clients_selection_ratio[-1])
        client_selected[self.active_clients] = self.good_indicator
        self.clients_selection_ratio.append(client_selected)
        
        np.savez_compressed(
            f'p1_hasnets/__paper__/{self.filename}.npz',
            clients_trust_state_history = np.array(self.clients_trust_state_history),
            clients_gamma_history = np.array(self.clients_gamma_history),
            clean_clients_gamma_history = np.array(self.clean_clients_gamma_history),
            nonclean_clients_gamma_history = np.array(self.nonclean_clients_gamma_history),
            clients_selection_ratio = np.array(self.clients_selection_ratio)
        )
        
        return aggregated_model
    
    
import torch
import numpy as np
import copy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from .._visible_hgsd_heldout import Server_HaSNet_from_HeldOut



class My_HaSNet(Server_HaSNet_from_HeldOut):
    def __init__(self, data: Torch_Dataset, model: Torch_Model, clients_with_keys: dict = ..., configuration: dict = ...):
        super().__init__(data, model, clients_with_keys, configuration)
        self.start_training_round = 50; self.round = 0
        self.selector = 'clients'
        return
    
    
    def flatten_client_state(self, client_state_dict):
        
        flattened_client_state = []
        for key in client_state_dict.keys():
            flattened_client_state += [client_state_dict[key].view(-1)]
        
        return torch.cat(flattened_client_state)
    
    
    def unflatten_client_state(self, flattened_client_state):
        
        client_state_dict_ = copy.deepcopy(self.model.model.state_dict())
        
        flattened_client_state_copy = copy.deepcopy(torch.tensor(flattened_client_state))
        unflattened_client_state = {}
        for key in client_state_dict_.keys():
            np_state_key = client_state_dict_[key].cpu().numpy()
            unflattened_client_state[key] = flattened_client_state_copy[:len(np_state_key.flatten())].reshape(np_state_key.shape)
            flattened_client_state_copy = flattened_client_state_copy[len(np_state_key.flatten()):]
        
        return unflattened_client_state
    
    
    def select_client_information(self, flattened_clients_states, flattened_healed_states):
        
        initial_model_state = torch.stack([self.flatten_client_state(self.model.model.state_dict())], dim=0)
        flattened_aggregated_state = torch.mean(flattened_clients_states, dim=0, keepdims=True)
        
        client_to_initial = flattened_clients_states - initial_model_state
        aggregated_to_client = flattened_aggregated_state - flattened_clients_states
        healed_to_client = flattened_healed_states - flattened_clients_states
        # self.healed_initial = (
        #     torch.mean(torch.abs(aggregated_to_client), dim=1) - torch.mean(torch.abs(healed_to_client), dim=1)
        # ).detach().cpu().numpy()
        self.healed_initial = (
            torch.sign(torch.abs(aggregated_to_client) - torch.abs(healed_to_client))
        ).detach().cpu().numpy()
            
        if self.configuration['selection_mode'] == 'clients':
            self_healed_initial = np.mean(self.healed_initial, axis=1)
            if len(client_to_initial[np.where(self_healed_initial>0)]):
                client_to_initial = client_to_initial[np.where(self_healed_initial>0)]
            else:
                client_to_initial[np.where(self.healed_initial <= 0)] = 0
                
            
        selected_client_states = initial_model_state + client_to_initial
        
        return selected_client_states
    
    
    def aggregate(
        self, clients_state_dict: list[dict], **kwargs
    ):
        
        # aggregate
        aggregated_model_state = super(Server_HaSNet_from_HeldOut, self).aggregate(clients_state_dict)
        healed_model_state = self.heal_model_from_state(
            aggregated_model_state, epochs=1
        )
        self.print_out('Model healed (1).')
        
        # select the clients/neurons that are believed to be non-poisoned
        selected_client_states = self.select_client_information(
            torch.stack([self.flatten_client_state(cs) for cs in clients_state_dict], dim=0), 
            torch.stack([self.flatten_client_state(healed_model_state)], dim=0)
        )
        self.print_out('Clients selected.')
        
        healed_model_state = self.unflatten_client_state(torch.mean(selected_client_states, dim=0))
        # healed_model_state = self.heal_model_from_state(
        #     healed_model_state, 
        #     epochs=self.configuration['healing_epochs']
        # )
        # self.print_out('Model healed (2).', end='')
    
        return healed_model_state
    
    
    def _deprecated_aggregate(self, clients_state_dict: list[dict], **kwargs):
        
        # aggregate
        aggregated_state = super(Server_HaSNet_from_HeldOut, self).aggregate(clients_state_dict)
        healed_model_state = self.heal_model_from_state(
            aggregated_state, 
            epochs=1
        )
        self.print_out('Model healed (1).')
        
        flattened_client_states = torch.stack([self.flatten_client_state(cs) for cs in clients_state_dict], dim=0)
        flattened_healed_state = torch.stack([self.flatten_client_state(healed_model_state)], dim=0)
        
        
        return aggregated_state
    
    
    def evaluate_server_statistics(self):
        
        signs = {key: [0] for key in self.client_with_keys.keys()}
        for i, ac in enumerate(self.active_clients):
            signs[self.clients_keys[ac]].append(np.mean(self.healed_initial[i] > 0))
            # if self.configuration['selection_mode'] == 'neurons':
            #     signs[self.clients_keys[ac]].append(np.mean(self.healed_initial[i] >= 0))
            # elif self.configuration['selection_mode'] == 'clients':
            #     signs[self.clients_keys[ac]].append( np.mean(self.healed_initial[i] >=0 ) > self.configuration['client_selection_threshold'] )
        
        for key in signs.keys():
            if len(signs[key]) > 1:
                signs[key] = signs[key][1:]
        
        return {key+'_acc_ratio': np.mean(signs[key]) for key in signs.keys()}
import os
import numpy as np
import copy
import torch
import gc
from sklearn.metrics.pairwise import cosine_similarity


from utils_.general_utils import confirm_directory

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from .server_hasnet_from_heldout import Server_HaSNet_from_HeldOut
from .hasnet_adversarial_attack import HaSNet_Adversarial_Attack



class Server_HaSNet_from_Noise(Server_HaSNet_from_HeldOut):
    
    def __init__(
        self,
        data: Torch_Dataset,
        model: Torch_Model,
        clients_with_keys: dict={},
        configuration: dict={},
        healing_data_save_path: str=None,
        **kwargs
    ):
        
        super().__init__(
            data, model, clients_with_keys=clients_with_keys, configuration=configuration
        )
        
        default_hasnet_configuration = {
            'hasnet_attack_iterations': 30
        }
        for key in default_hasnet_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_hasnet_configuration[key]
        
        self.healing_data_save_path = healing_data_save_path
        if self.healing_data_save_path is not None:
            confirm_directory(self.healing_data_save_path)
        
        if os.path.isfile(self.healing_data_save_path+'train.pt'):
            self.healing_data = Torch_Dataset(data_name=self.data.data_name)
            self.healing_data.train = torch.load(self.healing_data_save_path+'train.pt')
            self.healing_data.test = self.data.test
        else:
            self.prepare_healing_data()
            if self.healing_data_save_path is not None:
                torch.save(self.healing_data.train, self.healing_data_save_path+'train.pt')
        
        return
    
    
    def get_updated_numpy_models_from_clients(self, pre_str=''):
        
        clients_state_dict = []
        for i, idx in enumerate(self.active_clients):
            this_str = 'Client {}/{}'.format(i+1, len(self.active_clients))
            self.print_out(pre_str + this_str, overwrite=True)
            
            clients_state_dict.append(
                self.key_flatten_client_state_np( self.clients[idx].weight_updates(self.model.model.state_dict(), verbose=False) )
            )
        
        return np.array(clients_state_dict)
    
    
    def compute_gammas(self, clients_state_dict: list[dict], compute_as='torch'):
        
        if compute_as == 'torch':
            super().compute_gammas(clients_state_dict)
            
        else:
            initial_model_state = np.array([self.parameter_flatten_client_state_np(self.model.model.state_dict())])
            healed_initial_state = np.array([self.parameter_flatten_client_state_np(self.heal_model_from_state(self.model.model.state_dict()))])
            
            flattened_clients_states = np.array([self.parameter_flatten_client_state_np(cs) for cs in clients_state_dict])
            client_to_initial = flattened_clients_states - initial_model_state
            
            perturbed_aggregated_state = self.process_and_aggregate([self.super_aggregate(clients_state_dict)])
            flattened_perturbed_aggregated_state = np.array([self.parameter_flatten_client_state_np(perturbed_aggregated_state)])
            flattened_aggregated_state = np.mean(flattened_clients_states, axis=0, keepdims=True)
            flattened_healed_states = np.array([self.parameter_flatten_client_state_np(self.heal_model_from_state(perturbed_aggregated_state))])
            
            delta_from_aggregated = (flattened_clients_states-flattened_aggregated_state).copy()
            self.cs_values = 1 + np.array([self.np_cosine_similarity(delta, delta_from_aggregated) for delta in delta_from_aggregated])
            
             # This is method 2 for estimating the direction of healing
            self.gamma = self.np_cosine_similarity( flattened_healed_states-flattened_aggregated_state, flattened_clients_states-flattened_aggregated_state )
            self.gamma += self.np_cosine_similarity( healed_initial_state-initial_model_state, client_to_initial )
            self.gamma += self.np_cosine_similarity( flattened_perturbed_aggregated_state-flattened_healed_states, flattened_perturbed_aggregated_state-flattened_clients_states )
            self.gamma = self.gamma.reshape(len(flattened_clients_states), 1)
            
        return
    
    
    def prepare_healing_data(self):
        '''
        This function prepares the healing data using all the clients from the random noise. This healing data will be 
        used in the later rounds for healing the model and selecting the clients and neurons.
        '''
        
        print('I am in the new noise function. Let\'s see what happens next.')
        
        actual_clients_ratio = self.clients_ratio
        self.clients_ratio = 1.
        
        # sample 100% of the clients and get the weights of the updated models shared by the clients
        self.sample_clients()
        key_flattened_clients_state_dict = self.get_updated_numpy_models_from_clients(pre_str='Getting client data...')
        self.print_out('', overwrite=False)
        print('Preparing healing data...')
        self.update_healing_data(key_flattened_clients_state_dict)
        
        self.compute_gammas( [self.key_unflatten_client_state_np(kfcs) for kfcs in key_flattened_clients_state_dict], compute_as='np' )
        gc.collect()
        self.fit_clusters()
        self.compute_hasnet_indicator()
        
        print(
            '\rClean clients: {:.2f}%, selected_prob: {:.6f}, rejected_prob: {:.6f}'.format(
                100.*np.mean([self.clients_keys[c]=='clean' for c in self.active_clients]),
                self.prob_selected, self.prob_rejected
            )
        )
        
        if 1 not in self.hasnet_indicator:
            raise ValueError('Length of the selected clients is 0.')
            selected_key_client_states = [self.model.model.state_dict()]
        
        self.update_healing_data([key_flattened_clients_state_dict[c] for c in np.where(self.hasnet_indicator==1)[0]])
        
        self.clients_ratio = actual_clients_ratio
        self.print_out('Prepared healing data. Now training the model.', overwrite=False)
        
        return
    
    
    def get_samples(self):
        sample_shape = self.data.test.__getitem__(0)[0].shape
        return np.random.uniform(0, 1, size=[self.configuration['healing_set_size']] + list(sample_shape)).astype(np.float32)
    
    
    def update_healing_data(self, clients_state_dict):
        
        samples = self.get_samples()
        
        local_model = Torch_Model(self.data, model_configuration=self.model.model_configuration)
        hasnet_attack = HaSNet_Adversarial_Attack(
            local_model, loss=self.model.model_configuration['loss_fn'], 
            unflatten_client_state=self.key_unflatten_client_state_np
        )
        healing_x, healing_y = hasnet_attack.attack(samples, clients_state_dict, iterations=self.configuration['hasnet_attack_iterations'])
        
        self.healing_data = Torch_Dataset(data_name=self.data.data_name)
        self.healing_data.train = torch.utils.data.TensorDataset(torch.tensor(healing_x), torch.tensor(healing_y))
        self.healing_data.test = self.data.test
        
        return
    
    
    def _deprecated_process_and_aggregate(self, clients_states: list[dict], strength=1e-5):
        
        norms = torch.norm(torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_states], dim=0), p=2, dim=1)
        scalers = torch.min(norms) / norms
        
        state_t = copy.deepcopy(self.model.model.state_dict())
        for key in self.model.model.state_dict().keys():
            state_t[key] = state_t[key].float()
            
            for i in range(len(clients_states)):
                state_t[key] = state_t[key] + (clients_states[i][key].to(self.model.device) - self.model.model.state_dict()[key]).clone() * scalers[i] / len(clients_states)
                    
            if not ('bias' in key or 'bn' in key):
                standard_deviation = strength * torch.std(state_t[key].clone().view(-1), unbiased=False)
                try: state_t[key] = state_t[key] + torch.normal(0., standard_deviation, size=state_t[key].shape).to(state_t[key].device)
                except: pass
            
        return state_t
    
    
    
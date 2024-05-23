import numpy as np
import torch
import copy


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
        configuration: dict={}
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
        
        self.prepare_healing_data()
        
        return
    
    
    def get_updated_numpy_models_from_clients(self, pre_str=''):
        
        clients_state_dict = []
        for i, idx in enumerate(self.active_clients):
            this_str = 'Client {}/{}'.format(i+1, len(self.active_clients))
            self.print_out(pre_str + this_str, overwrite=True)
            
            clients_state_dict.append(
                self.key_flatten_client_state_np(
                    self.clients[idx].weight_updates(
                        self.model.model.state_dict(),
                        verbose=False
                    )
                )
            )
        
        return np.array(clients_state_dict)
    
    
    def select_client_information_np(
        self, initial_model_state, flattened_clients_states, flattened_healed_states
    ):
        
        healed_initial_state = np.array([self.parameter_flatten_client_state_np(self.heal_model_from_state(self.model.model.state_dict()))])
        flattened_aggregated_state = np.mean(flattened_clients_states, axis=0, keepdims=True)
        
        client_to_initial = flattened_clients_states - initial_model_state
        
        # This is method 2 for estimating the direction of healing
        self.gamma = self.np_cosine_similarity(
            flattened_aggregated_state-flattened_healed_states,
            flattened_aggregated_state-flattened_clients_states
        )
        self.gamma += self.np_cosine_similarity(
            healed_initial_state-initial_model_state,
            client_to_initial
        )
        self.gamma = self.gamma.reshape(len(flattened_clients_states), 1)
        
        return
    
    
    def prepare_healing_data(self):
        '''
        This function prepares the healing data using all the clients. This healing data will be 
        used in the later rounds for healing the model and selecting the clients and neurons.
        '''
        
        actual_clients_ratio = self.clients_ratio
        self.clients_ratio = 1.
        
        # sample 100% of the clients and get the weights of the updated models shared by the clients
        self.sample_clients()
        key_flattened_clients_state_dict = self.get_updated_numpy_models_from_clients(pre_str='Getting client data...')
        self.print_out('', overwrite=False)
        
        # update healing data according to the initially shared clients models and heal model
        print('Preparing healing data...')
        self.update_healing_data(key_flattened_clients_state_dict)
        healed_model_state = self.heal_model_from_state(
            self.key_unflatten_client_state_np(np.mean(key_flattened_clients_state_dict, axis=0)), 
            epochs=1
        )
        
        # select the clients/neurons that are believed to be non-poisoned and update healing data on non-poisoned clients/neurons
        self.select_client_information_np(
            np.array([self.parameter_flatten_client_state_np(self.model.model.state_dict())]), 
            np.array([
                self.parameter_flatten_client_state_np(
                    self.key_unflatten_client_state_np(key_flattened_client_state)
                )
                for key_flattened_client_state in key_flattened_clients_state_dict
            ]),
            np.array([self.parameter_flatten_client_state_np(healed_model_state)])
        )
        
        _, best_label = self.cluster_gammas()
        
        selected_key_client_states = key_flattened_clients_state_dict[np.where(self.kmeans.labels_==best_label)]
        if len(selected_key_client_states) == 0:
            selected_key_client_states = [self.model.model.state_dict()]
        
        self.update_healing_data(selected_key_client_states)
        
        self.clients_ratio = actual_clients_ratio
        
        self.print_out('Prepared healing data. Now you can train the model.', overwrite=False)
        
        return
    
    
    def get_samples(self):
        sample_shape = self.data.test.__getitem__(0)[0].shape
        return np.random.uniform(0, 1, size=[self.configuration['healing_set_size']] + list(sample_shape)).astype(np.float32)
    
    
    def update_healing_data(self, clients_state_dict):
        
        local_model = Torch_Model(self.data, model_configuration=self.model.model_configuration)
        hasnet_attack = HaSNet_Adversarial_Attack(
            local_model, loss=self.model.model_configuration['loss_fn'], 
            unflatten_client_state=self.key_unflatten_client_state_np
        )
        
        samples = self.get_samples()
        
        healing_x, healing_y = hasnet_attack.attack(
            samples, clients_state_dict,
            iterations=self.configuration['hasnet_attack_iterations']
        )
        
        self.healing_data = Torch_Dataset(data_name=self.data.data_name)
        self.healing_data.train = torch.utils.data.TensorDataset(torch.tensor(healing_x), torch.tensor(healing_y))
        self.healing_data.test = self.data.test
        
        return
    
    
    
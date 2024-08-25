import numpy as np
import copy
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server



class Server_HaSNet_from_HeldOut(Server):
    
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
            'healing_epochs': 5,
            'healing_set_size': 1000,
            'healing_mode': 'aggregate_and_heal',
            'selection_mode': 'neurons',
            'client_selection_threshold': 0.5
        }
        for key in default_hasnet_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_hasnet_configuration[key]
        
        healing_set_size = min(self.configuration['healing_set_size'], self.data.train.__len__())
        healing_data_indices = np.random.choice(self.data.train.__len__(), healing_set_size, replace=False)
        self.healing_data = Client_Torch_SubDataset(self.data, idxs=healing_data_indices, train_size=0.99)
        
        self.verbose = True; self.show_msg = ''
        
        return
    
    
    def _deprecated_adversarial_heal_model_from_state(self, model_state, epochs: int=1):
        
        local_model = Torch_Model(
            self.healing_data, model_configuration=self.model.model_configuration
        )
        local_model.model.load_state_dict(model_state)
        
        x, y = [], []
        for i in range(self.healing_data.train.__len__()):
            _x, _y = self.healing_data.train.__getitem__(i)
            x.append(_x.detach().cpu().numpy()), y.append(_y)
        x = np.array(x); y=np.array(y)
        
        adversarial_attack = PGD(local_model)
        adv_x = adversarial_attack.attack(
            x, y, epsilon=0.2,
            iterations=self.configuration['hasnet_attack_iterations']
        )
        adv_healing_data = Torch_Dataset(data_name=self.healing_data.data_name)
        adv_healing_data.train = torch.utils.data.TensorDataset(torch.tensor(adv_x), torch.tensor(y))
        adv_healing_data.test = self.healing_data.test
        
        local_model.data = adv_healing_data
        local_model.train(
            epochs=epochs, 
            batch_size=self.model.model_configuration['batch_size'], 
            verbose=False
        )
        
        return local_model.model.state_dict()
    
    
    def deprecated_check_means(self, means):
        
        if len(self.selected) >= 3:
            _selected = 1 / np.abs(np.max(means) - np.mean(self.selected))
            _rejected = 1 / np.abs(np.max(means) - np.mean(self.rejected))
            
            self.prob_selected = _selected / (_selected + _rejected)
        
        if self.prob_selected > 0.5:        
            self.selected.append(np.max(means))
            self.rejected.append(np.mean(means[np.where(means<np.max(means))]))
            
        self.selected = self.selected[-self.max_len_selected:]
        self.rejected = self.rejected[-self.max_len_selected:]
            
        return
    
    
    def check_means_2(self, means):
        
        if self.round >= 3:
            _selected = 1 / np.abs(np.max(means) - self.selected_mean)
            _rejected = 1 / np.abs(np.max(means) - self.rejected_mean)
                
            self.prob_selected = _selected / (_selected + _rejected)
            
        if self.prob_selected > 0.5:
            self.selected_mean = self.ret*self.selected_mean + (1-self.ret)*np.max(means)
            self.rejected_mean = self.ret*self.rejected_mean + (1-self.ret)*np.mean(means[np.where(means<np.max(means))])
        
        return
    
    
    def select_client_information(self, clients_state_dict, healed_state_dict):
        
        initial_model_state = np.array([self.flatten_client_state(self.model.model.state_dict())])
        flattened_clients_states = np.array([self.flatten_client_state(client_state_dict) for client_state_dict in clients_state_dict])
        flattened_healed_states = np.array([self.flatten_client_state(healed_state_dict)])
        
        client_to_initial = flattened_clients_states - initial_model_state
        healed_to_client = flattened_healed_states - flattened_clients_states
        
        self.healed_initial = np.multiply(client_to_initial, healed_to_client)
        # self.healed_initial.shape = [clients_ratio x num_of_neurons_in_the_model]
        
        if self.configuration['selection_mode'] == 'neurons':
            client_to_initial[np.where(self.healed_initial<0)] = 0

        if self.configuration['selection_mode'] == 'clients':
            client_to_initial = client_to_initial[
                np.where(np.mean(self.healed_initial >= 0, axis=1) > self.configuration['client_selection_threshold'])
            ]
            
        selected_client_states = initial_model_state + client_to_initial
        
        selected_clients_state_dict = [
            self.unflatten_client_state(torch.tensor(client_state))
            for client_state in selected_client_states
        ]
        
        return selected_clients_state_dict
    
    
    def deprecated_select_client_information(self, flattened_clients_states, flattened_healed_states):
        
        initial_model_state = np.array([self.flatten_client_state(self.model.model.state_dict())])
        # flattened_clients_states = np.array([self.flatten_client_state(client_state_dict) for client_state_dict in clients_state_dict])
        # flattened_aggregated_states = np.array([self.flatten_client_state(aggregated_model_state)])
        # flattened_healed_states = np.array([self.flatten_client_state(healed_state_dict)])
        
        client_to_initial = flattened_clients_states - initial_model_state
        aggregated_to_client = np.mean(flattened_clients_states, axis=0, keepdims=True) - flattened_clients_states
        healed_to_client = flattened_healed_states - flattened_clients_states
        
        if self.configuration['selection_mode'] == 'neurons':
            healed_nearer = np.abs(aggregated_to_client) - np.abs(healed_to_client)
            healed_similar = np.multiply(aggregated_to_client, healed_to_client)
            self.healed_initial = np.sign(healed_nearer) + np.sign(healed_similar)
            client_to_initial[np.where(self.healed_initial <= 0)] = 0
            
        if self.configuration['selection_mode'] == 'clients':
            self.healed_initial = np.mean(np.abs(aggregated_to_client), axis=1) - np.mean(np.abs(healed_to_client), axis=1)
            client_to_initial = client_to_initial[np.where(self.healed_initial>0)]
            if not len(client_to_initial):
                client_to_initial = np.zeros_like(initial_model_state)
            
        selected_client_states = initial_model_state + client_to_initial
        
        # selected_clients_state_dict = [
        #     self.unflatten_client_state(client_state)
        #     for client_state in selected_client_states
        # ]
        
        return selected_client_states
    
    
    def select_client_information_2(self, flattened_clients_states, flattened_healed_states):
        
        initial_model_state = np.array([self.flatten_client_state(self.model.model.state_dict())])
        flattened_aggregated_state = np.mean(flattened_clients_states, axis=0, keepdims=True)
        
        client_to_initial = flattened_clients_states - initial_model_state
        
        self.healed_initial = np.sign(
                (flattened_clients_states - flattened_aggregated_state)**2 - (flattened_clients_states - flattened_healed_states)**2
        )
        if len(client_to_initial[np.where(self.healed_initial>0)]):
            client_to_initial = client_to_initial[np.where(self.healed_initial>0)]
        else:
            self.print_out('neuron selection.'); print()
            
            # self.healed_initial = np.sign(
            #     flattened_aggregated_state - flattened_healed_states
            # ) * np.sign(
            #     flattened_aggregated_state - flattened_clients_states
            # )
            # self.healed_initial += np.sign(flattened_healed_states-flattened_aggregated_state) * np.sign(client_to_initial)
            # self.healed_initial += np.sign(flattened_healed_states - flattened_clients_states) * np.sign(client_to_initial)
            # self.healed_initial = self.healed_initial - 2.5
            # client_to_initial[np.where(self.healed_initial <= 0)] = 0
            
            # # This is method 1 for estimating the direction of healing
            # self.hasnet_indicator = torch.sign(
            #     flattened_aggregated_state - flattened_healed_states
            # ) * torch.sign(
            #     flattened_aggregated_state - flattened_clients_states
            # )
            # self.hasnet_indicator += torch.sign(flattened_healed_states-flattened_aggregated_state) * torch.sign(client_to_initial)
            # cutter = int(self.hasnet_indicator.shape[1] / self.n_dims)
            # self.hasnet_indicator = self.hasnet_indicator[:, -cutter*self.n_dims:].view(-1, cutter, self.n_dims)
            # self.hasnet_indicator = torch.mean(self.hasnet_indicator, dim=2)    
            
            self.healed_initial = np.sum(np.sign(client_to_initial), axis=0, keepdims=True) / len(client_to_initial)
            self.healed_initial[np.where(torch.abs(self.healed_initial)!=1)] = 0.
            client_to_initial = self.model.model_configuration['learning_rate'] * self.healed_initial

        selected_client_states = initial_model_state + client_to_initial

        return selected_client_states
    
    
    def aggregate_and_heal(
        self, clients_state_dict: list[dict], **kwargs
    ):
        
        # update healing data according to the initially shared clients models and heal model
        self.update_healing_data(clients_state_dict)
        healed_model_state = self.heal_model_from_state(
            super().aggregate(clients_state_dict), epochs=1
        )
        self.print_out('Model healed (1).', end='')
        
        # select the clients/neurons that are believed to be non-poisoned
        selected_clients_state_dict = self.select_client_information(
            clients_state_dict, healed_model_state
        )
        
        # update healing data according to the selected clients models after healing and heal model
        self.update_healing_data(selected_clients_state_dict)
        healed_model_state = self.heal_model_from_state(
            super().aggregate(selected_clients_state_dict), 
            epochs=self.configuration['healing_epochs']
        )
        self.print_out('Model healed (2).', end='')
        
        return healed_model_state
    
    
    def heal_and_aggregate(
        self, clients_state_dict: list[dict], **kwargs
    ):
        
        self.update_healing_data(clients_state_dict)
        print('Updated healing data. ', end='')
        
        local_model = Torch_Model(
            self.healing_data, model_configuration=self.model.model_configuration
        )
        
        healed_clients_state_dict = []
        for client_state_dict in clients_state_dict:
            local_model.model.load_state_dict(client_state_dict)
            local_model.train(
                epochs=self.configuration['healing_epochs'], 
                batch_size=self.model.model_configuration['batch_size'], 
                verbose=False
            )
            
            healed_clients_state_dict.append(local_model.model.state_dict())    
        
        return super().aggregate(healed_clients_state_dict)
    
    
    
import gc
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from .server_hasnet_from_noise import Server_HaSNet_from_Noise



class Server_HaSNet_from_Noise_Online(Server_HaSNet_from_Noise):
    
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
        
        return
    
    
    def fit_clusters_np(self, clients_state_dict):
        
        flattened_clients_state_dict = np.array([self.parameter_flatten_client_state_np(cs) for cs in clients_state_dict])
        print('flattened clients states have been prepared.')
        cosine_similarities = cosine_similarity(flattened_clients_state_dict)
        del flattened_clients_state_dict
        gc.collect()
        print('Computed cosine similarities.')
        self.clusterer.fit(cosine_similarities)
        self.cs_labels_ = self.clusterer.labels_
        
        self.clusterer.fit(self.gamma)
        self.gamma_labels_ = self.clusterer.labels_
        
        return
    
    
    def prepare_healing_data(self):
        '''
        This function prepares the healing data using all the clients. This healing data will be 
        used in the later rounds for healing the model and selecting the clients and neurons.
        '''
        
        print('I am in the new noise function. Let\'s see what happens next.')
        
        actual_clients_ratio = self.clients_ratio
        self.clients_ratio = 1.
        
        # sample 100% of the clients and get the weights of the updated models shared by the clients
        self.sample_clients()
        key_flattened_clients_state_dict = self.get_updated_numpy_models_from_clients(pre_str='Getting client data...')
        clients_state_dict = [self.key_unflatten_client_state_np(kfcs) for kfcs in key_flattened_clients_state_dict]
        self.print_out('', overwrite=False)
        
        # update healing data according to the initially shared clients models and heal model
        print('Preparing healing data...')
        self.update_healing_data(key_flattened_clients_state_dict)
        healed_model_state = self.heal_model_from_state(
            self.key_unflatten_client_state_np(np.mean(key_flattened_clients_state_dict, axis=0)), epochs=1
        )
        
        # select the clients/neurons that are believed to be non-poisoned and update healing data on non-poisoned clients/neurons
        self.compute_gammas_np(clients_state_dict, healed_model_state)
        gc.collect()
        print('Computed gammas')
        self.fit_clusters_np(clients_state_dict)
        print('Clustered Clients')
        self.compute_hasnet_indicator()
        
        print(
            '\rClean clients: {:.2f}%, selected_prob: {:.6f}, rejected_prob: {:.6f}'.format(
                100.*np.mean([self.clients_keys[c]=='clean' for c in self.active_clients]),
                self.prob_selected, self.prob_rejected
            )
        )
        
        if np.sum(self.hasnet_indicator) == 0:
            raise ValueError('Length of the selected clients is 0.')
            selected_key_client_states = [self.model.model.state_dict()]
        
        self.update_healing_data([key_flattened_clients_state_dict[c] for c in np.where(self.hasnet_indicator==1)[0]])
        
        self.clients_ratio = actual_clients_ratio
        
        self.print_out('Prepared healing data. Now you can train the model.', overwrite=False)
        
        return
    
    
    def compute_gammas_np(
        self, clients_state_dict, healed_model_state
    ):
        
        healed_initial_state = np.array([self.parameter_flatten_client_state_np(self.heal_model_from_state(self.model.model.state_dict()))])
        initial_model_state = np.array([self.parameter_flatten_client_state_np(self.model.model.state_dict())])
        flattened_clients_states = np.array([self.parameter_flatten_client_state_np(cs) for cs in clients_state_dict])
        flattened_healed_states = np.array([self.parameter_flatten_client_state_np(healed_model_state)])
        
        flattened_aggregated_state = np.mean(flattened_clients_states, axis=0, keepdims=True)
        client_to_initial = flattened_clients_states - initial_model_state
        
        # This is method 2 for estimating the direction of healing
        self.gamma = self.np_cosine_similarity( flattened_aggregated_state-flattened_healed_states, flattened_aggregated_state-flattened_clients_states )
        self.gamma += self.np_cosine_similarity( healed_initial_state-initial_model_state, client_to_initial )
        self.gamma = self.gamma.reshape(len(flattened_clients_states), 1)
        
        return
    
    
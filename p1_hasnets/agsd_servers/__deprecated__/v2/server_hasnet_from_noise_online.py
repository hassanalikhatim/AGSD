import numpy as np


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
    
    
    def prepare_healing_data(self):
        '''
        This function prepares the healing data using all the clients. This healing data will be 
        used in the later rounds for healing the model and selecting the clients and neurons.
        '''
        
        # sample clients and get the weights of the updated models shared by the clients
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
        
        self.print_out('Prepared healing data. Now you can train the model.', overwrite=False)
        
        return
    
    
    
from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset

from _3_federated_learning_utils.servers.server import Server
from _3_federated_learning_utils.clients.all_clients import *

from .helper_data import Helper_Data



implemented_clients = {
    'clean': Client,
    'simple_backdoor': Simple_Backdoor_Client,
    'invisible_backdoor': Invisible_Backdoor_Client,
    'neurotoxin_backdoor': Neurotoxin_Client,
    'iba_backdoor': Irreversible_Backdoor_Client
}


class Helper_Clients(Helper_Data):
    
    def __init__(self,
        my_model_configuration: dict,
        my_server_configuration: dict,
        my_clients_distribution: dict,
        versioning: bool=True,
        verbose: bool=True
    ):
        
        super().__init__(
            my_model_configuration, 
            my_server_configuration, 
            my_clients_distribution, 
            versioning=versioning,
            verbose=verbose
        )
        
        return
    
    
    def prepare_all_clients_with_keys(self, different_clients_configured: dict, client_configurations: dict):
        
        index, clients_with_keys = 0, {}
        for _key in self.my_clients_distribution.keys():
            
            key = different_clients_configured[_key]['type']
            clients_with_keys[_key] = []
            
            for k in range( int(self.my_clients_distribution[_key]) ):
                
                # load default configuration from the configuration file
                this_client_configuration = client_configurations[key]
                # make changes to the default configuration
                for updating_key in different_clients_configured[_key].keys():
                    this_client_configuration[updating_key] = different_clients_configured[_key][updating_key]
                
                clients_with_keys[_key].append(
                    implemented_clients[key](
                        Client_Torch_SubDataset(self.my_data, idxs=self.splits.all_client_indices[index]), 
                        self.global_model.model_configuration,
                        client_configuration=this_client_configuration
                    )
                ); print('\rPreparing clients:', index, end=''); index += 1
            print()
        
        self.clients_with_keys = clients_with_keys
        
        return
    
    
    
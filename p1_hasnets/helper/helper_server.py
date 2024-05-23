from _3_federated_learning_utils.servers.all_servers import *

from .._other_ideas_.server_hasnet_from_heldout import Server_HaSNet_from_HeldOut
from .._other_ideas_.server_hasnet_from_noise import Server_HaSNet_from_Noise
from .._other_ideas_.server_hasnet_from_another_dataset import Server_HaSNet_from_Another_Dataset

from .helper_clients import Helper_Clients



implemented_servers = {
    'simple': Server,
    'dp': Server_DP,
    'deepsight': Server_Deepsight,
    'krum': Server_Krum,
    'foolsgold': Server_FoolsGold,
    'flame': Server_Flame,
    # hasnet servers
    'hasnet_heldout': Server_HaSNet_from_HeldOut,
    'hasnet_noise': Server_HaSNet_from_Noise,
    'hasnet_another': Server_HaSNet_from_Another_Dataset
}


class Helper_Server(Helper_Clients):
    
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
    
    
    def prepare_server(self):
        
        self.server = implemented_servers[self.my_server_configuration['type']](
            self.my_data,
            self.global_model,
            clients_with_keys=self.clients_with_keys,
            configuration=self.my_server_configuration
        )
        
        return
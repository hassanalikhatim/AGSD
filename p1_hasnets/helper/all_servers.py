from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server
from _3_federated_learning_utils.servers.all_servers import *

from ..hasnet_servers._visible_hgsd_heldout import Server_HaSNet_from_HeldOut
from ..hasnet_servers.__deprecated__.v5.server_hasnet_from_noise import Server_HaSNet_from_Noise
from ..hasnet_servers.server_hasnet_from_ood import Sever_HaSNet_from_OOD
from ..hasnet_servers.server_hasnet_from_ood_random_labelling import Sever_HaSNet_from_OOD_Random_Labelling
from ..hasnet_servers.hidden_values_hgsd_id import Hidden_Values_HGSD_ID
from ..hasnet_servers.hgsd_id_initially_undefended import HGSD_ID_Initially_Undefended
from ..hasnet_servers.hgsd_id_for_changing_clients import HGSD_ID_for_Changing_Clients_Analysis



def get_server( 
    my_data: Torch_Dataset, 
    global_model: Torch_Model, 
    clients_with_keys: dict,
    my_server_configuration: dict,
    **kwargs
) -> Server:
    
    implemented_servers = {
        'simple': Server,
        'dp': Server_DP,
        'deepsight': Server_Deepsight,
        'krum': Server_Krum,
        'foolsgold': Server_FoolsGold,
        'flame': Server_Flame,
        'mesas': Server_Mesas,
        # hasnet servers
        'hasnet_heldout': Server_HaSNet_from_HeldOut,
        'hasnet_noise': Server_HaSNet_from_Noise,
        'hasnet_ood': Sever_HaSNet_from_OOD,
        'hasnet_ood_random_labelling': Sever_HaSNet_from_OOD_Random_Labelling,
        'hasnet_hidden_values': Hidden_Values_HGSD_ID,
        'hgsd_id_initially_undefended': HGSD_ID_Initially_Undefended,
        'hgsd_id_for_changing_clients': HGSD_ID_for_Changing_Clients_Analysis
    }
    
    return implemented_servers[my_server_configuration['type']](
        my_data,
        global_model,
        clients_with_keys=clients_with_keys,
        configuration=my_server_configuration,
        **kwargs
    )
    
    
from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server
from _3_federated_learning_utils.servers.all_servers import *

from ..agsd_servers.all_agsd_servers import *
# from ..agsd_servers._visible_agsd_server import AGSD_ID
# from ..agsd_servers.agsd_ood import AGSD_OOD
# from ..agsd_servers.agsd_ood_random_labelling import AGSD_OOD_Random_Labelling
# from ..agsd_servers.agsd_id_hidden_values import AGSD_ID_Hidden_Values
# from ..agsd_servers.agsd_id_initially_undefended import AGSD_ID_Initially_Undefended
# from ..agsd_servers.agsd_id_for_changing_clients import AGSD_ID_for_Changing_Clients_Analysis



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
        # AGSD servers
        'agsd_id': AGSD_ID,
        'agsd_ood': AGSD_OOD,
        'agsd_id_hidden_values': AGSD_ID_Hidden_Values,
        'hgsd_id_initially_undefended': AGSD_ID_Initially_Undefended,
        'hgsd_id_for_changing_clients': AGSD_ID_for_Changing_Clients_Analysis
    }
    
    return implemented_servers[my_server_configuration['type']](
        my_data,
        global_model,
        clients_with_keys=clients_with_keys,
        configuration=my_server_configuration,
        **kwargs
    )
    
    
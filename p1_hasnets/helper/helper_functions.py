from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.datasets import MNIST, CIFAR10, GTSRB
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset

from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor

from _3_federated_learning_utils.clients.all_clients import *
from _3_federated_learning_utils.servers.all_servers import *

from ..adaptive_backdoor_attacks.adversarial_training_client import Adversarial_Training_Backdoor_Client
from ..adaptive_backdoor_attacks.adversarial_optimization_client import Adversarial_Optimization_Backdoor_Client

from _3_federated_learning_utils.splits import Splits



my_datasets = {
    'mnist_toy': MNIST,
    'mnist': MNIST,
    'cifar10': CIFAR10,
    'gtsrb': GTSRB,
    'cifar10_non_sota': CIFAR10,
    'gtsrb_non_sota': GTSRB,
    'cifar10_non_sota_standard_non_iid': CIFAR10,
    'cifar10_non_sota_mesas_non_iid': CIFAR10,
    'gtsrb_non_sota_standard_non_iid1': GTSRB,
    'gtsrb_non_sota_standard_non_iid3': GTSRB,
    'gtsrb_non_sota_standard_non_iid5': GTSRB,
    'gtsrb_non_sota_standard_non_iid7': GTSRB,
    'gtsrb_non_sota_standard_non_iid9': GTSRB,
    'gtsrb_non_sota_mesas_non_iid': GTSRB,
}

implemented_clients = {
    'clean': Client,
    'simple_backdoor': Simple_Backdoor_Client,
    'invisible_backdoor': Invisible_Backdoor_Client,
    'neurotoxin_backdoor': Neurotoxin_Client,
    'iba_backdoor': Irreversible_Backdoor_Client,
    'multiple_target_backdoor': Multiple_Target_Backdoor_Client,
    'multitrigger_multitarget_backdoor': MultiTrigger_MultiTarget_Backdoor_Client,
    'visible_backdoor_initially_good': Visible_Backdoor_Initially_Clean,
    'class_specific_backdoor': Class_Specific_Backdoor_Client,
    'low_confidence_backdoor': Low_Confidence_Backdoor_Client,
    'distributed_backdoor': Distributed_Backdoor_Client,
    'adv_training_backdoor': Adversarial_Training_Backdoor_Client,
    'adv_optimization_backdoor': Adversarial_Optimization_Backdoor_Client
}

# implemented_servers = {
#     'simple': Server,
#     'dp': Server_DP,
#     'deepsight': Server_Deepsight,
#     'krum': Server_Krum,
#     'foolsgold': Server_FoolsGold,
#     'flame': Server_Flame,
#     # hasnet servers
#     'hasnet_heldout': Server_HaSNet_from_HeldOut,
#     'hasnet_noise': Server_HaSNet_from_Noise,
#     'hasnet_another': Sever_HaSNet_from_OOD
# }


def prepare_clean_and_poisoned_data(my_model_configuration: dict) -> list[Torch_Dataset,]:
    
    dataset_name = my_model_configuration['dataset_name']
    
    my_data = my_datasets[dataset_name]()
    poisoned_data = Simple_Backdoor(my_data, backdoor_configuration={'poison_ratio': 0.01})
    
    return my_data, poisoned_data


def prepare_data_splits(my_data, num_clients, split_type: str='iid', alpha: float=0):
    
    splits = Splits(my_data, split_type=split_type, num_clients=num_clients, alpha=alpha)
    splits.split()
    
    return splits


def perpare_all_clients_with_keys(
    my_data: Torch_Dataset,
    global_model: Torch_Model,
    splits: Splits,
    my_clients_distribution: dict,
    different_clients_configured: dict,
    client_configurations: dict,
):
    
    index, clients_with_keys = 0, {}
    for _key in my_clients_distribution.keys():
        
        key = different_clients_configured[_key]['type']
        clients_with_keys[_key] = []
        
        for k in range( int(my_clients_distribution[_key]) ):
            
            # load default configuration from the configuration file
            this_client_configuration = client_configurations[key]
            # make changes to the default configuration
            for updating_key in different_clients_configured[_key].keys():
                this_client_configuration[updating_key] = different_clients_configured[_key][updating_key]
            
            clients_with_keys[_key].append(
                implemented_clients[key](
                    Client_Torch_SubDataset(my_data, idxs=splits.all_client_indices[index]),
                    global_model.model_configuration,
                    client_configuration=this_client_configuration
                )
            ); print('\rPreparing clients:', index, end=''); index += 1
        print()
        
    return clients_with_keys


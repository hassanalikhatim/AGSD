import os
import numpy as np


from ..config import *

from _0_general_ML.data_utils.datasets import MNIST, CIFAR10, GTSRB
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset

from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor

from _3_federated_learning_utils.clients.all_clients import *
from _3_federated_learning_utils.servers.all_servers import *

from _3_federated_learning_utils.splits import Splits

from .._other_ideas_.server_hasnet_from_heldout import Server_HaSNet_from_HeldOut
from .._other_ideas_.server_hasnet_from_noise import Server_HaSNet_from_Noise
from .._other_ideas_.server_hasnet_from_another_dataset import Server_HaSNet_from_Another_Dataset

from ..helper.helper import Helper_Hasnets



my_datasets = {
    'mnist_toy': MNIST,
    'mnist': MNIST,
    'cifar10': CIFAR10,
    'gtsrb': GTSRB
}

implemented_clients = {
    'clean': Client,
    'simple_backdoor': Simple_Backdoor_Client,
    'invisible_backdoor': Invisible_Backdoor_Client,
    'neurotoxin_backdoor': Neurotoxin_Client,
    'iba_backdoor': Irreversible_Backdoor_Client
}

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


def sub_main(
    my_model_configuration: dict,
    my_server_configuration: dict,
    my_clients_distribution: dict
):
    
    # *** preparing some results-related variables ***
    csv_file_path = results_path + my_model_configuration['dataset_name'] + '/csv_file/'
    helper = Helper_Hasnets(
        my_model_configuration=my_model_configuration,
        my_server_configuration=my_server_configuration,
        my_clients_distribution=my_clients_distribution
    )
    experiment_conducted = helper.prepare_paths_and_names(
        results_path, csv_file_path, 
        model_name_prefix='federated', filename='accuracies_and_losses_test.csv'
    )
    
    # *** code starts here ***
    num_clients = my_server_configuration['num_clients']
    print('\nNumber of clean clients: ', my_clients_distribution['clean'])
    
    my_data = my_datasets[my_model_configuration['dataset_name']]()
    poisoned_data = Simple_Backdoor(my_data, backdoor_configuration={'poison_ratio': 0.01})
    
    splits = Splits(my_data, split_type='iid', num_clients=num_clients)
    splits.iid_split()
    
    global_model = Torch_Model(
        my_data, 
        my_model_configuration,
        path=helper.save_path
    )
    
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
    
    
    if experiment_conducted and (not reconduct_conducted_experiments):
        print(
            'Experiment already conducted. '
            'Variable {reconduct_conducted_experiments} is set to False. '
            'Moving on to the next experiment.'
        )
    
    else:
        server = implemented_servers[my_server_configuration['type']](
            my_data,
            global_model,
            clients_with_keys=clients_with_keys,
            configuration=my_server_configuration
        )
            
        for epoch in range(my_model_configuration['epochs']):
            show_str = ''
            
            server.shot(round=epoch)
            show_str += '\rRound {}: '.format(epoch)
            
            helper.evaluate_all_clients_on_test_set(epoch, clients_with_keys, server, poisoned_data)
            server_stats = helper.evaluate_server_statistics(epoch, server)
            show_str += ''.join(['{:s}: {:5f} | '.format(key, server_stats[key]) for key in server_stats.keys()])
            
            print(show_str)
        
        # restore the best test model
        global_model.model.load_state_dict(server.key_unflatten_client_state_np(server.saved_flattened_model))
        server_stats = helper.evaluate_server_statistics(epoch, server)
        show_str = ''.join(['{:s}: {:5f} | '.format(key, server_stats[key]) for key in server_stats.keys()])
        print(show_str)
        
        global_model.save(helper.model_name)
        print('saved model at:', global_model.save_directory + helper.model_name)
    
    return


def main():
    
    total_experiments = len(dataset_names) * len(clients_distributions) * len(server_types)
    current_experiment_number = 0
    
    # iterating over data
    for dataset_name in dataset_names:
        
        # iterating over clients distibutions
        for clients_distribution in clients_distributions:
            
            # iterating over server configurations
            for server_type in server_types:
                
                my_model_configuration = model_configurations[dataset_name].copy()
                my_model_configuration['dataset_name'] = dataset_name
                
                my_server_configuration = server_configurations[different_servers_configured[server_type]['type']].copy()
                for key in different_servers_configured[server_type].keys():
                    my_server_configuration[key] = different_servers_configured[server_type][key]
                my_server_configuration['type'] = different_servers_configured[server_type]['type']
                
                num_clients = my_server_configuration['num_clients']
                my_clients_distribution = clients_distribution.copy()
                for key in my_clients_distribution.keys():
                    my_clients_distribution[key] = int(my_clients_distribution[key] * num_clients)
                my_clients_distribution['clean'] = num_clients - np.sum(
                    np.array( [my_clients_distribution[key] for key in my_clients_distribution.keys()] )
                )
                
                print('\n\nCarrying out experiment: {}/{}'.format(current_experiment_number, total_experiments))
                print('Model configuration:', my_model_configuration)
                print('Server configuration:', my_server_configuration)
                print('Clients configuration:', my_clients_distribution)
                print()
                
                sub_main(
                    my_model_configuration,
                    my_server_configuration,
                    my_clients_distribution
                )
                
                current_experiment_number += 1
    
    return


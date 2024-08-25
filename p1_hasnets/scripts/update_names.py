import numpy as np
import torch
import os


from ..config import *

from _0_general_ML.model_utils.torch_model import Torch_Model

from ..helper.helper_functions import prepare_clean_and_poisoned_data, prepare_data_splits, perpare_all_clients_with_keys
from ..helper.helper import Helper_Hasnets

from utils_.general_utils import replace_all_occurences_in_string, confirm_directory



def update_model_names(
    my_model_configuration: dict,
    my_server_configuration: dict,
    my_clients_distribution: dict
):
    
    experiment_folder_nonfinal = results_path.split('/')[-2]
    experiment_folder_final = 'results_agsd_final'
    results_path_final = f'../../__all_results__/_p1_hasnets/{experiment_folder_final}/'
    new_names_to_old_names = {
        'agsd_id': 'hasnet_heldout',
        'agsd_ood': 'hasnet_ood'
    }
    
    # *** preparing some results-related variables ***
    csv_file_path = results_path + my_model_configuration['dataset_name'] + '/csv_file/'
    helper = Helper_Hasnets(
        my_model_configuration=my_model_configuration,
        my_server_configuration=my_server_configuration,
        my_clients_distribution=my_clients_distribution,
        versioning=False
    )
    helper.prepare_paths_and_names(results_path, csv_file_path, model_name_prefix='federated', filename='accuracies_and_losses_test_compiled.csv')
    
    # *** code starts here ***
    num_clients = my_server_configuration['num_clients']
    print('\nNumber of clean clients: ', my_clients_distribution['clean'])
    
    my_data, poisoned_data = prepare_clean_and_poisoned_data(my_model_configuration)
    splits = prepare_data_splits(my_data, num_clients)
    helper.check_conducted(data_name=my_data.data_name)
    named_updated = helper.experiment_conducted
    
    global_model = Torch_Model(my_data, my_model_configuration, path=helper.save_path)
    global_model.load_weights(global_model.save_directory + helper.model_name)
    
    if not helper.experiment_conducted:
        revised_directory = global_model.save_directory
        revised_model_name = helper.model_name
        
        # print('\n\n\n' + '*'*40)
        # print(global_model.save_directory)
        # print(helper.model_name)
        # print('*'*40)
        for key in new_names_to_old_names.keys():
            revised_directory = replace_all_occurences_in_string(revised_directory, key, new_names_to_old_names[key])
            revised_model_name = replace_all_occurences_in_string(revised_model_name, key, new_names_to_old_names[key])
        # print('\n\n\n' + '*'*40)
        # print(revised_directory)
        # print(revised_model_name)
        # print('*'*40)
            
    else:
        revised_directory = global_model.save_directory
        revised_model_name = helper.model_name
        
    model_found_and_loaded = global_model.load_weights(revised_directory+revised_model_name)
    print(f'Model found and loaded: {model_found_and_loaded}.')
    if model_found_and_loaded and (experiment_folder_final!=experiment_folder_nonfinal):
        named_updated = True
        
        all_folders = global_model.save_directory.split('/')
        revised_folders = []
        for folder in all_folders:
            revised_folders.append(experiment_folder_final if folder==experiment_folder_nonfinal else folder)
        revised_directory_final = '/'.join(revised_folders)
        
        confirm_directory( '/'.join(f'{revised_directory_final}{revised_model_name}'.split('/')[:-1]) )
        if not os.path.isfile(revised_directory_final+revised_model_name+'.pth'):
            global_model.save_weights(revised_directory_final+revised_model_name)
            print(f'Model has been saved at: {revised_directory_final+revised_model_name}')
        else:
            print(f'WOW...! The model was already found at: {revised_directory_final+revised_model_name}')
        
    else:
        print('$'*40, 'THIS MODEL HAS NOT BEEN FOUND.', '$'*40)
        
    return named_updated


def main(orientation=0):
    
    if orientation == 1:
        _experimental_setups = experimental_setups[::-1]
    else:
        _experimental_setups = experimental_setups
    
    total_experiments = [len(es.dataset_names)*len(es.clients_distributions)*len(es.server_types) for es in experimental_setups]
    total_experiments = np.sum(total_experiments)
    current_experiment_number = 0
    not_conducted_experiments = 0
    # iterating over the exprimental_setups
    for experimental_setup in _experimental_setups:
        dataset_names = experimental_setup.dataset_names
        clients_distributions = experimental_setup.clients_distributions
        server_types = experimental_setup.server_types
        
        # setting the orientation of the experiment
        if orientation == 1:
            _dataset_names = dataset_names[::-1]
            _clients_distributions = clients_distributions[::-1]
            _server_types = server_types[::-1]
        else:
            _dataset_names = dataset_names
            _clients_distributions = clients_distributions
            _server_types = server_types
        
        # iterating over data
        for dataset_name in _dataset_names:
            
            # iterating over clients distibutions
            for clients_distribution in _clients_distributions:
                
                # iterating over server configurations
                for server_type in _server_types:
                    
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
                    
                    print('\n\n{} NEW COMPILATION {}'.format( '*' * 30, '*' * 30 ))
                    print('Compilation # {}/{}.'.format(current_experiment_number+1, total_experiments))
                    print('Number of non-conducted experiments:', not_conducted_experiments)
                    print('Model configuration:', my_model_configuration)
                    print('Server configuration:', my_server_configuration)
                    print('Clients configuration:', my_clients_distribution)
                    print('\n')
                    
                    experiment_conducted = update_model_names(
                        my_model_configuration,
                        my_server_configuration,
                        my_clients_distribution
                    )
                    
                    current_experiment_number += 1
                    if not experiment_conducted:
                        not_conducted_experiments += 1
    
    return
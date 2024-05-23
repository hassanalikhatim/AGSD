import numpy as np
import copy


from ..config import *
from ..helper.helper import Helper_Hasnets



dataset_name_correction = {
    'gtsrb': 'gtsrb',
    'cifar10': 'cifar10',
    'mnist': 'mnist'
}


def load_results_from_settings(
    dataset_names, 
    clients_distributions, 
    server_types,
    keys: list[str] = ['test_acc'],
    continued = True,
    results_path_local :str = ''
):
    
    _results_path = results_path_local
    
    results_arr = np.zeros((
        len(dataset_names),
        len(clients_distributions),
        len(server_types),
        len(keys)
    ))
    
    total_results_to_load = len(dataset_names) * len(clients_distributions) * len(server_types)
    load_results_number = 0
    
    # iterating over data
    for d_ind, dataset_name in enumerate(dataset_names):
        for key in dataset_name_correction.keys():
            if key in dataset_name:
                correct_name = dataset_name_correction[key]
        
        # iterating over clients distibutions
        for c_ind, clients_distribution in enumerate(clients_distributions):
            
            # iterating over server configurations
            for s_ind, server_type in enumerate(server_types):
                print('\rLoading results {}/{}'.format(load_results_number, total_results_to_load), end='')
                
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
                
                # *** preparing some results-related variables ***
                csv_file_path = _results_path + dataset_name + '/csv_file/'
                helper = Helper_Hasnets(
                    my_model_configuration=my_model_configuration,
                    my_server_configuration=my_server_configuration,
                    my_clients_distribution=my_clients_distribution,
                    verbose=False, versioning=False
                )
                helper.prepare_paths_and_names(_results_path, csv_file_path, model_name_prefix='federated', filename='accuracies_and_losses_test_compiled.csv')
                # helper.check_conducted(local_verbose=True)
                
                load_columns = [helper.col_name_identifier + '_' + key for key in keys]
                helper.load_columns_in_dictionary(load_columns)
                for k, key in enumerate(keys):
                    load_column = helper.col_name_identifier + '_' + key
                    if load_column in helper.dictionary_to_load.keys():
                        
                        if continued:
                            for i in range(1, len(helper.dictionary_to_load[load_column])):
                                if helper.dictionary_to_load[load_column][i] == -1:
                                    helper.dictionary_to_load[load_column][i] = helper.dictionary_to_load[load_column][i-1]
                        
                        results_arr[d_ind, c_ind, s_ind, k] = helper.dictionary_to_load[load_column][my_model_configuration['epochs'] - 1]
                        
                    else:
                        results_arr[d_ind, c_ind, s_ind, k] = -2.
                
                load_results_number += 1
                
    return results_arr


def load_training_information_of_a_setting(
    dataset_name, 
    clients_distribution, 
    server_type,
    continued = True,
    results_path_local :str = ''
):
    
    _results_path = results_path_local
    
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
    
    # *** preparing some results-related variables ***
    csv_file_path = _results_path + dataset_name + '/csv_file/'
    helper = Helper_Hasnets(
        my_model_configuration=my_model_configuration,
        my_server_configuration=my_server_configuration,
        my_clients_distribution=my_clients_distribution,
        verbose=False, versioning=False
    )
    helper.prepare_paths_and_names(_results_path, csv_file_path, model_name_prefix='federated', filename='accuracies_and_losses_test.csv')
    # helper.check_conducted(local_verbose=True)
    
    helper.load_all_columns_in_dictionary_with_this_string(helper.col_name_identifier)
    
    return helper.col_name_identifier, copy.deepcopy(helper.dictionary_to_load)



import numpy as np
import torch


from ..config import *

from _0_general_ML.model_utils.torch_model import Torch_Model

from ..helper.helper_functions import prepare_clean_and_poisoned_data, prepare_data_splits, perpare_all_clients_with_keys
from ..helper.helper import Helper_Hasnets



def compile_results_shot(
    my_model_configuration: dict,
    my_server_configuration: dict,
    my_clients_distribution: dict
):
    
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
        
    global_model = Torch_Model(my_data, my_model_configuration, path=helper.save_path)
    global_model.load_weights(global_model.save_directory + helper.model_name)
    
    clients_with_keys = perpare_all_clients_with_keys(my_data, global_model, splits, my_clients_distribution, different_clients_configured, client_configurations)
    
    if helper.experiment_conducted:
        print('Experiment already conducted. Loading things.')
        
        test_loss, test_acc = global_model.test_shot(torch.utils.data.DataLoader(
            my_data.test, batch_size=global_model.model_configuration['batch_size']
        ))
        helper.dictionary_to_save[helper.col_name_identifier + '_test_acc'] = [test_acc]
        
        for _key in my_clients_distribution.keys():
            if len(clients_with_keys[_key]) > 0:
                slice_of_dict_results = clients_with_keys[_key][0].test_server_model(poisoned_data, global_model)
                
                for key_ in slice_of_dict_results.keys():
                    active_key = helper.col_name_identifier + '_' + key_
                    if active_key not in helper.dictionary_to_save.keys():
                        helper.dictionary_to_save[active_key] = [slice_of_dict_results[key_]]
        
        for epoch in range(my_model_configuration['epochs']):
            for each_key in helper.dictionary_to_save.keys():
                helper.dictionary_to_save[each_key].append(helper.dictionary_to_save[each_key][-1])
                
        print()
        helper.save_dataframe(force_overwrite=force_overwrite_csv_while_compiling)
        print('dataframe saved at:', helper.csv_path_and_filename)
    
    return helper.experiment_conducted


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
                    
                    experiment_conducted = compile_results_shot(
                        my_model_configuration,
                        my_server_configuration,
                        my_clients_distribution
                    )
                    
                    current_experiment_number += 1
                    if not experiment_conducted:
                        not_conducted_experiments += 1
    
    return
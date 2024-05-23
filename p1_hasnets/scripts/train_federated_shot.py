import torch
from termcolor import colored


# from ..config import *

from _0_general_ML.model_utils.torch_model import Torch_Model

from ..helper.all_servers import get_server
from ..helper.helper_functions import prepare_clean_and_poisoned_data, prepare_data_splits, perpare_all_clients_with_keys
from ..helper.helper import Helper_Hasnets



def federated_shot(
    configuration_variables: dict,
    my_model_configuration: dict,
    my_server_configuration: dict,
    my_clients_distribution: dict
):
    
    # *** preparing some results-related variables ***
    results_path = configuration_variables['results_path']
    versioning = configuration_variables['versioning']
    different_clients_configured = configuration_variables['different_clients_configured']
    client_configurations = configuration_variables['client_configurations']
    reconduct_conducted_experiments = configuration_variables['reconduct_conducted_experiments']
    csv_file_path = results_path + my_model_configuration['dataset_name'] + '/csv_file/'
    
    helper = Helper_Hasnets(
        my_model_configuration=my_model_configuration,
        my_server_configuration=my_server_configuration,
        my_clients_distribution=my_clients_distribution,
        versioning=versioning
    )
    helper.prepare_paths_and_names(results_path, csv_file_path, model_name_prefix='federated', filename='accuracies_and_losses_test.csv')
    
    # *** code starts here ***
    num_clients = my_server_configuration['num_clients']
    print('\nNumber of clean clients: ', my_clients_distribution['clean'])
    
    my_data, poisoned_data = prepare_clean_and_poisoned_data(my_model_configuration)
    splits = prepare_data_splits(my_data, num_clients)
    helper.check_conducted(data_name=my_data.data_name)
    
    if helper.experiment_conducted and (not reconduct_conducted_experiments):
        print('Experiment already conducted. Variable {reconduct_conducted_experiments} is set to False. Moving on to the next experiment.')
    
    else:
        global_model = Torch_Model(my_data, my_model_configuration, path=helper.save_path)
        # global_model.load_weights(global_model.save_directory + helper.model_name)
        
        clients_with_keys = perpare_all_clients_with_keys(my_data, global_model, splits, my_clients_distribution, different_clients_configured, client_configurations)
    
        healing_data_save_path = global_model.save_directory + helper.model_name + '_healing_data/'
        server = get_server(my_data, global_model, clients_with_keys, my_server_configuration, healing_data_save_path=healing_data_save_path)
        
        for epoch in range(my_model_configuration['epochs']):
            server.shot(round=epoch)
            
            helper.evaluate_all_clients_on_test_set(epoch, clients_with_keys, server, poisoned_data)
            server_stats = helper.evaluate_server_statistics(epoch, server)
            show_str, color = helper.get_display_string(server_stats)
            
            print_str = '\r({}-{:5d}) | Round {:3d}: {}'.format(my_server_configuration['type'][:10], my_server_configuration['healing_set_size'], epoch, show_str)
            print(colored(print_str, color))
        
        # restore the best test model
        global_model.model.load_state_dict(server.key_unflatten_client_state_np(server.saved_flattened_model))
        server_stats = helper.evaluate_server_statistics(epoch, server)
        show_str = ''.join(['{:s}: {:.5f} | '.format(key, server_stats[key]) for key in server_stats.keys()])
        print(show_str)
        
        global_model.save(helper.model_name)
        print('saved model at:', global_model.save_directory + helper.model_name)
        
        helper.save_dataframe()
        print('dataframe saved at:', helper.csv_path_and_filename)
    
    return


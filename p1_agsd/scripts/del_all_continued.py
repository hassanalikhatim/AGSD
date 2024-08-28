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
    
    if 'split_type' not in my_model_configuration.keys():
        my_model_configuration['split_type'] = 'iid'
    if 'alpha' not in my_model_configuration.keys():
        my_model_configuration['alpha'] = 0.
    
    # *** preparing some results-related variables ***
    results_path = configuration_variables['results_path']
    versioning = configuration_variables['versioning']
    different_clients_configured = configuration_variables['different_clients_configured']
    client_configurations = configuration_variables['client_configurations']
    reconduct_conducted_experiments = configuration_variables['reconduct_conducted_experiments']
    count_continued_as_conducted = configuration_variables['count_continued_as_conducted']
    save_continued = configuration_variables['save_continued']
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
    splits = prepare_data_splits(my_data, num_clients, split_type=my_model_configuration['split_type'], alpha=my_model_configuration['alpha'])
    helper.check_conducted(data_name=my_data.data_name, count_continued_as_conducted=count_continued_as_conducted)
    
    global_model = Torch_Model(my_data, my_model_configuration, path=helper.save_path)
    global_model.unsave(helper.model_name_cont)
    
    return


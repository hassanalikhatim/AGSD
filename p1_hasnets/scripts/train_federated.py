import numpy as np
import gc

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


from ..config import *

from .train_federated_shot import federated_shot



def main(orientation=0):
    
    # setting the orientation of the experiment
    if orientation == 1:
        _dataset_names = dataset_names[::-1]
        _clients_distributions = clients_distributions[::-1]
        _server_types = server_types[::-1]
    else:
        _dataset_names = dataset_names
        _clients_distributions = clients_distributions
        _server_types = server_types
    
    # starts here
    total_experiments = len(dataset_names) * len(clients_distributions) * len(server_types)
    current_experiment_number = 0; exceptions_met = 0
    all_processes = []
    
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
                
                print('\n\n{} NEW EXPERIMENT {}'.format( '*' * 30, '*' * 30 ))
                print('Carrying out experiment: {}/{}'.format(current_experiment_number, total_experiments))
                print('Exceptions met: {}'.format(exceptions_met))
                print('Model configuration:', my_model_configuration)
                print('Server configuration:', my_server_configuration)
                print('Clients configuration:', my_clients_distribution)
                print('\n')
                
                federated_shot(
                    my_model_configuration,
                    my_server_configuration,
                    my_clients_distribution
                )
                
    return
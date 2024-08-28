import numpy as np
import gc

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


from ..config import *

from .train_federated_shot import federated_shot
# from .del_all_continued import federated_shot

from ..helper.helper_multiprocessing import Helper_Multiprocessing



def sub_main_mp(
    kwargs
):
    
    # *** preparing variables out of kwargs ***
    configuration_variables = kwargs['configuration_variables']
    my_model_configuration = kwargs['my_model_configuration']
    my_server_configuration = kwargs['my_server_configuration']
    my_clients_distribution = kwargs['my_clients_distribution']
    
    federated_shot(
        configuration_variables,
        my_model_configuration,
        my_server_configuration,
        my_clients_distribution
    )
    
    return


def sub_main(
    configuration_variables: dict,
    my_model_configuration: dict,
    my_server_configuration: dict,
    my_clients_distribution: dict
):
    
    return multiprocessing.Process(
        target = sub_main_mp,
        args = (
            {
                'configuration_variables': configuration_variables,
                'my_model_configuration': my_model_configuration,
                'my_server_configuration': my_server_configuration,
                'my_clients_distribution': my_clients_distribution
            },
        )
    )


def main(orientation=0):
    
    if orientation == 1:
        _experimental_setups = experimental_setups[::-1]
    else:
        _experimental_setups = experimental_setups
    
    # starts here
    current_experiment_number = 0; exceptions_met = 0
    all_processes = []
    for experimental_setup in _experimental_setups:
        dataset_names = experimental_setup.dataset_names
        clients_distributions = experimental_setup.clients_distributions
        server_types = experimental_setup.server_types
        
        total_experiments = len(dataset_names) * len(clients_distributions) * len(server_types)
    
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
                    
                    print('\n\n{} NEW EXPERIMENT {}'.format( '*' * 30, '*' * 30 ))
                    print('Carrying out experiment: {}/{}'.format(current_experiment_number, total_experiments))
                    print('Exceptions met: {}'.format(exceptions_met))
                    print('Model configuration:', my_model_configuration)
                    print('Server configuration:', my_server_configuration)
                    print('Clients configuration:', my_clients_distribution)
                    print('\n')
                    
                    configuration_variables = {
                        'results_path': results_path,
                        'versioning': versioning,
                        'different_clients_configured': different_clients_configured,
                        'client_configurations': client_configurations,
                        'reconduct_conducted_experiments': reconduct_conducted_experiments,
                        'count_continued_as_conducted': count_continued_as_conducted,
                        'save_continued': save_continued,
                        'force_overwrite_csv_while_compiling': force_overwrite_csv_while_compiling
                    }
                                
                    all_processes.append(
                        sub_main(
                            configuration_variables,
                            my_model_configuration,
                            my_server_configuration,
                            my_clients_distribution
                        )
                    )
                
    mp_helper = Helper_Multiprocessing(all_processes, shots_at_a_time=shots_at_a_time)
    mp_helper.run_all_processes()
    
    return


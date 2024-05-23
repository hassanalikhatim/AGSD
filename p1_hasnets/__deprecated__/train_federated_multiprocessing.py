import numpy as np
import gc

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


from ..config import *

from .train_federated import sub_main



def sub_main_mp(
    kwargs
):
    
    # *** preparing variables out of kwargs ***
    my_model_configuration = kwargs['my_model_configuration']
    my_server_configuration = kwargs['my_server_configuration']
    my_clients_distribution = kwargs['my_clients_distribution']
    
    sub_main(
        my_model_configuration,
        my_server_configuration,
        my_clients_distribution
    )
        
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
                
                process = multiprocessing.Process(
                    target = sub_main_mp,
                    args = (
                        {
                            'my_model_configuration': my_model_configuration,
                            'my_server_configuration': my_server_configuration,
                            'my_clients_distribution': my_clients_distribution
                        },
                    )
                )
                process.start()
                process.join()
                
                gc.collect()
                    
                current_experiment_number += 1
    
    return


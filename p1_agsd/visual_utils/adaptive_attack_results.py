import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from .loading import load_results_from_settings



nicer_names = {
        
    'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
    'clean': 'Clean', 
    'simple_(poison-0.75)_(scale-2)': 'Simple Backdoor Attack',
    'simple_(num_clients-100)_(clients_ratio-0.1)': 'FedAvg',
    'dp_(num_clients-100)_(clients_ratio-0.1)': 'DP-SGD',
    'krum_(num_clients-100)_(clients_ratio-0.1)': 'Krum',
    'foolsgold_(num_clients-100)_(clients_ratio-0.1)': 'FoolsGold',
    'deepsight_(num_clients-100)_(clients_ratio-0.1)': 'DeepSight',
    'flame_(num_clients-100)_(clients_ratio-0.1)': 'Flame',
    'mesas_(num_clients-100)_(clients_ratio-0.1)': 'MESAS',
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\dfh{}',
    'not_used_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\df{}',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\dfo{}',
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\dfh{}',
    'not_used_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\df{}',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\dfo{}',
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)': '\\dfh{}3',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)': '\\dfo{}3',
    
    'multiple_target_(poison-0.25)_(scale-2)': 'DyBA & 0.25',
    'multiple_target_(poison-0.35)_(scale-2)': 'DyBA & 0.35',
    'multiple_target_(poison-0.45)_(scale-2)': 'DyBA & 0.45',
    'multiple_target_(poison-0.55)_(scale-2)': 'DyBA & 0.55',
    'multiple_target_(poison-0.65)_(scale-2)': 'DyBA & 0.65',
    
    'multitrigger_multitarget_(poison-0.25)_(scale-2)': '& 0.25',
    'multitrigger_multitarget_(poison-0.35)_(scale-2)': '& 0.35',
    'multitrigger_multitarget_(poison-0.45)_(scale-2)': '& 0.45',
    'multitrigger_multitarget_(poison-0.55)_(scale-2)': '& 0.55',
    'multitrigger_multitarget_(poison-0.65)_(scale-2)': '& 0.65',
    
    'low_confidence_(poison-0.25)_(scale-2)': 'LBA & 0.25',
    'low_confidence_(poison-0.35)_(scale-2)': 'LBA & 0.35',
    'low_confidence_(poison-0.45)_(scale-2)': 'LBA & 0.45',
    'low_confidence_(poison-0.55)_(scale-2)': 'LBA & 0.55',
    'low_confidence_(poison-0.65)_(scale-2)': 'LBA & 0.65',
    
    'distributed_(poison-0.25)_(scale-2)': 'DBA & 0.25',
    'distributed_(poison-0.45)_(scale-2)': 'DBA & 0.45',
    
    'adv_training_(poison-0.25)_(scale-2)': 'RBA & 0.25',
    'adv_optimization_(poison-0.25)_(scale-2)': 'PBA & 0.25',
    
}


def adaptive_attacks_evaluation_dyba(dataset_names, results_path_local: str):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    clients_distributions = [
        # DyBA - different strengths of dynamic backdoors
        {'multiple_target_(poison-0.25)_(scale-2)': 0.45},
        {'multiple_target_(poison-0.35)_(scale-2)': 0.45},
        {'multiple_target_(poison-0.45)_(scale-2)': 0.45},
        {'multiple_target_(poison-0.55)_(scale-2)': 0.45},
        {'multiple_target_(poison-0.65)_(scale-2)': 0.45},
        
    ]
    
    server_types = [
        'simple_(num_clients-100)_(clients_ratio-0.1)',
        # 'dp_(num_clients-100)_(clients_ratio-0.1)',
        'krum_(num_clients-100)_(clients_ratio-0.1)',
        # 'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
        # 'deepsight_(num_clients-100)_(clients_ratio-0.1)',
        'flame_(num_clients-100)_(clients_ratio-0.1)',
        # 'mesas_(num_clients-100)_(clients_ratio-0.1)',
        'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(clients_distributions))
    
    table_string = '\n& '
    table_string += ' & '.join([f'{nicer_names[ser_]}' for ser_ in server_types])
    table_string += '\n'
    for c, clients_distribution in enumerate(clients_distributions):
        table_string += '{}'.format(nicer_names[list(clients_distribution.keys())[0]])
        
        for d, dataset_name in enumerate(dataset_names):
            for s, server_type in enumerate(server_types):
                
                result_ = results_arr[d, c, s]
                table_string += ' & {}({})'.format(
                    '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                    '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                )
        
            table_string += '\n'
        table_string += '\\\\\n'
    
    return table_string


def adaptive_attacks_evaluation_mtba(dataset_names, results_path_local: str):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    clients_distributions = [
        # DyBA - different strengths of dynamic backdoors
        {'multitrigger_multitarget_(poison-0.25)_(scale-2)': 0.45},
        {'multitrigger_multitarget_(poison-0.35)_(scale-2)': 0.45},
        {'multitrigger_multitarget_(poison-0.45)_(scale-2)': 0.45},
        {'multitrigger_multitarget_(poison-0.55)_(scale-2)': 0.45},
        {'multitrigger_multitarget_(poison-0.65)_(scale-2)': 0.45},
        
    ]
    
    server_types = [
        'simple_(num_clients-100)_(clients_ratio-0.1)',
        # 'dp_(num_clients-100)_(clients_ratio-0.1)',
        'krum_(num_clients-100)_(clients_ratio-0.1)',
        # 'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
        # 'deepsight_(num_clients-100)_(clients_ratio-0.1)',
        'flame_(num_clients-100)_(clients_ratio-0.1)',
        # 'mesas_(num_clients-100)_(clients_ratio-0.1)',
        'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(clients_distributions))
    
    table_string = '\n& '
    table_string += ' & '.join([f'{nicer_names[ser_]}' for ser_ in server_types])
    table_string += '\n'
    for c, clients_distribution in enumerate(clients_distributions):
        table_string += f'{nicer_names[list(clients_distribution.keys())[0]]}'
        
        for d, dataset_name in enumerate(dataset_names):
            for s, server_type in enumerate(server_types):
                
                result_ = results_arr[d, c, s]
                table_string += ' & {}({})'.format(
                    '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                    '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                )
        
            table_string += '\n'
        table_string += '\\\\\n'
    
    return table_string


def adaptive_attacks_evaluation_lba(dataset_names, results_path_local: str):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    clients_distributions = [
        # LBA
        {'low_confidence_(poison-0.25)_(scale-2)': 0.45},
        {'low_confidence_(poison-0.35)_(scale-2)': 0.45},
        {'low_confidence_(poison-0.45)_(scale-2)': 0.45},
        {'low_confidence_(poison-0.55)_(scale-2)': 0.45},
        {'low_confidence_(poison-0.65)_(scale-2)': 0.45},
        
    ]
    
    server_types = [
        'simple_(num_clients-100)_(clients_ratio-0.1)',
        # 'dp_(num_clients-100)_(clients_ratio-0.1)',
        'krum_(num_clients-100)_(clients_ratio-0.1)',
        # 'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
        # 'deepsight_(num_clients-100)_(clients_ratio-0.1)',
        'flame_(num_clients-100)_(clients_ratio-0.1)',
        # 'mesas_(num_clients-100)_(clients_ratio-0.1)',
        'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        # 'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)',
        # 'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)',
        # 'not_used_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        # 'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
        # 'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
        # 'not_used_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(clients_distributions))
    
    table_string = ''
    for c, clients_distribution in enumerate(clients_distributions):
        table_string += '{}'.format(nicer_names[list(clients_distribution.keys())[0]])
        
        for d, dataset_name in enumerate(dataset_names):
            for s, server_type in enumerate(server_types):
                
                result_ = results_arr[d, c, s]
                table_string += ' & {}({})'.format(
                    '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                    '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                )
        
            table_string += '\n'
        table_string += '\\\\\n'
    
    return table_string


def adaptive_attacks_evaluation_dba(dataset_names, results_path_local: str):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    clients_distributions = [
        # LBA
        {'distributed_(poison-0.25)_(scale-2)': 0.45},
        {'distributed_(poison-0.45)_(scale-2)': 0.45},
    ]
    
    server_types = [
        'simple_(num_clients-100)_(clients_ratio-0.1)',
        # 'dp_(num_clients-100)_(clients_ratio-0.1)',
        'krum_(num_clients-100)_(clients_ratio-0.1)',
        # 'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
        # 'deepsight_(num_clients-100)_(clients_ratio-0.1)',
        'flame_(num_clients-100)_(clients_ratio-0.1)',
        # 'mesas_(num_clients-100)_(clients_ratio-0.1)',
        'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        # 'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)',
        # 'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)',
        # 'not_used_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        # 'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
        # 'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
        # 'not_used_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(clients_distributions))
    
    table_string = ''
    for c, clients_distribution in enumerate(clients_distributions):
        table_string += '{}'.format(nicer_names[list(clients_distribution.keys())[0]])
        
        for d, dataset_name in enumerate(dataset_names):
            for s, server_type in enumerate(server_types):
                
                result_ = results_arr[d, c, s]
                table_string += ' & {}({})'.format(
                    '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                    '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                )
        
            table_string += '\n'
        table_string += '\\\\\n'
    
    return table_string


def adaptive_attacks_evaluation_specifically_desined(dataset_names, results_path_local: str):
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    clients_distributions = [
        
        # Tailored adaptive attacks
        {'adv_training_(poison-0.25)_(scale-2)': 0.45},
        {'adv_optimization_(poison-0.25)_(scale-2)': 0.45},
        
    ]
    
    server_types = [
        'simple_(num_clients-100)_(clients_ratio-0.1)',
        # 'dp_(num_clients-100)_(clients_ratio-0.1)',
        'krum_(num_clients-100)_(clients_ratio-0.1)',
        # 'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
        # 'deepsight_(num_clients-100)_(clients_ratio-0.1)',
        'flame_(num_clients-100)_(clients_ratio-0.1)',
        # 'mesas_(num_clients-100)_(clients_ratio-0.1)',
        'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        # 'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)',
        # 'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)',
        # 'not_used_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        # 'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
        # 'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
        # 'not_used_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(server_types))
    
    table_string = ''
    for c, clients_distribution in enumerate(clients_distributions):
        table_string += '{}'.format(nicer_names[list(clients_distribution.keys())[0]])
        
        for d, dataset_name in enumerate(dataset_names):
            for s, server_type in enumerate(server_types):
                
                result_ = results_arr[d, c, s]
                table_string += ' & {}({})'.format(
                    '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                    '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                )
        
            table_string += '\n'
        table_string += '\\\\\n'
    
    return table_string



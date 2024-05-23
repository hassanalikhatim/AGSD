# Dataset to perform the analysis on
dataset_names = [
    # 'gtsrb_non_sota',
    
    # 'cifar10_non_sota',
    
    # 'cifar10_non_sota_standard_non_iid',
    # 'cifar10_non_sota_mesas_non_iid',
    'gtsrb_non_sota_standard_non_iid',
    'gtsrb_non_sota_mesas_non_iid',
]


# Federated learning configurations
clients_distributions = [
    # {},
    # {'simple_(poison-0.25)_(scale-2)': 0.45},
    
    # # Nature changing clients
    # {'visible_backdoor_initially_good_(poison-0.25)_(scale-1)': 0.45},
    
    # # HGSD at different scales
    # {'simple_(poison-0.25)': 0.45},
    # {'simple_(poison-0.25)_(scale-2)': 0.45},
    # {'simple_(poison-0.25)_(scale-3)': 0.45},
    
    # # Initially Undefended Analysis
    # {'simple_(poison-0.25)_(scale-2)': 0.45},
    
    # Non-iid
    {},
    {'simple_(poison-0.25)_(scale-2)': 0.45},
    # {'invisible_(poison-0.25)_(scale-2)': 0.45},
    
]


server_types = [
    
    # 'hasnet_hidden_values_server',
    
    # 'hgsd_id_for_changing_clients',
    
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    
    # 'hgsd_id_initially_undefended'
    
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    
]

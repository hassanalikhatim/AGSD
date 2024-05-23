# Dataset to perform the analysis on
dataset_names = ['gtsrb_non_sota']


# Federated learning configurations
clients_distributions = [
    # {},
    # {'simple_(poison-0.25)_(scale-2)': 0.45},
    
    {'visible_backdoor_initially_good_(poison-0.25)_(scale-1)': 0.45},
    
    # {'simple_(poison-0.25)': 0.45},
    # {'simple_(poison-0.25)_(scale-2)': 0.45},
    # {'simple_(poison-0.25)_(scale-3)': 0.45},
    
    # {'simple_(poison-0.25)_(scale-2)': 0.45},
    
]


server_types = [
    
    # 'hasnet_hidden_values_server',
    
    # 'hasnet_hidden_values_server_visbile_backdoor_initially_good',
    'hgsd_id_for_changing_clients',
    
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    
    # 'hgsd_id_initially_undefended'
    
]

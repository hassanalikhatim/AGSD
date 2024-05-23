# Dataset to perform the analysis on
dataset_names = [
    # 'mnist',
    # 'cifar10_non_sota',
    'gtsrb_non_sota',
]


# Federated learning configurations
clients_distributions = [
    
    # Multiple adaptive attacks
    {'simple_(poison-0.25)_(scale-2)': 0.42},
    {'simple_(poison-0.25)_(scale-2)': 0.21, 'invisible_(poison-0.25)_(scale-2)': 0.21},
    {'simple_(poison-0.25)_(scale-2)': 0.14, 'invisible_(poison-0.25)_(scale-2)': 0.14, 'neurotoxin_(poison-0.25)_(scale-2)': 0.14},
    
]


server_types = [
    # # SOTA SERVERS
    # 'dp_(num_clients-100)_(clients_ratio-0.1)',
    # 'krum_(num_clients-100)_(clients_ratio-0.1)',
    # 'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
    # 'deepsight_(num_clients-100)_(clients_ratio-0.1)',
    # 'simple_(num_clients-100)_(clients_ratio-0.1)',
    # 'flame_(num_clients-100)_(clients_ratio-0.1)',
    # 'mesas_(num_clients-100)_(clients_ratio-0.1)',
    
    # # HASNET SERVER ANALYSIS - THIS WILL BE A VERY DETAILED ANALYSIS
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)',
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)',
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-4)',
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-4)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)',
]

# Dataset to perform the analysis on
dataset_names = [
    'mnist',
    'cifar10',
    'gtsrb',
]


# Federated learning configurations
clients_distributions = [
    
    # different backdoor clients (one at a time) with 45% backdoor distribution
    {'simple_(poison-0.25)_(scale-2)': 0.45},
    {'neurotoxin_(poison-0.25)_(scale-2)': 0.45},
    {'invisible_(poison-0.25)_(scale-2)': 0.45},
    {'iba_(poison-0.25)_(scale-2)': 0.45},
    
    # no attack
    {},
    
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
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)',
]

# Dataset to perform the analysis on
dataset_names = ['gtsrb_non_sota']


# Federated learning configurations
clients_distributions = [
    {'simple_(poison-0.25)_(scale-2)': 0.45},
    {'simple_(poison-0.25)_(scale-2)': 0.55},
    {'simple_(poison-0.25)_(scale-2)': 0.65},
    {'simple_(poison-0.25)_(scale-2)': 0.75},
    {'simple_(poison-0.25)_(scale-2)': 0.85},
]


server_types = [
    
    # analysis of different healing set size
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    
]
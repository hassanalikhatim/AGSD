# Dataset to perform the analysis on
dataset_names = [
    'mnist_toy',
    'cifar10_toy',
    'gtsrb_toy',
]


# Federated learning configurations
clients_distributions = [
    {'simple_(poison-0.25)_(scale-2)': 0.45},
    {}, # no attack
]


server_types = [
    # SOTA SERVERS
    'dp_(num_clients-100)_(clients_ratio-0.1)',
    'krum_(num_clients-100)_(clients_ratio-0.1)',
    'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
    'deepsight_(num_clients-100)_(clients_ratio-0.1)',
    'simple_(num_clients-100)_(clients_ratio-0.1)',
    'flame_(num_clients-100)_(clients_ratio-0.1)',
    'mesas_(num_clients-100)_(clients_ratio-0.1)',
    
    # AGSD SERVER ANALYSIS - THIS WILL BE A VERY DETAILED ANALYSIS
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
]

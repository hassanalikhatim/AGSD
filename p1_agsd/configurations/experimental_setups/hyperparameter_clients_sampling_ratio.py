# Dataset to perform the analysis on
dataset_names = [
    'gtsrb_non_sota', 
    # 'cifar10_non_sota'
]


# Federated learning configurations
clients_distributions = [
    # {'simple_(poison-0.25)': 0.45},
    {'simple_(poison-0.25)_(scale-2)': 0.45},
]


server_types = [
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)',
    'agsd_id_(num_clients-100)_(clients_ratio-0.2)',
    'agsd_id_(num_clients-100)_(clients_ratio-0.3)',
    'agsd_id_(num_clients-100)_(clients_ratio-0.4)',    
]

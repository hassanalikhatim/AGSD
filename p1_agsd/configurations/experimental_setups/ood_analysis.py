# Dataset to perform the analysis on
dataset_names = ['gtsrb_non_sota', 'cifar10_non_sota']


# Federated learning configurations
clients_distributions = [
    {}, # no attack
    {'simple_(poison-0.25)_(scale-2)': 0.45},
    {'invisible_(poison-0.25)_(scale-2)': 0.45},
]


server_types = [
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'agsd_ood_random_labelling_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
]

# Dataset to perform the analysis on
dataset_names = [
    'gtsrb_non_sota', 
    'cifar10_non_sota'
]


# Federated learning configurations
clients_distributions = [
    {'simple_(poison-0.25)_(scale-2)': 0.45},
]


server_types = [
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)',
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-4)',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-4)',
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-5)',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-5)',
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-6)',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-6)',
]

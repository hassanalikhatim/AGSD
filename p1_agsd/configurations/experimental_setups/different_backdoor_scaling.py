# Dataset to perform the analysis on
dataset_names = [
    'gtsrb_non_sota',
]


# Federated learning configurations
clients_distributions = [
    # AGSD at different scales
    # {'simple_(poison-0.25)': 0.45},
    # {'simple_(poison-0.25)_(scale-2)': 0.45},
    {'simple_(poison-0.25)_(scale-3)': 0.45},
    # {'simple_(poison-0.25)_(scale-5)': 0.45},
]


server_types = [
    # 'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
]

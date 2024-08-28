# Dataset to perform the analysis on
dataset_names = [
    'gtsrb_non_sota',
]


# Federated learning configurations
clients_distributions = [
    # Nature changing clients
    {'visible_backdoor_initially_good_(poison-0.25)_(scale-1)': 0.45},
]


server_types = [
    'agsd_id_for_changing_clients',
]

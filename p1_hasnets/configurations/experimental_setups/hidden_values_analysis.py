# Dataset to perform the analysis on
dataset_names = [
    'gtsrb_non_sota',
]


# Federated learning configurations
clients_distributions = [
    {'simple_(poison-0.25)_(scale-2)': 0.45},
]


server_types = [
    'agsd_id_hidden_values_server_simple_backdoor'
]

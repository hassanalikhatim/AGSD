# Dataset to perform the analysis on
dataset_names = [
    'gtsrb_non_sota', 'cifar10_non_sota'
]


# Federated learning configurations
clients_distributions = [
    # Initially Undefended Analysis
    {'simple_(poison-0.25)_(scale-2)': 0.45},
]


server_types = [
    'agsd_id_initially_undefended_10',
    'agsd_id_initially_undefended_20',
    'agsd_id_initially_undefended_30',
    'agsd_id_initially_undefended_40',
]

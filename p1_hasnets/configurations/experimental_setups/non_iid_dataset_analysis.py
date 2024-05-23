# Dataset to perform the analysis on
dataset_names = [
    # 'gtsrb_non_sota_standard_non_iid_01',
    # 'gtsrb_non_sota_standard_non_iid_03',
    # 'gtsrb_non_sota_standard_non_iid_05',
    # 'gtsrb_non_sota_standard_non_iid_07',
    # 'gtsrb_non_sota_standard_non_iid_09',
    'gtsrb_non_sota_mesas_non_iid',
]


# Federated learning configurations
clients_distributions = [
    
    {'simple_(poison-0.25)_(scale-2)': 0.45},
    
]


server_types = [
    
    # # HASNET SERVER ANALYSIS - THIS WILL BE A VERY DETAILED ANALYSIS
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    
    
]


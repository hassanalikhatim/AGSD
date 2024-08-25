# Dataset to perform the analysis on
dataset_names = [
    # 'gtsrb_non_sota_standard_non_iid1',
    # 'gtsrb_non_sota_standard_non_iid3',
    # 'gtsrb_non_sota_standard_non_iid5',
    # 'gtsrb_non_sota_standard_non_iid7',
    # 'gtsrb_non_sota_standard_non_iid9',
    'gtsrb_non_sota_mesas_non_iid',
]


# Federated learning configurations
clients_distributions = [
    
    {'simple_(poison-0.25)_(scale-2)': 0.45},
    
]


server_types = [
    
    # Can we see how SOTA servers/defenses perform against adaptive attacks?
    # # SOTA SERVERS
    'simple_(num_clients-100)_(clients_ratio-0.1)',
    # 'dp_(num_clients-100)_(clients_ratio-0.1)',
    'krum_(num_clients-100)_(clients_ratio-0.1)',
    # 'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
    # 'deepsight_(num_clients-100)_(clients_ratio-0.1)',
    'flame_(num_clients-100)_(clients_ratio-0.1)',
    # 'mesas_(num_clients-100)_(clients_ratio-0.1)',
    
    # # HASNET SERVER ANALYSIS - THIS WILL BE A VERY DETAILED ANALYSIS
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
       
]


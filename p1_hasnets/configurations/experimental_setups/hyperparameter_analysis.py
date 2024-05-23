# Dataset to perform the analysis on
dataset_names = ['gtsrb_non_sota']


# Federated learning configurations
clients_distributions = [
    {'simple_(poison-0.25)_(scale-2)': 0.45},
]


server_types = [
    
    # analysis of different clients ratio
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)',
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.2)',
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.3)',
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.4)',
    
    # analysis of different healing set size
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)',
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)',
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)',
    
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)',
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)',
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)',
    
    # # # additional but not required
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-1)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-1)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-1)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-1)',
    
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-5)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-5)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-5)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-5)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-5)',
    
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-1)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-1)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-1)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-1)',
    
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-5)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-5)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-5)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-5)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-5)',
    
]

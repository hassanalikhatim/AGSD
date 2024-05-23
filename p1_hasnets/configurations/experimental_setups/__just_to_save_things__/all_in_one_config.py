#########################
# Visible GPU
visible_gpu = '0'
multiprocessing_shot = False
shots_at_a_time = 6
versioning = False

# General configurations
experiment_folder = 'del_them_later_3/'
results_path = '../../__all_results__/_p1_hasnets/' + experiment_folder
reconduct_conducted_experiments = False

# Data configurations
dataset_folder = '../../_Datasets/'
dataset_names = ['gtsrb']

# Federated learning configurations
clients_distributions = [
    # no attack
    {},
    
    # # simple backdoor analysis with multiple poison ratio
    # {'simple_(poison-0.25)_(scale-2)': 0.45},
    # {'simple_(poison-0.50)_(scale-2)': 0.45},
    # {'simple_(poison-0.75)_(scale-2)': 0.45},
    
    # # simple backdoor analysis with different number of backdoor clients
    # {'simple_(poison-0.25)_(scale-2)': 0.1},
    # {'simple_(poison-0.25)_(scale-2)': 0.3},
    # {'simple_(poison-0.25)_(scale-2)': 0.5},
    # {'simple_(poison-0.25)_(scale-2)': 0.7},
    # {'simple_(poison-0.25)_(scale-2)': 0.9},
    
    # different backdoor clients (one at a time) with 45% backdoor distribution
    {'simple_(poison-0.25)_(scale-2)': 0.45},
    {'invisible_(poison-0.25)_(scale-2)': 0.45},
    # {'neurotoxin_(poison-0.25)_(scale-2)': 0.45},
    # {'iba_(poison-0.25)_(scale-2)': 0.45},
    
]

server_types = [
    
    # # 0.1 all servers
    # 'dp_(num_clients-100)_(clients_ratio-0.1)',
    # 'krum_(num_clients-100)_(clients_ratio-0.1)',
    # 'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
    # 'deepsight_(num_clients-100)_(clients_ratio-0.1)',
    # 'simple_(num_clients-100)_(clients_ratio-0.1)',
    # 'flame_(num_clients-100)_(clients_ratio-0.1)',
    # 'mesas_(num_clients-100)_(clients_ratio-0.1)',
    
    # # HASNET SERVER ANALYSIS - THIS WILL BE A VERY DETAILED ANALYSIS
    # # basic
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    # 'hasnet_ood_random_labelling_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    
    # # analysis of different clients ratio
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.2)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.3)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.4)',
    
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.2)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.3)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.4)',
    
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.2)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.3)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.4)',
    
    # # analysis of different healing set size
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)',
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)',
    
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)',
    
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)',
    
    # # analysis of hasnet attack iterations
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(hasnet_attack_iterations-10)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(hasnet_attack_iterations-30)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(hasnet_attack_iterations-50)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(hasnet_attack_iterations-100)',
    # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(hasnet_attack_iterations-500)',
    
    # # analysis of different healing epochs
    # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)',
    # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)',
    
]

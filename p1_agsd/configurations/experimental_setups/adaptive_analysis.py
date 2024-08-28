# Dataset to perform the analysis on
dataset_names = [
    # 'cifar10_non_sota',
    'gtsrb_non_sota',
]


# Federated learning configurations
clients_distributions = [
    
    # LBA
    {'low_confidence_(poison-0.25)_(scale-2)': 0.45},
    {'low_confidence_(poison-0.35)_(scale-2)': 0.45},
    {'low_confidence_(poison-0.45)_(scale-2)': 0.45},
    {'low_confidence_(poison-0.55)_(scale-2)': 0.45},
    {'low_confidence_(poison-0.65)_(scale-2)': 0.45},
    
    # MTBA- different strengths of dynamic backdoors
    {'multitrigger_multitarget_(poison-0.25)_(scale-2)': 0.45},
    {'multitrigger_multitarget_(poison-0.35)_(scale-2)': 0.45},
    {'multitrigger_multitarget_(poison-0.45)_(scale-2)': 0.45},
    {'multitrigger_multitarget_(poison-0.55)_(scale-2)': 0.45},
    {'multitrigger_multitarget_(poison-0.65)_(scale-2)': 0.45},
    
    # DBA - Distributed Backdoor Attack
    {'distributed_(poison-0.25)_(scale-2)': 0.45},
    {'distributed_(poison-0.45)_(scale-2)': 0.45},
    
    # Tailored adaptive attacks
    {'adv_training_(poison-0.25)_(scale-2)': 0.45},
    {'adv_optimization_(poison-0.25)_(scale-2)': 0.45},
    
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
    
    # AGSD SERVER ANALYSIS
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    
]

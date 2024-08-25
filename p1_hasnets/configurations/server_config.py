# server configurations
num_clients = 100
clients_ratio = 0.1

simple_server_configuration = {
    'num_clients': num_clients,
    'clients_ratio': clients_ratio
}
dp_server_configuration = {
    'num_clients': num_clients,
    'clients_ratio': clients_ratio,
    'differential': 1e-5,
    'clip_value': 1
}
foolsgold_server_configuration = {
    'num_clients': num_clients,
    'clients_ratio': clients_ratio,
    'k': 1
}
flame_server_configuration = {
    'num_clients': num_clients,
    'clients_ratio': clients_ratio,
    'lambda': 0.000012,
    'differential': 0.001
}
mesas_server_configuration = {
    'num_clients': num_clients,
    'clients_ratio': clients_ratio
}
agsd_id_server_configuration = {
    'num_clients': num_clients,
    'clients_ratio': clients_ratio,
    'healing_set_size': 50,
    'epsilon': 0.,
    'healing_epochs': 1
}
agsd_ood_server_configuration = {
    'num_clients': num_clients,
    'clients_ratio': clients_ratio,
    'healing_set_size': 50,
    'epsilon': 0.,
    'healing_epochs': 1,
    'hasnet_attack_iterations': 30  # this is not used in AGSD.
}

server_configurations = {
    'simple': simple_server_configuration,
    'dp': dp_server_configuration,
    'deepsight': simple_server_configuration,
    'krum': simple_server_configuration,
    'foolsgold': foolsgold_server_configuration,
    'flame': flame_server_configuration,
    'mesas': mesas_server_configuration,
    # agsd_id servers
    'agsd_id': agsd_id_server_configuration,
    'hasnet_noise': agsd_ood_server_configuration,
    'agsd_ood': agsd_ood_server_configuration,
    'agsd_ood_random_labelling': agsd_ood_server_configuration,
    'agsd_ood_old': agsd_ood_server_configuration,
    'agsd_id_hidden_values': agsd_id_server_configuration,
    'agsd_id_initially_undefended': agsd_id_server_configuration,
    'agsd_id_for_changing_clients': agsd_id_server_configuration
}



different_servers_configured = {
    'simple_(num_clients-100)_(clients_ratio-0.1)': {'type': 'simple'},
    'simple_(num_clients-100)_(clients_ratio-0.2)': {'type': 'simple', 'clients_ratio': 0.2},
    'simple_(num_clients-100)_(clients_ratio-0.3)': {'type': 'simple', 'clients_ratio': 0.3},
    'simple_(num_clients-100)_(clients_ratio-0.4)': {'type': 'simple', 'clients_ratio': 0.4},
    'simple_(num_clients-100)_(clients_ratio-0.5)': {'type': 'simple', 'clients_ratio': 0.5},
    
    'simple_(num_clients-500)_(clients_ratio-0.1)': {'type': 'simple', 'num_clients': 500},
    
    'dp_(num_clients-100)_(clients_ratio-0.1)': {'type': 'dp', 'clients_ratio': 0.1},
    'dp_(num_clients-100)_(clients_ratio-0.2)': {'type': 'dp', 'clients_ratio': 0.2},
    'dp_(num_clients-100)_(clients_ratio-0.3)': {'type': 'dp', 'clients_ratio': 0.3},
    'dp_(num_clients-100)_(clients_ratio-0.4)': {'type': 'dp', 'clients_ratio': 0.4},
    'dp_(num_clients-100)_(clients_ratio-0.5)': {'type': 'dp', 'clients_ratio': 0.5},
    
    'krum_(num_clients-100)_(clients_ratio-0.1)': {'type': 'krum', 'clients_ratio': 0.1},
    'krum_(num_clients-100)_(clients_ratio-0.2)': {'type': 'krum', 'clients_ratio': 0.2},
    'krum_(num_clients-100)_(clients_ratio-0.3)': {'type': 'krum', 'clients_ratio': 0.3},
    'krum_(num_clients-100)_(clients_ratio-0.4)': {'type': 'krum', 'clients_ratio': 0.4},
    'krum_(num_clients-100)_(clients_ratio-0.5)': {'type': 'krum', 'clients_ratio': 0.5},
    
    'foolsgold_(num_clients-100)_(clients_ratio-0.1)': {'type': 'foolsgold', 'clients_ratio': 0.1},
    'foolsgold_(num_clients-100)_(clients_ratio-0.2)': {'type': 'foolsgold', 'clients_ratio': 0.2},
    'foolsgold_(num_clients-100)_(clients_ratio-0.3)': {'type': 'foolsgold', 'clients_ratio': 0.3},
    'foolsgold_(num_clients-100)_(clients_ratio-0.4)': {'type': 'foolsgold', 'clients_ratio': 0.4},
    'foolsgold_(num_clients-100)_(clients_ratio-0.5)': {'type': 'foolsgold', 'clients_ratio': 0.5},
    
    'deepsight_(num_clients-100)_(clients_ratio-0.1)': {'type': 'deepsight', 'clients_ratio': 0.1},
    'deepsight_(num_clients-100)_(clients_ratio-0.2)': {'type': 'deepsight', 'clients_ratio': 0.2},
    'deepsight_(num_clients-100)_(clients_ratio-0.3)': {'type': 'deepsight', 'clients_ratio': 0.3},
    'deepsight_(num_clients-100)_(clients_ratio-0.4)': {'type': 'deepsight', 'clients_ratio': 0.4},
    'deepsight_(num_clients-100)_(clients_ratio-0.5)': {'type': 'deepsight', 'clients_ratio': 0.5},
    
    'flame_(num_clients-100)_(clients_ratio-0.1)': {'type': 'flame', 'clients_ratio': 0.1},
    'flame_(num_clients-100)_(clients_ratio-0.2)': {'type': 'flame', 'clients_ratio': 0.2},
    'flame_(num_clients-100)_(clients_ratio-0.3)': {'type': 'flame', 'clients_ratio': 0.3},
    'flame_(num_clients-100)_(clients_ratio-0.4)': {'type': 'flame', 'clients_ratio': 0.4},
    'flame_(num_clients-100)_(clients_ratio-0.5)': {'type': 'flame', 'clients_ratio': 0.4},
    
    'mesas_(num_clients-100)_(clients_ratio-0.1)': {'type': 'mesas', 'clients_ratio': 0.1},
    'mesas_(num_clients-100)_(clients_ratio-0.2)': {'type': 'mesas', 'clients_ratio': 0.2},
    'mesas_(num_clients-100)_(clients_ratio-0.3)': {'type': 'mesas', 'clients_ratio': 0.3},
    'mesas_(num_clients-100)_(clients_ratio-0.4)': {'type': 'mesas', 'clients_ratio': 0.4},
    'mesas_(num_clients-100)_(clients_ratio-0.5)': {'type': 'mesas', 'clients_ratio': 0.4},
    
    # DIFFERENT AGSD ID HELDOUT CONFIGURATIONS FOR EXPERIMENTS
    # * * different number of clients
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)': {'type': 'agsd_id', 'clients_ratio': 0.1},
    'agsd_id_(num_clients-100)_(clients_ratio-0.2)': {'type': 'agsd_id', 'clients_ratio': 0.2},
    'agsd_id_(num_clients-100)_(clients_ratio-0.3)': {'type': 'agsd_id', 'clients_ratio': 0.3},
    'agsd_id_(num_clients-100)_(clients_ratio-0.4)': {'type': 'agsd_id', 'clients_ratio': 0.4},
    'agsd_id_(num_clients-100)_(clients_ratio-0.5)': {'type': 'agsd_id', 'clients_ratio': 0.5},
    
    # * * different healing set sizes
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 10},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 50},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 100},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 500},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 1000},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 5000},
    
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-1)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 10, 'healing_epochs': 1},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 50, 'healing_epochs': 1},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-1)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 100, 'healing_epochs': 1},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-1)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 500, 'healing_epochs': 1},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-1)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 1000, 'healing_epochs': 1},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)_(healing_epochs-1)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 5000, 'healing_epochs': 1},
    
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-5)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 10, 'healing_epochs': 5},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-5)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 50, 'healing_epochs': 5},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-5)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 100, 'healing_epochs': 5},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-5)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 500, 'healing_epochs': 5},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-5)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 1000, 'healing_epochs': 5},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)_(healing_epochs-5)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 5000, 'healing_epochs': 5},
    
    # DIFFERENT AGSD OOD CONFIGURATIONS FOR EXPERIMENTS
    # * * different number of clients
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)': {'type': 'agsd_ood', 'clients_ratio': 0.1},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.2)': {'type': 'agsd_ood', 'clients_ratio': 0.2},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.3)': {'type': 'agsd_ood', 'clients_ratio': 0.3},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.4)': {'type': 'agsd_ood', 'clients_ratio': 0.4},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.5)': {'type': 'agsd_ood', 'clients_ratio': 0.5},
    
    # * * different healing set sizes
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 10},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 50},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 100},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 500},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 1000},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 5000},
    
    # agsd_id ood with 1 healing epoch
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-1)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 10, 'healing_epochs': 1},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 50, 'healing_epochs': 1},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-1)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 100, 'healing_epochs': 1},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-1)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 500, 'healing_epochs': 1},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-1)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 1000, 'healing_epochs': 1},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)_(healing_epochs-1)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 5000, 'healing_epochs': 1},
    
    # agsd_id ood with 5 healing epoch
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-5)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 10, 'healing_epochs': 5},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-5)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 50, 'healing_epochs': 5},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-5)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 100, 'healing_epochs': 5},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-5)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 500, 'healing_epochs': 5},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-5)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 1000, 'healing_epochs': 5},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)_(healing_epochs-5)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 5000, 'healing_epochs': 5},
    
    # SPECIAL SERVERS FOR SPECIAL PURPOSES
    'agsd_ood_random_labelling_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': {'type': 'agsd_ood_random_labelling', 'clients_ratio': 0.1, 'healing_set_size': 50, 'num_clients': 100},
    'agsd_id_hidden_values_server': {'type': 'agsd_id_hidden_values'},
    'agsd_id_hidden_values_server_simple_backdoor': {'type': 'agsd_id_hidden_values', 'suffix_phrase': 'backdoor'},
    'agsd_id_hidden_values_server_visbile_backdoor_initially_good': {'type': 'agsd_id_hidden_values', 'suffix_phrase': 'visible_backdoor_initially_good'},
    'agsd_id_for_changing_clients': {'type': 'agsd_id_for_changing_clients', 'suffix_phrase': 'visible_backdoor_initially_good', 'bad_clients_remain_good_epoch': 30},
    
    # more than two clusters
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 50, 'n_clusters': 3},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 50, 'n_clusters': 3},
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-4)': {'type': 'agsd_id', 'clients_ratio': 0.1, 'healing_set_size': 50, 'n_clusters': 4},
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-4)': {'type': 'agsd_ood', 'clients_ratio': 0.1, 'healing_set_size': 50, 'n_clusters': 4},
    
    
    # initially undefended hgsd
    'agsd_id_initially_undefended_10': {'type': 'agsd_id_initially_undefended', 'defense_start_round': 10},
    'agsd_id_initially_undefended_20': {'type': 'agsd_id_initially_undefended', 'defense_start_round': 20},
    'agsd_id_initially_undefended_30': {'type': 'agsd_id_initially_undefended', 'defense_start_round': 30},
    'agsd_id_initially_undefended_40': {'type': 'agsd_id_initially_undefended', 'defense_start_round': 40},
    'agsd_id_initially_undefended_50': {'type': 'agsd_id_initially_undefended', 'defense_start_round': 50},
    
}

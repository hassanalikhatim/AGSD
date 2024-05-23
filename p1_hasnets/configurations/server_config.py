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
hasnet_heldout_server_configuration = {
    'num_clients': num_clients,
    'clients_ratio': clients_ratio,
    'healing_set_size': 50,
    'epsilon': 0.,
    'healing_epochs': 1
}
hasnet_noise_server_configuration = {
    'num_clients': num_clients,
    'clients_ratio': clients_ratio,
    'healing_set_size': 50,
    'epsilon': 0.,
    'healing_epochs': 1,
    'hasnet_attack_iterations': 30
}

server_configurations = {
    'simple': simple_server_configuration,
    'dp': dp_server_configuration,
    'deepsight': simple_server_configuration,
    'krum': simple_server_configuration,
    'foolsgold': foolsgold_server_configuration,
    'flame': flame_server_configuration,
    'mesas': mesas_server_configuration,
    # hasnet servers
    'hasnet_heldout': hasnet_heldout_server_configuration,
    'hasnet_noise': hasnet_noise_server_configuration,
    'hasnet_ood': hasnet_noise_server_configuration,
    'hasnet_ood_random_labelling': hasnet_noise_server_configuration,
    'hasnet_ood_old': hasnet_noise_server_configuration,
    'hasnet_hidden_values': hasnet_heldout_server_configuration,
    'hgsd_id_initially_undefended': hasnet_heldout_server_configuration,
    'hgsd_id_for_changing_clients': hasnet_heldout_server_configuration
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
    
    # DIFFERENT HASNET HELDOUT CONFIGURATIONS FOR EXPERIMENTS
    # * * different number of clients
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.2)': {'type': 'hasnet_heldout', 'clients_ratio': 0.2},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.3)': {'type': 'hasnet_heldout', 'clients_ratio': 0.3},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.4)': {'type': 'hasnet_heldout', 'clients_ratio': 0.4},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.5)': {'type': 'hasnet_heldout', 'clients_ratio': 0.5},
    
    # * * different healing set sizes
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 10},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 50},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 100},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 500},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 1000},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 5000},
    
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-1)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 10, 'healing_epochs': 1},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 50, 'healing_epochs': 1},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-1)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 100, 'healing_epochs': 1},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-1)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 500, 'healing_epochs': 1},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-1)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 1000, 'healing_epochs': 1},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)_(healing_epochs-1)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 5000, 'healing_epochs': 1},
    
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-5)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 10, 'healing_epochs': 5},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-5)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 50, 'healing_epochs': 5},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-5)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 100, 'healing_epochs': 5},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-5)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 500, 'healing_epochs': 5},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-5)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 1000, 'healing_epochs': 5},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)_(healing_epochs-5)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 5000, 'healing_epochs': 5},
    
    # DIFFERENT HASNET NOISE CONFIGURATIONS FOR EXPERIMENTS
    # * * different number of clients
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)': {'type': 'hasnet_noise', 'clients_ratio': 0.1},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.2)': {'type': 'hasnet_noise', 'clients_ratio': 0.2},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.3)': {'type': 'hasnet_noise', 'clients_ratio': 0.3},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.4)': {'type': 'hasnet_noise', 'clients_ratio': 0.4},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.5)': {'type': 'hasnet_noise', 'clients_ratio': 0.5},
    
    # * * different healing set sizes
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)': {'type': 'hasnet_noise', 'clients_ratio': 0.1, 'healing_set_size': 10},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': {'type': 'hasnet_noise', 'clients_ratio': 0.1, 'healing_set_size': 50},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': {'type': 'hasnet_noise', 'clients_ratio': 0.1, 'healing_set_size': 100},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)': {'type': 'hasnet_noise', 'clients_ratio': 0.1, 'healing_set_size': 500},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)': {'type': 'hasnet_noise', 'clients_ratio': 0.1, 'healing_set_size': 1000},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)': {'type': 'hasnet_noise', 'clients_ratio': 0.1, 'healing_set_size': 5000},
    
    # * * different attack_iterations
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(hasnet_attack_iterations-10)': {'type': 'hasnet_noise', 'clients_ratio': 0.1, 'healing_set_size': 100, 'hasnet_attack_iterations': 10},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(hasnet_attack_iterations-30)': {'type': 'hasnet_noise', 'clients_ratio': 0.1, 'healing_set_size': 100, 'hasnet_attack_iterations': 30},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(hasnet_attack_iterations-50)': {'type': 'hasnet_noise', 'clients_ratio': 0.1, 'healing_set_size': 100, 'hasnet_attack_iterations': 50},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(hasnet_attack_iterations-100)': {'type': 'hasnet_noise', 'clients_ratio': 0.1, 'healing_set_size': 100, 'hasnet_attack_iterations': 100},
    'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(hasnet_attack_iterations-500)': {'type': 'hasnet_noise', 'clients_ratio': 0.1, 'healing_set_size': 100, 'hasnet_attack_iterations': 500},
    
    # DIFFERENT HASNET OOD CONFIGURATIONS FOR EXPERIMENTS
    # DIFFERENT HASNET HELDOUT CONFIGURATIONS FOR EXPERIMENTS
    # * * different number of clients
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)': {'type': 'hasnet_ood', 'clients_ratio': 0.1},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.2)': {'type': 'hasnet_ood', 'clients_ratio': 0.2},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.3)': {'type': 'hasnet_ood', 'clients_ratio': 0.3},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.4)': {'type': 'hasnet_ood', 'clients_ratio': 0.4},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.5)': {'type': 'hasnet_ood', 'clients_ratio': 0.5},
    
    # * * different healing set sizes
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 10},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 50},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 100},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 500},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 1000},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 5000},
    
    # hasnet ood with 1 healing epoch
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-1)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 10, 'healing_epochs': 1},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 50, 'healing_epochs': 1},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-1)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 100, 'healing_epochs': 1},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-1)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 500, 'healing_epochs': 1},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-1)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 1000, 'healing_epochs': 1},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)_(healing_epochs-1)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 5000, 'healing_epochs': 1},
    
    # hasnet ood with 5 healing epoch
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-5)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 10, 'healing_epochs': 5},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-5)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 50, 'healing_epochs': 5},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-5)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 100, 'healing_epochs': 5},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-5)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 500, 'healing_epochs': 5},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-5)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 1000, 'healing_epochs': 5},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)_(healing_epochs-5)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 5000, 'healing_epochs': 5},
    
    # special servers for special purposes
    'hasnet_ood_random_labelling_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': {'type': 'hasnet_ood_random_labelling', 'clients_ratio': 0.1, 'healing_set_size': 50, 'num_clients': 100},
    'hasnet_hidden_values_server': {'type': 'hasnet_hidden_values'},
    'hasnet_hidden_values_server_simple_backdoor': {'type': 'hasnet_hidden_values', 'suffix_phrase': 'backdoor'},
    'hasnet_hidden_values_server_visbile_backdoor_initially_good': {'type': 'hasnet_hidden_values', 'suffix_phrase': 'visible_backdoor_initially_good'},
    'hgsd_id_for_changing_clients': {'type': 'hgsd_id_for_changing_clients', 'suffix_phrase': 'visible_backdoor_initially_good', 'bad_clients_remain_good_epoch': 30},
    
    # more than two clusters
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 50, 'n_clusters': 3},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 50, 'n_clusters': 3},
    'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-4)': {'type': 'hasnet_heldout', 'clients_ratio': 0.1, 'healing_set_size': 50, 'n_clusters': 4},
    'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-4)': {'type': 'hasnet_ood', 'clients_ratio': 0.1, 'healing_set_size': 50, 'n_clusters': 4},
    
    
    # initially undefended hgsd
    'hgsd_id_initially_undefended_10': {'type': 'hgsd_id_initially_undefended', 'defense_start_round': 10},
    'hgsd_id_initially_undefended_20': {'type': 'hgsd_id_initially_undefended', 'defense_start_round': 20},
    'hgsd_id_initially_undefended_30': {'type': 'hgsd_id_initially_undefended', 'defense_start_round': 30},
    'hgsd_id_initially_undefended_40': {'type': 'hgsd_id_initially_undefended', 'defense_start_round': 40},
    'hgsd_id_initially_undefended_50': {'type': 'hgsd_id_initially_undefended', 'defense_start_round': 50},
    
}

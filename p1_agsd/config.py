from .configurations import model_config as model_configurations_config
from .configurations import client_config as clients_configurations_config
from .configurations import server_config as server_configurations_config

from .configurations.experimental_setups import all_experimental_setups



# experimental setup
experimental_setups = [
    all_experimental_setups.comparison_with_sota,
    
    # all_experimental_setups.hyperparameter_clients_sampling_ratio,
    # all_experimental_setups.hyperparameter_heldout_set_size,
    # all_experimental_setups.hyperparameter_backdoor_scaling_constant,
    # all_experimental_setups.hyperparameter_backdoored_clients_ratio,
    
    # all_experimental_setups.backdoor_and_defend,
    # all_experimental_setups.hidden_values_analysis,
    # all_experimental_setups.nature_changing_clients,
    
    # all_experimental_setups.adaptive_analysis,
    
    # all_experimental_setups.non_iid_dataset_analysis,
]

#########################
# Visible GPU
visible_gpu = '0'
multiprocessing_shot = False
shots_at_a_time = 10
versioning = False

# General configurations
# experiment_folder = 'results_to_del/'
# experiment_folder = 'results_hgsd_(std_transfer)_(efficient_sampling)_(guided_clustering)/'
# experiment_folder = 'results_agsd_noniid/'
experiment_folder = 'results_agsd_final/'
results_path = '../../__all_results__/_p1_hasnets/' + experiment_folder
reconduct_conducted_experiments = True
count_continued_as_conducted = True
save_continued = True
force_overwrite_csv_while_compiling = True

# Data configurations
dataset_folder = '../../_Datasets/'

# model configurations
model_configurations = model_configurations_config.model_configurations

# federated learning configurations
# clients configurations
client_configurations = clients_configurations_config.client_configurations
different_clients_configured = clients_configurations_config.different_clients_configured

# server configurations
server_configurations = server_configurations_config.server_configurations
different_servers_configured = server_configurations_config.different_servers_configured


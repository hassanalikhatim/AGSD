from .configurations import model_config as model_configurations_config
from .configurations import client_config as clients_configurations_config
from .configurations import server_config as server_configurations_config

from .configurations.experimental_setups import all_experimental_setups



# experimental setup
experimental_setup = all_experimental_setups.one_big_analysis

#########################
# Visible GPU
visible_gpu = '0'
multiprocessing_shot = False
shots_at_a_time = 9
versioning = False

# General configurations
experiment_folder = 'experiment_spectral_old/'
results_path = '../../__all_results__/_p1_hasnets/' + experiment_folder
reconduct_conducted_experiments = True

# Data configurations
dataset_names = experimental_setup.dataset_names
dataset_folder = '../../_Datasets/'

# model configurations
model_configurations = model_configurations_config.model_configurations

# federated learning configurations
# clients configurations
clients_distributions = experimental_setup.clients_distributions
client_configurations = clients_configurations_config.client_configurations
different_clients_configured = clients_configurations_config.different_clients_configured

# server configurations
server_types = experimental_setup.server_types
server_configurations = server_configurations_config.server_configurations
different_servers_configured = server_configurations_config.different_servers_configured


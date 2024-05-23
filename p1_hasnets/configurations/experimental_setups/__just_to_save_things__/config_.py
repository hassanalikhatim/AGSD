from .configurations import model_config as model_configurations_config
from .configurations import client_config as clients_configurations_config
from .configurations import server_config as server_configurations_config

from .... import config



# general configuration
multiprocessing_shot = config.multiprocessing_shot
shots_at_a_time = config.shots_at_a_time
versioning = config.versioning

results_path = config.results_path
reconduct_conducted_experiments = config.reconduct_conducted_experiments

# data configuration
dataset_names = config.experimental_setup.dataset_names
dataset_folder = config.dataset_folder

# model configuration
model_configurations = model_configurations_config.model_configurations


# federated learning configurations
# clients configurations
client_configurations = clients_configurations_config.client_configurations
different_clients_configured = clients_configurations_config.different_clients_configured
clients_distributions = config.experimental_setup.clients_distributions

# server configurations
server_configurations = server_configurations_config.server_configurations
different_servers_configured = server_configurations_config.different_servers_configured
server_types = config.experimental_setup.server_types

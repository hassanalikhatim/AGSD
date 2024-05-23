"""
Entangled Configurations
"""
# model configuarations
mnist_model_configuration = {
    'model_architecture': 'mnist_cnn',
    'learning_rate': 1e-3,
    'loss_fn': 'crossentropy',
    'epochs': 20,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.5
}
cifar10_model_configuration = {
    'model_architecture': 'cifar10_vgg11',
    'learning_rate': 1e-4,
    'loss_fn': 'crossentropy',
    'epochs': 50,
    'batch_size': 256,
    'optimizer': 'adam',
    'momentum': 0.5,
    'weight_decay': 0.
}
gtsrb_model_configuration = {
    'model_architecture': 'resnet50_gtsrb',
    'learning_rate': 1e-4,
    'loss_fn': 'crossentropy',
    'epochs': 100,
    'batch_size': 256,
    'optimizer': 'adam',
    'momentum': 0.5
}

model_configurations = {
    'mnist': mnist_model_configuration,
    'gtsrb': gtsrb_model_configuration,
    'cifar10': cifar10_model_configuration
}


# client configurations
clean_client_configuration = {
    'local_epochs': 1
}
simple_backdoor_client_configuration = {
    'local_epochs': 1,
    'trigger': None,
    'target': 0,
    'poison_ratio': 0.8
}
flip_client_configuration = {
    'local_epochs': 1
}
neurotoxin_client = {
    'mask_ratio': 0.02,
    'local_epochs': 5,
    'trigger': None,
    'target': 0,
    'poison_ratio': 0.8
}

client_configurations = {
    'clean': clean_client_configuration,
    'simple_backdoor': simple_backdoor_client_configuration,
    'invisible_backdoor': simple_backdoor_client_configuration,
    'neurotoxin': neurotoxin_client,
    'flip': flip_client_configuration
}


# server configurations
flame_server_configuration = {
    'lambda': 0.0001,
    'differential': 0.001
}
dp_server_configuration = {
    'differential': 1e-5,
    'clip_value': 1
}
foolsgold_server_configuration = {
    'k': 1
}
hasnet_server_configuration = {
    'healing_epochs': 5
}

server_configurations = {
    'simple': None,
    'dp': dp_server_configuration,
    'flame': flame_server_configuration,
    'foolsgold': foolsgold_server_configuration,
    'deepsight': None,
    'krum': None,
    'hasnet': hasnet_server_configuration
}



# #########################################
# General configurations
results_path = 'hasnets/__all_results__/results_initial/'
gpu_number = 1

# Data configurations
dataset_folder = '../../_Datasets/'
data_name = 'mnist'
preferred_image_size = -1

# Federated configurations
num_clients = 100
clients_ratio = 0.1 # Number of clients selected in each iteration
data_distribution = 'random_uniform'
train_size = 0.85
clients_distribution = {
    'simple_backdoor': 0.,
    'invisible_backdoor': 0.,
    'neurotoxin': 0.1
}

# Server configurations
server_type = 'simple'

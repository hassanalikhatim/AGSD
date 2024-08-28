import torch

from _0_general_ML.data_utils.datasets import MNIST, GTSRB, CIFAR10
from _0_general_ML.model_utils.torch_model import Torch_Model
from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset
from _3_federated_learning_utils.splits import Splits
from _3_federated_learning_utils.clients.client import Client
from _3_federated_learning_utils.clients.backdoor_attacks.simple_backdoor_client import Simple_Backdoor_Client
from _3_federated_learning_utils.clients.backdoor_attacks.neurotoxin_client import Neurotoxin_Client
from _3_federated_learning_utils.clients.clients_node import Client_Node
from _3_federated_learning_utils.clients.backdoor_attacks.iba_client import Irreversible_Backdoor_Client

from .motivational_server import Motivational_Server
from .motivational_flame import Motivational_Server_Flame
from .motivational_mesas import Motivational_Server_Mesas
from .motivational_hasnets_h import Motivational_AGSD_ID

from ..helper.helper import Helper_Hasnets



servers_to_use = {
    'simple': Motivational_Server,
    'flame': Motivational_Server_Flame,
    'mesas': Motivational_Server_Mesas,
    'agsd_id': Motivational_AGSD_ID
}


def shot(server_name):
    my_data = MNIST()
    my_model = Torch_Model(
        my_data, 
        model_configuration = {
            'model_architecture': 'mnist_cnn',
            'learning_rate': 1e-4,
            'loss_fn': 'crossentropy',
            'gpu_number': 0,
            'epochs': 50
        }
    )
    # my_model.train(epochs=4)

    # let us define some federated learning parameters
    my_server_configuration = {
        'num_clients': 100,
        'clients_ratio': 0.1
    }
    my_server_configuration['type'] = server_name

    my_client_configuration = {
        'poison_ratio': 0.25,
        'scale': 2,
        'mask_ratio': 0.02,
        'trigger_inversion_iterations': 500
    }

    # split data among all clients
    data_splits = Splits(my_data, num_clients=my_server_configuration['num_clients'])
    data_splits.iid_split()
    c_index = 0


    bad_client_key = 'simple_(poison-0.25)_(scale-2)'
    all_clients = {'clean': [], bad_client_key: []}
    n_backdoor = 0.45
    # adding node clients
    for i in range( int(n_backdoor*my_server_configuration['num_clients']) ):
        all_clients[bad_client_key].append(
            Simple_Backdoor_Client(
                Client_Torch_SubDataset(my_data, idxs=data_splits.all_client_indices[c_index]),
                global_model_configuration=my_model.model_configuration,
                client_configuration=my_client_configuration
            )
        ); c_index += 1
        
    for i in range( int((1-n_backdoor)*my_server_configuration['num_clients']) ):
        all_clients['clean'].append(
            Client(
                Client_Torch_SubDataset(my_data, idxs=data_splits.all_client_indices[c_index]),
                global_model_configuration=my_model.model_configuration,
                client_configuration={}
            )
        ); c_index += 1


    my_model.model_configuration['dataset_name'] = my_data.data_name
    helper = Helper_Hasnets(
        my_model_configuration=my_model.model_configuration,
        my_server_configuration=my_server_configuration,
        my_clients_distribution={'simple_(poison-0.25)_(scale-2)': 45, 'clean': 55},
        versioning=False
    )
    helper.prepare_paths_and_names(
        'p1_hasnets/motivational_analysis/results/', 'p1_hasnets/motivational_analysis/csv_file/', 
        model_name_prefix='federated', filename='accuracies_and_losses_test_2.csv'
    )
    
    my_server = servers_to_use[server_name](
        my_data,
        my_model,
        all_clients,
        configuration=my_server_configuration
    )
    
    poisoned_data = Simple_Backdoor(my_data, backdoor_configuration={'poison_ratio': 0.1})
    
    for epoch in range(my_model.model_configuration['epochs']):
        show_str = ''
        
        my_server.shot(round=epoch)
        show_str += '\rRound {}: '.format(epoch)
        
        helper.evaluate_all_clients_on_test_set(epoch, all_clients, my_server, poisoned_data)
        server_stats = helper.evaluate_server_statistics(epoch, my_server)
        show_str += ''.join(['{:s}: {:5f} | '.format(key, server_stats[key]) for key in server_stats.keys()])
        
        print(show_str)
        
    helper.save_dataframe(force_overwrite=True)
    print('dataframe saved at:', helper.csv_path_and_filename)
    
    return



def main():
    
    # for server_name in ['simple', 'flame', 'mesas', 'agsd_id']:
    for server_name in ['agsd_id']:
        shot(server_name)
        
    return


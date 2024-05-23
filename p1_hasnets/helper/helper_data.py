from _0_general_ML.data_utils.datasets import MNIST, CIFAR10, GTSRB
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor

from _3_federated_learning_utils.splits import Splits

from .helper_paths import Helper_Paths



my_datasets = {
    'mnist_toy': MNIST,
    'mnist': MNIST,
    'cifar10': CIFAR10,
    'gtsrb': GTSRB
}


class Helper_Data(Helper_Paths):
    
    def __init__(
        self,
        my_model_configuration: dict,
        my_server_configuration: dict,
        my_clients_distribution: dict,
        versioning: bool=True,
        verbose: bool=True
    ):
        
        super().__init__(
            my_model_configuration, 
            my_server_configuration, 
            my_clients_distribution, 
            versioning=versioning,
            verbose=verbose
        )
        
        return
    
    
    def prepare_clean_and_poisoned_data(self):
        
        self.my_data = my_datasets[self.my_model_configuration['dataset_name']]()
        self.poisoned_data = Simple_Backdoor(self.my_data, backdoor_configuration={'poison_ratio': 0.01})
        
        return
    
    def prepare_data_splits(self):
        
        num_clients = self.my_server_configuration['num_clients']
        print('\nNumber of clean clients: ', self.my_clients_distribution['clean'])
        
        self.splits = Splits(self.my_data, split_type='iid', num_clients=num_clients)
        self.splits.iid_split()
        
        return
    
    
    def prepare_model(self):
        
        self.global_model = Torch_Model(
            self.my_data, 
            self.my_model_configuration,
            path=self.save_path
        )
    
        return
    
    
import torch
from sklearn.utils import shuffle
import numpy as np


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset



class Splits:
    
    def __init__(
        self, 
        data: Torch_Dataset, 
        split_type: str = 'iid', 
        num_clients: int = 100,
        alpha: float = 0.
    ):
        
        self.data = data
        
        self.num_clients = num_clients
        self.split_type = split_type
        self.alpha = alpha
        
        return
    
    
    def split(self):
        
        splits_dictionary = {
            'iid': self.iid_split,
            'standard_non_iid': self.standard_client_non_iid_split,
            'mesas_non_iid': self.mesas_inter_client_non_iid_split
        }
        
        return splits_dictionary[self.split_type]()
    

    def iid_split(self):
        
        self.split_size = int(self.data.train.__len__()/self.num_clients)
        
        self.all_client_indices = []
        for k in range(self.num_clients):
            indices = [i for i in range(self.data.train.__len__())]
            indices = shuffle(indices)[:self.split_size]
            
            self.all_client_indices.append(indices)
        
        return
    
    
    def standard_client_non_iid_split(self):
        '''
        This is the standard non-iid data distribution setting.
        '''
        
        num_classes = self.data.get_num_classes()
        num_samples = self.data.train.__len__()
        
        # How many different groups of clients [num_groups] are you trying to make? Clients in each group will carry IID data.
        # Note that the [num_groups] <= [num_clients] and [num_groups] <= [num_classes]
        num_groups = min(num_classes, self.num_clients)
        clients_in_each_group = self.num_clients // num_groups if self.num_clients%num_groups==0 else self.num_clients // num_groups + 1
        shuffled_clients = shuffle(np.arange(self.num_clients))
        data_for_each_group = num_samples // num_groups
        
        # Step 1: Find out the total number of samples for each class in the train set, and create an array for each class
        # containing indices of the samples of that class in the dataset.
        indices_as_per_classes = {k: [] for k in range(num_classes)}
        for sample_number in range(num_samples):
            x, y = self.data.train.__getitem__(sample_number)
            indices_as_per_classes[y].append(sample_number)
        shuffled_indices_per_class = {k: shuffle(indices_as_per_classes[k]) for k in range(num_classes)}
        
        # Step 2: Assign [indices_as_per_class] to each group.
        # If [alpha]=0, these indices should be randomly assigned.
        # If [alpha]=1, these indices should be non-IID assigned.
        # 
        # First assign pre-determined classes to clients. If [alpha]=0, no classes are assigned beforehand.
        indices_of_each_group = {k: [] for k in range(num_groups)}
        clients_group_number = -1. * np.ones((self.num_clients))
        for group_ in range(num_groups):
            assign_to_group = min(int(self.alpha*len(indices_as_per_classes[group_]))+1, data_for_each_group)
            indices_of_each_group[group_] += shuffled_indices_per_class[group_][:assign_to_group]
            shuffled_indices_per_class[group_] = shuffled_indices_per_class[group_][assign_to_group:]
            
            clients_group_number[ shuffled_clients[:clients_in_each_group] ] = group_
            shuffled_clients = shuffled_clients[clients_in_each_group:]
            
        assert -1. not in clients_group_number, 'No client should be left unassigned.'
        
        # Step 3: Randomly ssign all the unassigned indices.
        # Computing the unassigned indices
        all_non_assigned_indices = []
        for class_ in shuffled_indices_per_class.keys():
            all_non_assigned_indices += shuffled_indices_per_class[class_]
        all_non_assigned_indices = shuffle(all_non_assigned_indices)
        
        for group_ in range(num_groups):
            n_indices_preassigned = np.clip(data_for_each_group-len(indices_of_each_group[group_]), 0, data_for_each_group).astype('int')
            indices_of_each_group[group_] += all_non_assigned_indices[:n_indices_preassigned]
            all_non_assigned_indices = all_non_assigned_indices[n_indices_preassigned:]
        shuffled_indices_of_each_group = {group_: shuffle(indices_of_each_group[group_]) for group_ in range(num_groups)}
        
        # Step 4: Randomly assign indices to each client from its group
        each_client_data_size = self.data.train.__len__() // self.num_clients
        self.all_client_indices = []
        for client_ in range(self.num_clients):
            group_ = clients_group_number[client_]
            self.all_client_indices.append(shuffled_indices_of_each_group[group_][:each_client_data_size])
            shuffled_indices_of_each_group[group_] = shuffled_indices_of_each_group[group_][each_client_data_size:]
            # self.all_client_indices.append(shuffle(indices_of_each_group[group_])[:each_client_data_size])
        
        return
    
    
    def mesas_inter_client_non_iid_split(self):
        '''
        This is the non-iid data distribution from MESAS
        '''
        
        num_classes = self.data.get_num_classes()
        num_samples = self.data.train.__len__()
        
        indices_as_per_classes = {k: [] for k in range(num_classes)}
        for sample_number in range(num_samples):
            x, y = self.data.train.__getitem__(sample_number)
            indices_as_per_classes[y].append(sample_number)
        shuffled_indices_per_class = {k: shuffle(indices_as_per_classes[k]) for k in range(num_classes)}
        
        each_client_classes = np.random.randint(10, 100, size=(self.num_clients, num_classes)).astype('float')
        each_client_classes /= np.sum(each_client_classes, axis=0, keepdims=True)
        
        self.all_client_indices = []
        for client_ in range(self.num_clients):
            this_client_indices = []
            for class_ in range(num_classes):
                num_samples_assigned_to_client_from_this_class = int(each_client_classes[client_, class_]*len(indices_as_per_classes[class_]))
                this_client_indices += shuffle(shuffled_indices_per_class[class_])[:num_samples_assigned_to_client_from_this_class]
                
                shuffled_indices_per_class[class_] = shuffled_indices_per_class[class_][num_samples_assigned_to_client_from_this_class:]
            self.all_client_indices.append(shuffle(this_client_indices))
        
        return
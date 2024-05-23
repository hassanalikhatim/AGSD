import torch
from sklearn.utils import shuffle
import numpy as np


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset



class Splits:
    
    def __init__(
        self, 
        data: Torch_Dataset, 
        split_type='iid', num_clients=100
    ):
        
        self.data = data
        
        self.num_clients = num_clients
        self.split_type = split_type
        
        return
    

    def iid_split(self):
        
        self.split_size = int(self.data.train.__len__()/self.num_clients)
        
        self.all_client_indices = []
        for k in range(self.num_clients):
            indices = [i for i in range(self.data.train.__len__())]
            indices = shuffle(indices)[:self.split_size]
            
            self.all_client_indices.append(indices)
        
        return
    
    
    
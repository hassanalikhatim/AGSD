import numpy as np
import copy
import torch
import os


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from ..agsd_servers.all_agsd_servers import AGSD_ID



class Motivational_AGSD_ID(AGSD_ID):
    
    def __init__(
        self, 
        data: Torch_Dataset,
        model: Torch_Model,
        clients_with_keys: dict={}, 
        configuration: dict={}
    ):
        
        super().__init__(data, model, clients_with_keys=clients_with_keys, configuration=configuration)
        
        self.epoch = 0
        sampling_path = 'p1_hasnets/motivational_analysis/clients_samplings.npy'
        if os.path.isfile(sampling_path):
            self.clients_samplings = np.load(sampling_path)
            print('Loaded sampling file.')
        else:
            self.clients_samplings = []
            for i in range(50):
                self.clients_samplings.append(np.random.choice(
                    len(self.clients), int(self.clients_ratio*len(self.clients)), replace=False
                ))
            self.clients_samplings = np.array(self.clients_samplings)
            np.save(sampling_path, self.clients_samplings)
        
        return
    
    
    def sample_clients(self):
        
        self.active_clients = self.clients_samplings[self.epoch]
        self.epoch += 1
        
        return
    
    
    
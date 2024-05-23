import torch
import numpy as np

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset
from _0_general_ML.data_utils.datasets import Fashion_MNIST

from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.adversarial_attacks.pgd import PGD

from .server_hasnet_from_ood import Sever_HaSNet_from_OOD



class Sever_HaSNet_from_OOD_Random_Labelling(Sever_HaSNet_from_OOD):
    
    def __init__(
        self, 
        data: Torch_Dataset,
        model: Torch_Model,
        clients_with_keys: dict={},
        configuration: dict={},
        **kwargs
    ):
        
        super().__init__(data, model, clients_with_keys, configuration)
        
        return
    
    
    def get_samples(self):
        
        self.channeled_samples, labels = super().get_samples()
        self.labels = np.random.randint(np.min(labels), np.max(labels), size=(len(labels)))
        
        return self.channeled_samples, self.labels
    
    
    
import numpy as np

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset
from _0_general_ML.data_utils.datasets import Fashion_MNIST

from _0_general_ML.model_utils.torch_model import Torch_Model

from .server_hasnet_from_noise import Server_HaSNet_from_Noise



class Server_HaSNet_from_Another_Dataset(Server_HaSNet_from_Noise):
    
    def __init__(
        self, 
        data: Torch_Dataset,
        model: Torch_Model,
        clients_with_keys: dict={},
        configuration: dict={}
    ):
        
        super().__init__(data, model, clients_with_keys, configuration)
        self.random_sample_indices = []
        
        return
    
    
    def get_samples(self):
        
        channels, height, width = self.data.train.__getitem__(0)[0].shape
        another_dataset = Fashion_MNIST(preferred_size=height)
        
        if not len(self.random_sample_indices):
            self.random_sample_indices = np.random.choice(
                another_dataset.train.__len__(), self.configuration['healing_set_size'], replace=False
            )
        
        samples = []
        for random_index in self.random_sample_indices:
            samples.append(another_dataset.train.__getitem__(random_index)[0].detach().cpu().numpy())
        samples = np.array(samples)
        
        channeled_samples = []
        for c in range(channels):
            channeled_samples.append(samples[:, 0])
        channeled_samples = np.array(channeled_samples)
        channeled_samples = np.transpose(channeled_samples, axes=(1, 0, 2, 3))
        
        return channeled_samples
    
    
    
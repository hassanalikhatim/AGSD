import torch
import numpy as np

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset
from _0_general_ML.data_utils.datasets import CIFAR100, Fashion_MNIST, CIFAR10, GTSRB

from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.adversarial_attacks.pgd import PGD

from ._visible_agsd_server import AGSD_ID



class AGSD_OOD(AGSD_ID):
    
    def __init__(
        self, 
        data: Torch_Dataset,
        model: Torch_Model,
        clients_with_keys: dict={},
        configuration: dict={},
        **kwargs
    ):
        
        self.random_sample_indices = []
        
        super().__init__(data, model, clients_with_keys, configuration)
        
        self.prepare_healing_data()
        self.server_name = f'hgsd_(ood-{{{self.real_healing_data.train.__len__()}}})'
        
        return
    
    
    def prepare_healing_data(self):
        '''
        This function prepares the healing data using all the clients. This healing data will be 
        used in the later rounds for healing the model and selecting the clients and neurons.
        '''
        
        channeled_samples, labels = self.get_samples()
        
        self.real_healing_data = Torch_Dataset(data_name=self.data.data_name)
        self.real_healing_data.train = torch.utils.data.TensorDataset(torch.tensor(channeled_samples), torch.tensor(labels))
        self.real_healing_data.test = self.data.test
        
        return
    
    
    def get_samples(self):
        
        an_item = self.data.train.__getitem__(0)[0]
        channels, height, width = an_item.shape
        if 'gtsrb' not in self.model.data.data_name:
            another_dataset = GTSRB(preferred_size=(height, width))
        else:
            another_dataset = CIFAR10(preferred_size=(height, width))
        num_classes = len(self.data.get_class_names())
        
        print('Preparing the healing dataset...')
        samples = []; labels = []
        while len(self.random_sample_indices) < self.configuration['healing_set_size']:
            random_index = np.random.randint(another_dataset.train.__len__())
            x, y = another_dataset.train.__getitem__(random_index)
            if y < num_classes:
                samples.append(x.detach().cpu().numpy()); labels.append(y)
                self.random_sample_indices.append(random_index)
            print('\r Len of healing data:', len(self.random_sample_indices), end='')
        samples = np.array(samples); self.labels = np.array(labels)
        
        if samples.shape[1] != channels:
            channeled_samples = []
            for c in range(channels):
                channeled_samples.append(samples[:, 0])
            channeled_samples = np.array(channeled_samples)
            self.channeled_samples = np.transpose(channeled_samples, axes=(1, 0, 2, 3))
        else:
            self.channeled_samples = samples
        
        return self.channeled_samples, self.labels
    
    
    
import torch
import torchvision
import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.local_config import dataset_folder



class GTSRB(Torch_Dataset):
    
    def __init__(
        self,
        preferred_size: int=(50, 50),
        **kwargs
    ):
        
        super().__init__(
            data_name='gtsrb',
            preferred_size=preferred_size
        )
        
        if not self.preferred_size:
            self.preferred_size = (80, 80)
        
        self.renew_data()
        self.num_classes = len(self.get_class_names())
        
        return
    
    
    def renew_data(
        self, **kwargs
    ):
        
        pytorch_transforms = []
        if self.preferred_size:
            pytorch_transforms = [torchvision.transforms.Resize(self.preferred_size)]
        pytorch_transforms += [torchvision.transforms.ToTensor()]
        pytorch_transforms += [torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        
        self.train = torchvision.datasets.GTSRB(
            dataset_folder, split='train', download=True,
            transform=torchvision.transforms.Compose(pytorch_transforms)
        )
        
        self.test = torchvision.datasets.GTSRB(
            dataset_folder, split='test', download=True,
            transform=torchvision.transforms.Compose(pytorch_transforms)
        )
        
        return
    
    
    def get_class_names(self):
        return np.arange(len(np.unique( [self.train[i][1] for i in range(self.train.__len__())] )))
    
    
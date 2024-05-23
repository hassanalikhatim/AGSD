import numpy as np
import torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.local_config import dataset_folder



class CIFAR100(Torch_Dataset):
    
    def __init__(
        self,
        preferred_size: int=0,
        **kwargs
    ):
        
        super().__init__(
            data_name='cifar100',
            preferred_size=preferred_size
        )
        
        self.renew_data()
        self.num_classes = len(self.get_class_names())
        
        return
    
    
    def renew_data(
        self, **kwargs
    ):
        
        test_transform = []
        if self.preferred_size:
            test_transform = [torchvision.transforms.Resize(self.preferred_size)]
        test_transform += [torchvision.transforms.ToTensor()] # convert the image to tensor so that it can work with torch
        test_transform += [torchvision.transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))] #Normalize all the images
        
        train_transform = []
        if self.preferred_size:
            train_transform = [torchvision.transforms.Resize(self.preferred_size)]
        train_transform += [torchvision.transforms.RandomHorizontalFlip()] # FLips the image w.r.t horizontal axis
        train_transform += [torchvision.transforms.RandomRotation((-7,7))]     #Rotates the image to a specified angel
        train_transform += [torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8,1.2))] #Performs actions like zooms, change shear angles.
        train_transform += [torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)] # Set the color params
        train_transform += [torchvision.transforms.ToTensor()] # convert the image to tensor so that it can work with torch
        train_transform += [torchvision.transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))] #Normalize all the images
        # train_transform += [torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        
        self.train = torchvision.datasets.CIFAR100(
            dataset_folder, train=True, download=True,
            transform=torchvision.transforms.Compose(train_transform)
        )
        
        self.test = torchvision.datasets.CIFAR100(
            dataset_folder, train=False, download=True,
            transform=torchvision.transforms.Compose(test_transform)
        )
        
        return


    def get_class_names(self):
        return np.arange(len(np.unique( [self.train[i][1] for i in range(self.train.__len__())] )))

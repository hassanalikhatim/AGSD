import numpy as np
import torch
import torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.local_config import dataset_folder

from .imagenet_class_mapping import imagenet_class_mapping_dictionary



class Kaggle_Imagenet(Torch_Dataset):
    
    def __init__(
        self,
        preferred_size: int=(224, 224),
        **kwargs
    ):
        
        super().__init__(
            data_name='kaggle_imagenet',
            preferred_size=preferred_size
        )
        
        self.dataset_folder = dataset_folder+'imagenet/ILSVRC/Data/CLS-LOC/'
        
        self.renew_data()
        
        print('Calculating the number of classes...', end='')
        self.num_classes = 1000 # len(self.get_class_names())
        print(f'\rThe number of classes in {self.data_name} is: {self.num_classes}.')
        
        return
    
    
    def renew_data(self, **kwargs):
        
        test_transform = []
        if self.preferred_size:
            print(f'Preferred input size is {self.preferred_size}.')
            test_transform = [torchvision.transforms.Resize(self.preferred_size)]
        test_transform += [torchvision.transforms.ToTensor()] # convert the image to tensor so that it can work with torch
        test_transform += [torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        
        train_transform = []
        if self.preferred_size:
            train_transform = [torchvision.transforms.Resize(self.preferred_size)]
        # train_transform += [torchvision.transforms.RandomCrop(32)]
        # train_transform += [torchvision.transforms.RandomHorizontalFlip()]
        train_transform += [torchvision.transforms.ToTensor()]
        train_transform += [torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        
        full_dataset = torchvision.datasets.ImageFolder(self.dataset_folder+'train/', transform=torchvision.transforms.Compose(train_transform))
        
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        self.train, self.test = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        
        return
    
    
    def compute_class_names(self):
        
        dataloader = torch.utils.data.DataLoader(self.train, batch_size=256)
        len_dataloader = len(dataloader)
        print('Length of dataloader is: ', len_dataloader)
        
        labels_list = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                labels_list += target.view(-1)
                labels_list = list(np.unique(labels_list))
                
                print_str = f'Processing: [{batch_idx+1}/{len_dataloader}]({(100.*(batch_idx+1)/len_dataloader):3.1f}%). Found classes: {len(labels_list)}'
                print('\r'+print_str, end='')
        
        return np.unique(labels_list)
    
    
    def get_class_names(self): return [f'{imagenet_class_mapping_dictionary[k].split(',')[0]}' for k in imagenet_class_mapping_dictionary.keys()]
    def get_num_classes(self): return self.num_classes
    
    
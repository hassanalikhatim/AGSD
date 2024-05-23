import torchvision


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.local_config import dataset_folder



class MNIST(Torch_Dataset):
    
    def __init__(
        self,
        preferred_size: int=0,
        **kwargs
    ):
        
        super().__init__(
            data_name='mnist',
            preferred_size=preferred_size
        )
        
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
        pytorch_transforms += [torchvision.transforms.Normalize((0.1307,), (0.3081,))]
        
        self.train = torchvision.datasets.MNIST(
            dataset_folder, train=True, download=True,
            transform=torchvision.transforms.Compose(pytorch_transforms)
        )
        
        self.test = torchvision.datasets.MNIST(
            dataset_folder, train=False, download=True,
            transform=torchvision.transforms.Compose(pytorch_transforms)
        )
        
        return
    
    
    def get_class_names(self):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

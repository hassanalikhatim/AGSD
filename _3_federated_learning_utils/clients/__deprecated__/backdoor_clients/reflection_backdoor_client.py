import numpy as np
import torch
import torchvision


from _3_federated_learning_utils.clients.client import Client

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset



class Reflection_Backdoor_Client(Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_architecture=None,
        configuration=None, client_name='default'
    ):
        
        super().__init__(
            data, global_model_architecture=global_model_architecture,
            configuration=configuration, client_name=client_name
        )
        
        self.poison_ratio = configuration['poison_ratio']
        
        self.kernel_size = 2*int(np.max(data.train.data[0].shape)/2)-1
        
        self.set_trigger(trigger=configuration['trigger'], target=configuration['target'])
        self.poison_data()
        
        return
    
    
    def poison_data(self):
        
        poison_indices = np.random.choice(
            self.data.train.data.shape[0],
            int(self.poison_ratio * self.data.train.data.shape[0])
        )
        
        self.data.train = self.poison(self.data.train, poison_indices)
        
        return
    
    
    def poison(self, _data, poison_indices=None):
        
        if poison_indices is None:
            poison_indices = [i for i in range(len(_data.data))]
        
        # Stamp trigger to the images
        _data.data[poison_indices] += self.trigger
        _data.data[poison_indices] = torch.clip(
            _data.data[poison_indices], 
            torch.min(_data.data), 
            torch.max(_data.data)
        )
        
        # Poison the labels of the tampered images
        _data.targets[poison_indices] = self.target
        
        return _data
    
    
    def reflection_in_focus(
        self, data
    ):
        
        alpha = np.random.uniform(0.05, 0.4, size=len(data))
        
        return alpha * data.clone()
    
    
    def reflection_out_of_focus(
        self, data
    ):
        
        guassian_transform_function = torchvision.transforms.GaussianBlur(self.kernel_size, sigma=(1.0, 5.0))
        
        blurred_images = []
        for i in range(len(data)):
            blurred_images.append(guassian_transform_function(data.clone())[0])
        
        return torch.stack(blurred_images)
    
    
    def reflection_ghost_effect(
        self, data
    ):
        
        two_pulsed_images = []
        for i in range(len(data)):
            two_pulse_kernel = self.torch_two_pulse_kernel(self.kernel_size)
            two_pulsed_images.append(self.torch_convolution(data.clone(), two_pulse_kernel)[:, 0])
        
        return torch.stack(two_pulsed_images)
    
    
    def torch_two_pulse_kernel(
        self, kernel_size
    ):

        kernel = torch.zeros((kernel_size, kernel_size))
        kernel[int(kernel_size/2), int(kernel_size/2)] = 1.
        
        alpha = np.random.uniform(0.15, 0.35)
        delta = np.random.randint(3, 8)
        kernel[int(kernel_size/2)+delta, int(kernel_size/2)+delta] = alpha
        
        return kernel
    

    def torch_convolution(
        self, 
        batch_input, kernel
    ):
        
        batch_input = torch.unsqueeze(batch_input, 1).float()
        kernel = torch.unsqueeze(torch.unsqueeze(kernel, 0), 1)
        
        return torch.nn.functional.conv2d(batch_input, kernel, padding='same')
    
    
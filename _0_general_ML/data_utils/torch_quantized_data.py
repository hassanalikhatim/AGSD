import numpy as np
import torch



class Quantized_Dataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        data, n_bits=3
    ):
        
        self.data = data
        self.n_bits = n_bits
        
        self.repetitions = int(8/self.n_bits)
        self.base = 2**self.n_bits
        self.divider = 2**(self.repetitions-1)
        
        input_len = self.data.__getitem__(0)[0].view(-1).shape[0]
        additional_values = input_len % self.repetitions
        if additional_values:
            self.appended_values = torch.zeros(size=(1, self.repetitions - additional_values))
        else:
            self.appended_values = None
        
        self.multiplier = torch.zeros(size=(self.repetitions, 1))
        for k in range(self.repetitions):
            self.multiplier[k] = 2**(k*self.n_bits)
        
        return
    
    
    def __getitem__(self, index):
        
        x, y = self.data.__getitem__(index)
        
        x *= 255 / self.divider
        x = x.to(torch.int32).to(torch.float32)
        x = x.view(-1)
        if self.appended_values:
            x = torch.cat([x, self.appended_values[0]], dim=0)
        x = x.view(self.repetitions, -1)
        x *= self.multiplier
        x = torch.sum(x, dim=0).view(-1)
        
        return x/255, y
    
    
    def __len__(self):
        return self.data.__len__()
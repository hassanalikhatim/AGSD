import torch



class Poisonable_Data(torch.utils.data.Dataset):
    
    def __init__(self, data, **kwargs):
        
        self.data = data
        
        self.poison_indices = []
        self.poisoner_fn = self.no_poison
        
        return
    
    
    def __getitem__(self, index):
        
        x, y = self.data.__getitem__(index)
        
        if index in self.poison_indices:
            x, y = self.poisoner_fn(x, y, type_=0)
        
        return x, y
    
    
    def __len__(self):
        return self.data.__len__()
    
    
    def no_poison(self, x, y, **kwargs):
        return x, y
    
    
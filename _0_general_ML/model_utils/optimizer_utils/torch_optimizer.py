import torch



class Torch_Optimizer:
    
    def __init__(
        self, name :str='adam',
        lr :float=1e-4, momentum :float=0.5,
        weight_decay: float=0.
    ):
        
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.name = name
        
        self.optimizer_dict = {
            'adam': self.adam,
            'sgd': self.sgd
        }
        
        return
    
    
    def return_optimizer(
        self, parameters
    ):
        
        return self.optimizer_dict[self.name](parameters)
    
    
    def adam(self, parameters):
        
        my_optimizer = torch.optim.Adam(
            parameters, lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        return my_optimizer
    
    
    def sgd(self, parameters):
        
        my_optimizer = torch.optim.SGD(
            parameters, lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        return my_optimizer
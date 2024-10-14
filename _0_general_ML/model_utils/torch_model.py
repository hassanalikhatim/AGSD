import torch
import os


from _0_general_ML.model_utils.torch_model_plugin import Torch_Model_Plugin

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from utils_.general_utils import confirm_directory



class Torch_Model(Torch_Model_Plugin):
    
    def __init__(
        self,
        data: Torch_Dataset, model_configuration,
        path: str='',
        **kwargs
    ):
        
        super().__init__(
            data, model_configuration,
            path=path
        )
        
        return
    
    
    def train(
        self, start_epoch=1, epochs=1,
        batch_size=64, 
        verbose=True, 
        validate=True,
        **kwargs
    ):
        
        train_loader, test_loader = self.data.prepare_data_loaders(batch_size=self.model_configuration['batch_size'])
        
        test_loss, test_acc = self.test_shot(test_loader, verbose=verbose)
        for epoch in range(start_epoch, epochs+1):
            train_loss, train_acc, train_str = self.train_shot(train_loader, epoch, verbose=verbose)
            if validate:
                test_loss, test_acc = self.test_shot(test_loader, verbose=verbose, pre_str=train_str)
        
        return
        
        
    def save_weights(self, name, save_optimizer=True):
        
        torch.save(self.model.state_dict(), name+'.pth')
        if save_optimizer:
            try: torch.save(self.optimizer.state_dict(), name+'_optimizer.pth')
            except: print('Unable to save optimizer.')
        
        return
    
    
    def unsave_weights(self, name, **kwargs):
        
        if os.path.isfile(name+'.pth'):
            os.remove(f'{name}.pth')
        if os.path.isfile(name+'_optimizer.pth'):
            os.remove(name+'_optimizer.pth')
        
        return
    
    
    def load_weights(
        self, name, load_optimizer=True
    ):
        
        if os.path.isfile(name+'.pth'):
            
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(name+'.pth'))
            else:
                self.model.load_state_dict(torch.load(name+'.pth', map_location=torch.device(self.device)))
            if load_optimizer and os.path.isfile(name+'_optimizer.pth'):
                self.optimizer.load_state_dict(torch.load(name+'_optimizer.pth'))
            print('Loaded pretrained weights.')
            
            return True
        
        return False
        
    
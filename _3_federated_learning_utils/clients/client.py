import torch
import copy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model



class Client:
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={},
        client_type='clean'
    ):
        
        self.global_model_configuration = global_model_configuration
        self.client_type = client_type
        
        self.reset_client(data=data, client_configuration=client_configuration)
        
        return
    
    
    def reset_client(self, data=None, client_configuration: dict={}):
        
        if data:
            self.data = data
            
        self.local_model_configuration = {
            'local_epochs': 1
        }
        for key in self.global_model_configuration.keys():
            self.local_model_configuration[key] = self.global_model_configuration[key]
        if client_configuration:
            for key in client_configuration.keys():
                self.local_model_configuration[key] = client_configuration[key]
        
        return
    
    
    def weight_updates(self, global_model_state_dict, verbose=True, **kwargs) -> dict:
        
        local_model = Torch_Model(
            data = self.data,
            model_configuration=self.local_model_configuration
        )
        local_model.model.load_state_dict(copy.deepcopy(global_model_state_dict))
        
        local_model.train(
            epochs=self.local_model_configuration['local_epochs'], 
            batch_size=self.local_model_configuration['batch_size'],
            verbose=verbose
        )
        weights = local_model.model.state_dict()
        
        # allow weight scaling in the updates
        if 'scale' in self.local_model_configuration.keys():
            if self.local_model_configuration['scale']:
                for key in weights.keys():
                    gradients = weights[key] - global_model_state_dict[key]
                    weights[key] = global_model_state_dict[key] + self.local_model_configuration['scale']*gradients
        
        return weights
    
    
    def test_server_model(self, poisoned_data: Torch_Dataset, model: Torch_Model):
        '''
        This function is just for analysis and results.
        '''
        
        test_data_loader = torch.utils.data.DataLoader(poisoned_data.test, batch_size=model.model_configuration['batch_size'])
        
        c_loss, c_acc = model.test_shot(test_data_loader, verbose=False)
        
        return {'clean_loss': c_loss, 'clean_acc': c_acc}
    
    
    
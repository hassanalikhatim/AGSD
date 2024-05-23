import copy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from .simple_backdoor_client import Simple_Backdoor_Client



class Visible_Backdoor_Initially_Clean(Simple_Backdoor_Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        self.clean_data = copy.deepcopy(data)
        
        super().__init__(
            data, global_model_configuration,
            client_configuration=client_configuration
        )
        
        self.client_type = 'visible_backdoor_initially_good'
        
        return
    
    
    def weight_updates(self, global_model_state_dict, be_good: bool=False, verbose=True, **kwargs) -> dict:
        
        if be_good:
            local_model = Torch_Model(data=self.clean_data, model_configuration=self.local_model_configuration)
        else:
            local_model = Torch_Model(data = self.data, model_configuration=self.local_model_configuration)
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
    
    
    
import copy
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server



class Server_DP(Server):
    
    def __init__(
        self,
        data: Torch_Dataset,
        model: Torch_Model,
        clients_with_keys: dict={},
        configuration=None,
        **kwargs
    ):
        
        super().__init__(
            data, model, clients_with_keys=clients_with_keys, configuration=configuration
        )
        
        
        default_configuration = {
            'differential': 1e-5,
            'clip_value': 1
        }
        for key in default_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_configuration[key]
        
        return
    
    
    def aggregate(
        self, clients_state_dict,
        **kwargs
    ):
        
        assert len(clients_state_dict) > 0
        
        state_t_minus_1 = self.model.model.state_dict()
        
        state_t = copy.deepcopy(state_t_minus_1)
        updates = copy.deepcopy(state_t_minus_1)
        for key in state_t.keys():
            for i in range(len(clients_state_dict)):
                weight_update = clients_state_dict[i][key] - state_t_minus_1[key]
                
                weight_update = self.clip_weights(
                    weight_update, clip_value=self.configuration['clip_value']
                ).float()
                
                if i == 0:
                    updates[key] = torch.div(weight_update, len(clients_state_dict))
                else:
                    updates[key] += torch.div(weight_update, len(clients_state_dict))
                
            state_t[key] = state_t[key].float() + updates[key]
            state_t[key] += torch.normal(
                0., self.configuration['differential']*torch.std(state_t[key], unbiased=False), size=state_t[key].shape
            ).to(state_t[key].device)
        
        return state_t
    
    
    def clip_weights(
        self, weight_update, clip_value=1
    ):
        
        # clipper = 1 / torch.sum( torch.square(weight_update) )
        # clipper = clip_value - torch.relu(clip_value - clipper)
        # weight_update *= clipper
        
        weight_update = torch.clip(weight_update, min=-clip_value, max=clip_value)
        
        return weight_update
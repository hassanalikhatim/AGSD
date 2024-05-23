import numpy as np
import copy
import torch
from sklearn.cluster import HDBSCAN


from _3_federated_learning_utils.servers.server import Server

from _0_general_ML.model_utils.torch_model import Torch_Model



class Server_Divider(Server):
    
    def __init__(
        self,
        model: Torch_Model, data=None,
        clients: list=[], clients_ratio=0.1,
        configuration=None
    ):
        
        super().__init__(
            model, data=data, clients=clients, clients_ratio=clients_ratio
        )
        
        return
    
    
    def aggregate(
        self, clients_state_dict,
        **kwargs
    ):
        
        assert len(self.clients_state_dict) > 0
        
        state_t_minus_1 = self.model.model.state_dict()
        
        state_t = copy.deepcopy(state_t_minus_1)
        updates = copy.deepcopy(state_t_minus_1)
        for key in state_t.keys():
            for i in range(len(clients_state_dict)):
                weight_update = clients_state_dict[i][key] - state_t_minus_1[key]
                
                weight_update = self.clip_weights(
                    weight_update, clip_value=self.configuration['clip_value']
                )
                
                if i == 0:
                    updates[key] = torch.div(weight_update, len(clients_state_dict))
                else:
                    updates[key] += torch.div(weight_update, len(clients_state_dict))
                
            state_t[key] += updates[key]
            state_t[key] += torch.normal(
                0., self.configuration['differential'], size=state_t[key].shape
            ).to(state_t[key].device)
        
        return state_t
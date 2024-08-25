import numpy as np
import torch
import copy

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from ._visible_hgsd_heldout import AGSD_ID



class AGSD_ID_Median(AGSD_ID):
    
    def __init__(
        self,
        data: Torch_Dataset,
        model: Torch_Model,
        clients_with_keys: dict={},
        configuration: dict={},
        verbose: bool=True,
        **kwargs
    ):
        
        super().__init__(
            data, model, clients_with_keys=clients_with_keys, configuration=configuration, verbose=verbose
        )
        
        return
    
    
    def rescaled_noisy_aggregate(self, clients_states: list[dict], scalers: np.ndarray=None, strength=1e-5):
        
        if scalers is None:
            # scalers = np.ones((len(clients_states)))
            initial_state = self.model.model.state_dict()
            initial_model_state = torch.stack([self.parameter_flatten_client_state_torch(initial_state)])
            flattened_states = torch.stack([self.parameter_flatten_client_state_torch(cl_s) for cl_s in clients_states], dim=0)
            norms = torch.norm(flattened_states-initial_model_state, p=2, dim=1, keepdim=True).view(-1)
            scalers = torch.median(norms) / norms
        
        state_t = copy.deepcopy(self.model.model.state_dict())
        delta_state_t = copy.deepcopy(self.model.model.state_dict())
        for key in self.model.model.state_dict().keys():
            
            deltas = []
            for i, client_state in enumerate(clients_states):
                delta = (client_state[key] - self.model.model.state_dict()[key]).clone() * scalers[i]
                try: delta += strength * torch.normal(0., torch.std(delta.view(-1), unbiased=False), size=state_t[key].shape).to(state_t[key].device)
                except: pass
                deltas.append(delta)
                
            deltas = torch.stack([deltas[i] for i, client_state in enumerate(clients_states)], dim=0)
            delta_state_t[key] = torch.median(torch.stack([delta for delta in deltas], dim=0), dim=0)[0]
            # delta_state_t[key] = torch.mean(torch.stack([delta for delta in deltas], dim=0), dim=0)
            
            state_t[key] = state_t[key].float() + delta_state_t[key]
                
        return state_t
    
    
    
import numpy as np
import copy
import torch
from sklearn.metrics.pairwise import cosine_similarity


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server



epsilon_correction_for_alpha = 1e-3


class Server_FoolsGold(Server):
    """
        Mitigating Sybils in Federated Learning Poisoning
        URL = https://arxiv.org/abs/1808.04866
        
        @article{fung2018mitigating,
            title={Mitigating sybils in federated learning poisoning},
            author={Fung, Clement and Yoon, Chris JM and Beschastnikh, Ivan},
            journal={arXiv preprint arXiv:1808.04866},
            year={2018}
        }
        
        Credits: This implementation is inspired from: https://github.com/sail-research/iba/blob/main/defense.py
    """
    
    def __init__(
        self,
        data: Torch_Dataset,
        model: Torch_Model,
        clients_with_keys: dict={},
        configuration=None
    ):
        
        super().__init__(
            data, model, clients_with_keys=clients_with_keys, configuration=configuration
        )
        
        default_configuration = {
            'k': 1
        }
        for key in default_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_configuration[key]
        
        self.clients_aggregated_history = [{}] * len(self.clients)
        self.len_aggregated_history = [0] * len(self.clients)
        
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        
        return
    
    
    def aggregate(
        self, clients_state_dict,
        **kwargs
    ):
        
        assert len(clients_state_dict) > 0
        
        state_t_minus_1 = self.model.model.state_dict()
        
        self.update_clients_history(clients_state_dict)
        
        # flatten clients state for cosine similarity of currently active clients
        flattened_clients_history = [
            self.flatten_client_state(client_state) for client_state in [self.clients_aggregated_history[i] for i in self.active_clients]
        ]
        
        # add here the code for computing weights {alphas} by measuring the magnitude of 
        # model parameters in the output layer of the global model. For now the weights
        # are all 1.
        all_client_cs = cosine_similarity(flattened_clients_history)  - np.eye(len(self.active_clients))
        all_client_cs = self.pardoning_and_adjustment(all_client_cs)
        
        alphas = 1 - np.max(all_client_cs, axis=1)
        alphas /= (np.max(alphas) + np.sign(np.max(alphas))*epsilon_correction_for_alpha)
        alphas = self.configuration['k'] * ( np.log( (alphas/(1-alphas)) ) + 0.5 )
        alphas = len(self.active_clients)*np.exp(alphas)/np.sum(np.exp(alphas))
        assert len(alphas) == len(self.active_clients)
        
        state_t = {}
        for key in state_t_minus_1.keys():
            for i, client_state in enumerate(clients_state_dict):
                if key not in state_t.keys():
                    state_t[key] = copy.deepcopy(alphas[i] * client_state[key])
                else:
                    state_t[key] += alphas[i] * client_state[key]
            
            state_t[key] = torch.div(state_t[key], len(clients_state_dict))
        
        return state_t
    
    
    def flatten_client_state(
        self, client_state_dict
    ):
        
        # Flatten all the weights into a 1-D array
        flattened_client_state = []
        for key in client_state_dict.keys():
            if ('weight' in key) or ('bias' in key):
                flattened_client_state += client_state_dict[key].cpu().flatten().tolist()
        
        return np.array(flattened_client_state)
    
    
    def update_clients_history(
        self, clients_state_dict
    ):
        
        for i, ac in enumerate(self.active_clients):
            for key in clients_state_dict[0].keys():
                if key not in self.clients_aggregated_history[i].keys():
                    self.clients_aggregated_history[ac][key] = clients_state_dict[i][key]
                else:
                    self.clients_aggregated_history[ac][key] += clients_state_dict[i][key]
            self.len_aggregated_history[ac] += 1
        
        return
    
    
    def pardoning_and_adjustment(
        self, all_client_cs
    ):
        
        all_clients_v = [np.sort(cs)[-2] for cs in all_client_cs] 
        
        for i in range(len(self.active_clients)):
            for j in range(len(self.active_clients)):
                if all_clients_v[i] > all_clients_v[j]:
                    all_client_cs[i, j] *= all_clients_v[i]/all_clients_v[j]
            
        return all_client_cs
    
    
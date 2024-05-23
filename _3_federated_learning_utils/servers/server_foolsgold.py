import numpy as np
import copy
import torch
from sklearn.metrics.pairwise import cosine_similarity


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server



epsilon_correction_for_alpha = 1e-5


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
        configuration=None,
        **kwargs
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
        
        self.clients_aggregated_history = [0] * len(self.clients)
        self.len_aggregated_history = [0] * len(self.clients)
        
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        return
    
    
    def aggregate(
        self, clients_state_dict,
        **kwargs
    ):
        
        assert len(clients_state_dict) > 0
        
        state_t_minus_1 = self.model.model.state_dict()
        
        self.update_clients_history(clients_state_dict)
        
        # flatten clients state for cosine similarity of currently active clients
        flattened_clients_history = np.array([self.clients_aggregated_history[i] for i in self.active_clients])
        
        cosine_similarities = cosine_similarity(flattened_clients_history)
        all_client_cs = cosine_similarities - np.eye(len(self.active_clients))
        all_client_cs = self.pardoning_and_adjustment(all_client_cs)
        
        alphas = 1 - np.max(all_client_cs, axis=1)
        alphas[np.where(alphas > 1)] = 1
        alphas[np.where(alphas < 0)] = 0
        alphas /= (np.max(alphas) + epsilon_correction_for_alpha)
        alphas[np.where(alphas==1)] = 0.99
        alphas = np.log( (alphas/(1-alphas)) + epsilon_correction_for_alpha ) + 0.5
        alphas[np.where(np.isinf(alphas))] = 1
        alphas[np.where(alphas>1)] = 1
        alphas[np.where(alphas<0)] = 0
        assert len(alphas) == len(self.active_clients)
        
        state_t = copy.deepcopy(state_t_minus_1)
        for key in state_t_minus_1.keys():
            for i, client_state in enumerate(clients_state_dict):
                alpha_multiplier = alphas[i] * self.multipliers[i] / len(self.active_clients)
                state_t[key] = state_t[key].float() + ( alpha_multiplier * (client_state[key] - state_t[key]).clone() )
        
        return state_t
    
    
    def update_clients_history(self, clients_state_dict):
        
        self.multipliers = np.ones(( len(self.active_clients) ))
        for i, ac in enumerate(self.active_clients):
            difference = self.parameter_flatten_client_state_np(clients_state_dict[i])
            difference -= self.parameter_flatten_client_state_np(self.model.model.state_dict())
            if np.linalg.norm(difference) > 1:
                self.multipliers[i] = 1 / np.linalg.norm(difference)
            self.clients_aggregated_history[ac] += self.multipliers[i] * difference
        self.multipliers /= np.max(self.multipliers)
        
        return
    
    
    def pardoning_and_adjustment(self, all_client_cs: np.ndarray):
        
        max_cs = np.max(all_client_cs, axis=0) + epsilon_correction_for_alpha
        
        for i in range(len(self.active_clients)):
            for j in range(len(self.active_clients)):
                if max_cs[i] < max_cs[j]:
                    all_client_cs[i, j] *= max_cs[i]/max_cs[j]
            
        return all_client_cs
    
    
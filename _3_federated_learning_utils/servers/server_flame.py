import numpy as np
import copy
import torch
from sklearn.cluster import HDBSCAN


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server



class Server_Flame(Server):
    """
        FLAME: Taming Backdoors in Federated Learning
        URL = https://arxiv.org/abs/2101.02281
        
        @inproceedings{nguyen2022flame,
            title={$\{$FLAME$\}$: Taming backdoors in federated learning},
            author={Nguyen, Thien Duc and Rieger, Phillip and De Viti, Roberta and Chen, Huili and Brandenburg, Bj{\"o}rn B and Yalame, Hossein and M{\"o}llering, Helen and Fereidooni, Hossein and Marchal, Samuel and Miettinen, Markus and others},
            booktitle={31st USENIX Security Symposium (USENIX Security 22)},
            pages={1415--1432},
            year={2022}
        }
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
            'lambda': 0.000012,
            'differential': 1e-3
        }
        for key in default_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_configuration[key]
        
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        self.hdb = HDBSCAN(
            min_cluster_size=1+int(self.configuration['clients_ratio']*len(self.clients)/2),
            min_samples=1, allow_single_cluster=True
        )
        
        return
    
    
    def aggregate(
        self, clients_state_dict,
        pre_str=''
    ):
        
        assert len(clients_state_dict) > 0
        
        state_t_minus_1 = self.model.model.state_dict()
        
        state_t_minus_1_flattened = torch.stack([self.parameter_flatten_client_state_torch(self.model.model.state_dict())], dim=0)
        flattened_clients_state_dict = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_state_dict], dim=0)
        
        # cosine_similarities = self.cosine_similarity(flattened_clients_state_dict, flattened_clients_state_dict)
        cosine_similarities = [
            self.cosine_similarity(fcs.view(1, -1), flattened_clients_state_dict).detach().cpu().numpy() 
            for fcs in flattened_clients_state_dict
        ]
        
        euclidean_distances = torch.norm(flattened_clients_state_dict-state_t_minus_1_flattened, p=2, dim=1)
        median_euclidean_distance = torch.median(euclidean_distances)
        clippers = torch.clip(median_euclidean_distance / euclidean_distances, min=0, max=1)
        
        self.hdb.fit(cosine_similarities)
        selected_indices = np.where(self.hdb.labels_ != -1)[0]
        
        if len(selected_indices) == 0:
            print('***********THIS IS BAD :(***********')
            state_t = state_t_minus_1
        
        else:
            state_t = copy.deepcopy(state_t_minus_1)
            for key in state_t.keys():
                for i in selected_indices:
                    state_t[key] = state_t[key].float() + (clients_state_dict[i][key] - state_t_minus_1[key]).clone() * clippers[i] / len(selected_indices)
                
                if not ('bias' in key or 'bn' in key):
                    standard_deviation = median_euclidean_distance * self.configuration['lambda'] * torch.std(state_t[key].clone().view(-1), unbiased=False)
                    state_t[key] += torch.normal(0., standard_deviation, size=state_t[key].shape).to(state_t[key].device)
        
        self.good_indicator = np.ones((len(self.active_clients)))
        self.good_indicator[np.where(self.hdb.labels_ == -1)] = -1.
        
        return state_t
    
    
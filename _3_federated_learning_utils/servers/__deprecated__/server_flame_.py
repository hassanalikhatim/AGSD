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
        configuration=None
    ):
        
        super().__init__(
            data, model, clients_with_keys=clients_with_keys, configuration=configuration
        )
        
        default_configuration = {
            'lambda': 1e-4,
            'differential': 1e-3
        }
        for key in default_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_configuration[key]
        
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
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
        state_t_minus_1_flattened = self.flatten_states([state_t_minus_1])
        
        flattened_clients_state_dict = self.flatten_states(clients_state_dict)
        cosine_similarities = self.flattened_cosine_similarities(flattened_clients_state_dict)
        
        euclidean_distances = torch.sum(
            torch.square( flattened_clients_state_dict-state_t_minus_1_flattened ), dim=0
        )
        median_euclidean_distance = torch.median(euclidean_distances)
        
        # print('\r' + pre_str + 'Clustering the cosine similarities. Please wait.', end='')
        self.hdb.fit(cosine_similarities)
        # print('\r' + pre_str, end='')
        selected_indices = np.where(self.hdb.labels_ != -1)[0]
        
        # selected_indices = [i for i in range(len(clients_state_dict))]
        
        if len(selected_indices) == 0:
            print('***********THIS IS BAD :(***********')
            state_t = state_t_minus_1
        
        else:
            state_t = copy.deepcopy(state_t_minus_1)
            for key in state_t.keys():
                for i in selected_indices:
                    clipper = torch.clip(median_euclidean_distance / euclidean_distances[i], max=1)
                    state_t[key] = state_t[key].float() + (clients_state_dict[i][key] - state_t_minus_1[key]) * clipper / len(selected_indices)
                    
                standard_deviation = median_euclidean_distance * self.configuration['lambda']
                # standard_deviation = (median_euclidean_distance / self.epsilon) * torch.sqrt( 2 * torch.log(1.25/self.configuration['differential']) )
                state_t[key] += torch.normal(0., standard_deviation, size=state_t[key].shape).to(state_t[key].device)
        
        return state_t
    
    
    def flatten_states(
        self, clients_state_dict
    ):
        
        # Flatten all the weights into a 1-D array
        flattened_clients_state_dict = []
        for cl, client_state in enumerate(clients_state_dict):
            
            flattened_client_state = []
            for k, key in enumerate(client_state.keys()):
                 if ('weight' in key) or ('bias' in key):
                    flattened_client_state.append(client_state[key].view(-1))
            
            flattened_clients_state_dict.append( torch.cat(flattened_client_state) )
        
        flattened_clients_state_dict = torch.stack(flattened_clients_state_dict, dim=0)
        
        return flattened_clients_state_dict
    
    
    def flattened_cosine_similarities(
        self, flattened_clients_state_dict
    ):
        
        # Compute cosine similarities of the flattened weights
        cosine_similarities = []
        for cl_1, client_state_1 in enumerate(flattened_clients_state_dict):
            
            _cosine_similarities_ = []
            for cl_2, client_state_2 in enumerate(flattened_clients_state_dict):
                _cosine_similarities_.append(self.cosine_similarity(client_state_1, client_state_2))
                
            cosine_similarities.append( torch.Tensor(_cosine_similarities_) )
            
        cosine_similarities = torch.stack( cosine_similarities, dim=0 )
        
        return cosine_similarities
    
    
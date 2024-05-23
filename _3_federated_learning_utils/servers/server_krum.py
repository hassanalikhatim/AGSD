import numpy as np
import copy
import torch
from sklearn.cluster import HDBSCAN


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server



class Server_Krum(Server):
    """
        Machine learning with adversaries: Byzantine tolerant gradient descent
        URL = http://papers.neurips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
        
        @article{blanchard2017machine,
            title={Machine learning with adversaries: Byzantine tolerant gradient descent},
            author={Blanchard, Peva and El Mhamdi, El Mahdi and Guerraoui, Rachid and Stainer, Julien},
            journal={Advances in neural information processing systems},
            volume={30},
            year={2017}
        }
    """
    
    def __init__(
        self,
        data: Torch_Dataset,
        model: Torch_Model,
        clients_with_keys: dict={},
        configuration: dict=None,
        **kwargs
    ):
        
        super().__init__(
            data, model, clients_with_keys=clients_with_keys, configuration=configuration
        )
        
        krum_message = '{Total clients: K} must be > 2 x {backdoor_clients: b} + 3'
        if not (2*len(clients_with_keys['clean'])) > (len(self.clients)+3):
            print(krum_message)
            print('This condition of krum has not been met. Doing the experiment anyways.')
        
        clients_in_one_round = int( self.configuration['clients_ratio'] * len(self.clients) )
        # assuming {K} = 2{b}+3 => {K}-{b}-2 = {b}+1 = ({K}-3)/2 + 1
        self.closest_local_updates = int ( (clients_in_one_round-3)/2 )
        # assuming {K} = 2{b}+3 => {selected updates: l} = {K}-{b} = {K} - ({K}-3)/2 = ({K}+3)/2
        self.selected_updates = int( (clients_in_one_round+3)/2 )
        
        return
    
    
    def aggregate(
        self, clients_state_dict,
        **kwargs
    ):
        
        assert len(clients_state_dict) > 0
        
        state_t_minus_1 = self.model.model.state_dict()
        
        flattened_clients_state_dict = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_state_dict], dim=0)
        
        all_client_scores = [
            self.compute_client_scores(flattened_client_state_dict, flattened_clients_state_dict)
            for flattened_client_state_dict in flattened_clients_state_dict
        ]
        score_threshold = np.sort(all_client_scores)[self.selected_updates]
        selected_client_indices = np.where(all_client_scores < score_threshold)[0]
        # selected_clients_state_dict = clients_state_dict[np.where(all_client_scores < score_threshold)]
        
        w_avg = {}
        for key in state_t_minus_1.keys():
            for i in selected_client_indices:
                if key not in w_avg.keys():
                    w_avg[key] = copy.deepcopy(clients_state_dict[i][key])
                else:
                    w_avg[key] += clients_state_dict[i][key]
            
            w_avg[key] = torch.div(w_avg[key], len(selected_client_indices))
            
        self.good_indicator = -1. * np.ones((len(self.active_clients)))
        self.good_indicator[selected_client_indices] = 1.
        
        return w_avg
    
    
    def flatten_client_state(
        self, client_state_dict
    ):
        
        # Flatten all the weights into a 1-D array
        flattened_client_state = []
        for key in client_state_dict.keys():
            flattened_client_state += client_state_dict[key].cpu().flatten().tolist()
        
        return np.array(flattened_client_state)
    
    
    def compute_client_scores(
        self, 
        flattened_client_state_dict,
        flattened_clients_state_dict
    ):
        
        current_client_scores = torch.mean(torch.abs(flattened_client_state_dict.view(1, -1) - flattened_clients_state_dict), dim=1)
        current_client_scores = current_client_scores.detach().cpu().numpy()
        
        local_threshold_value = np.sort(current_client_scores)[self.closest_local_updates]
        current_client_score = np.sum(
            current_client_scores[np.where(current_client_scores < local_threshold_value)]
        )
        
        return current_client_score
    
    
    
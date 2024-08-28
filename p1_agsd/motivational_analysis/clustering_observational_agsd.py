import numpy as np
import copy
import torch
import os


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from ..agsd_servers._visible_agsd_server import AGSD_ID



class Motivational_AGSD_ID(AGSD_ID):
    
    def __init__(
        self, 
        data: Torch_Dataset,
        model: Torch_Model,
        clients_with_keys: dict={}, 
        configuration: dict={}
    ):
        
        super().__init__(data, model, clients_with_keys=clients_with_keys, configuration=configuration)
        
        self.epoch = 0
        sampling_path = 'p1_hasnets/motivational_analysis/clients_samplings.npy'
        if os.path.isfile(sampling_path):
            self.clients_samplings = np.load(sampling_path)
            print('Loaded sampling file.')
        else:
            self.clients_samplings = []
            for i in range(50):
                self.clients_samplings.append(np.random.choice(
                    len(self.clients), int(self.clients_ratio*len(self.clients)), replace=False
                ))
            self.clients_samplings = np.array(self.clients_samplings)
            np.save(sampling_path, self.clients_samplings)
        
        return
    
    
    def get_metric(self, values: np.ndarray):
        
        this_clusterer = self.kmeans
        
        # cluster using spectral clustering
        try:
            this_clusterer.fit(values)
            self.labels_ = this_clusterer.labels_
            sorted_labels = np.sort(np.unique(self.labels_))
            
            current_clients = np.array(self.clients_keys)[self.active_clients]
            metric = 0
            for label in sorted_labels:
                if np.sum(current_clients=='clean')>0 and np.sum(current_clients=='clean') < 1:
                    metric += np.mean(current_clients[np.where(self.labels_==label)]=='clean')*np.mean(current_clients[np.where(self.labels_!=label)]=='clean')
                else:
                    metric += 0
        except:
            metric = -1
            
        return metric
    
    
    def aggregate(self, clients_state_dict, pre_str=''):
        
        # Functions to normalize torch array values
        def normalized_torch(arr_in: torch.Tensor):
            return torch.exp(arr_in)/torch.sum(torch.exp(arr_in))
        def linearly_normalized_torch(arr_in: torch.Tensor):
            return (arr_in-torch.min(arr_in))/(torch.max(arr_in)-torch.min(arr_in)) if torch.max(arr_in)>torch.min(arr_in) else arr_in/torch.max(arr_in)
        
        
        current_clients = np.array(self.clients_keys)[self.active_clients]
        
        # observational code here
        initial_model_state = torch.stack([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        
        flattened_clients_states = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_state_dict], dim=0)
        self.compute_scalers(flattened_clients_states-initial_model_state)
        flattened_clients_states = initial_model_state.clone() + (flattened_clients_states.clone()-initial_model_state.clone()) * self.scalers.view(-1, 1)
        
        # Preliminary aggregation
        perturbed_aggregated_state = self.rescaled_noisy_aggregate(clients_state_dict, scalers=self.scalers, strength=0.)
        perturbed_aggregated_state, flattened_perturbed_aggregated_state = self.rescale_a_state(perturbed_aggregated_state)
        
        self_cs_values_0 = self.pairwise_cosine_similarity_torch( flattened_clients_states )
        self_cs_values_1 = self.pairwise_cosine_similarity_torch( flattened_clients_states-flattened_perturbed_aggregated_state )
        self_cs_values_2 = self.pairwise_cosine_similarity_torch( flattened_clients_states-initial_model_state )
        self_cs_values =  linearly_normalized_torch(self_cs_values_1 + self_cs_values_2)
        
        self.metrics = [-2] * 3
        metrics = []
        for value in [self_cs_values_0, self_cs_values_2, self_cs_values]:
            assert torch.sum(torch.isnan(value)) == 0, 'Input contains NaN, therefore, skipping this training round.'
            metrics.append(self.get_metric(value.detach().cpu().numpy()))
        self.metrics = metrics
        
        # print(', '.join([str(torch.std(value, unbiased=False).item()) for value in [self_cs_values_0, self_cs_values_1, self_cs_values_2, self_cs_values]]))
        
        return self.super_aggregate([clients_state_dict[i] for i in np.where(current_clients=='clean')[0]]) if 'clean' in current_clients else self.model.model.state_dict()
    
    
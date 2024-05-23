import numpy as np
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy import stats


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server



class Server_Mesas(Server):
    """
        MESAS: Poisoning Defense for Federated Learning Resilient against Adaptive Attackers
        URL = https://dl.acm.org/doi/abs/10.1145/3576915.3623212
        
        @inproceedings{krauss2023mesas,
            title={MESAS: Poisoning Defense for Federated Learning Resilient against Adaptive Attackers},
            author={Krau{\ss}, Torsten and Dmitrienko, Alexandra},
            booktitle={Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
            pages={1526--1540},
            year={2023}
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
            'significance_level': 1e-4
        }
        for key in default_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_configuration[key]
        
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.agg_cluster = AgglomerativeClustering()
        
        return
    
    
    def aggregate(
        self, clients_state_dict,
        pre_str=''
    ):
        
        mesas_values = self.get_mesas_values(clients_state_dict)
        filtered_selected_indices = self.statistical_test(mesas_values)
        
        if len(filtered_selected_indices) >= 2:
            mesas_values = np.array([mesas_values[:, c] for c in filtered_selected_indices])
            assert len(mesas_values) == len(filtered_selected_indices)
            self.selected_states = self.cluster(mesas_values)
        elif len(filtered_selected_indices) == 1:
            self.selected_states = filtered_selected_indices
        else:
            self.selected_states = []
            return self.model.model.state_dict()
        
        self.good_indicator = -1. * np.ones((len(self.active_clients)))
        self.good_indicator[self.selected_states] = 1.
        
        return super().aggregate([clients_state_dict[c] for c in self.selected_states])
    
    
    def cluster(self, mesas_values):
        
        mesas_values = np.array(mesas_values).reshape(len(mesas_values), -1)
        self.agg_cluster.fit(mesas_values)
        dominant_label = -1
        for label in np.unique(self.agg_cluster.labels_):
            if np.mean( self.agg_cluster.labels_==label ) > dominant_label:
                dominant_label = label
        
        return np.where(self.agg_cluster.labels_==dominant_label)[0]
    
    
    def statistical_test(self, mesas_values_all: np.ndarray):
        
        list_of_filtered_indices = []
        for mesas_values in mesas_values_all:
            median_value = np.median(mesas_values)
            abs_distances = np.abs(mesas_values - median_value)
            list_1 = abs_distances[np.where(mesas_values >= median_value)]
            list_2 = abs_distances[np.where(mesas_values <= median_value)]
            
            # perform V-test
            p_test_1 = stats.ttest_ind(list_1, abs_distances).pvalue > self.configuration['significance_level']
            p_test_2 = stats.ttest_ind(list_2, abs_distances).pvalue > self.configuration['significance_level']
            
            v_test_1 = stats.levene(list_1, abs_distances).pvalue > self.configuration['significance_level']
            v_test_2 = stats.levene(list_2, abs_distances).pvalue > self.configuration['significance_level']
            
            d_test_1 = stats.kstest(list_1, abs_distances).pvalue > self.configuration['significance_level']
            d_test_2 = stats.kstest(list_2, abs_distances).pvalue > self.configuration['significance_level']
            
            selected_indices = []
            if p_test_1 and v_test_1 and d_test_1:
                selected_indices += np.where(mesas_values > median_value)[0].reshape(-1).tolist()
            if p_test_2 and v_test_2 and d_test_2:
                selected_indices += np.where(mesas_values < median_value)[0].reshape(-1).tolist()
            
            # remove outliers beyond 3 standard deviation
            filtered_selected_indices = []
            if len(selected_indices) > 0:
                selected_mesas_values = np.array([mesas_values[c] for c in selected_indices])
                mean_, std_ = np.mean(mesas_values), np.std(mesas_values)
                
                for i, index in enumerate(selected_indices):
                    if (selected_mesas_values[i] < mean_+3*std_) and (selected_mesas_values[i] > mean_-3*std_):
                        filtered_selected_indices.append(selected_indices[i])
                        
            list_of_filtered_indices.append(filtered_selected_indices)
        
        commonly_selected_indices = []
        for i in range(len(self.active_clients)):
            i_test = True
            for filtered_indices in list_of_filtered_indices:
                i_test = i_test and (i in filtered_indices)
            if i_test:
                commonly_selected_indices.append(i)
        
        return commonly_selected_indices
    
    
    def get_mesas_values(self, clients_state_dict):
        
        flattened_global = torch.stack([self.parameter_flatten_client_state_torch(self.model.model.state_dict())], dim=0)
        flattened_clients = torch.stack([
            self.parameter_flatten_client_state_torch(client_state_dict)
            for client_state_dict in clients_state_dict
        ], dim=0)
        
        flattened_diffs = (flattened_clients - flattened_global).detach().cpu()
        nz_diff = torch.where(flattened_diffs>0, flattened_diffs, torch.max(torch.abs(flattened_diffs)))
        
        cosine_distances = torch.tensor([
            1 - self.cosine_similarity(fcs.view(-1), flattened_global.view(-1)).detach().cpu().numpy() 
            for fcs in flattened_clients
        ])
        euclidean_distances = torch.norm( flattened_diffs, p=2, dim=1 ) / flattened_global.shape[-1]
        counts = torch.sum( torch.nn.functional.relu(torch.sign(flattened_diffs)), dim=1 )
        vars = torch.var( flattened_diffs, dim=1 )
        mins = torch.min(torch.abs(nz_diff), dim=1).values
        maxs = torch.max(torch.abs(flattened_diffs), dim=1).values
        
        # cosine_distances /= torch.max(torch.abs(cosine_distances))
        # euclidean_distances /= torch.max(torch.abs(euclidean_distances))
        # counts /= torch.max(torch.abs(counts))
        # vars /= torch.max(torch.abs(vars))
        # mins /= torch.max(torch.abs(mins))
        # maxs /= torch.max(torch.abs(maxs))
        
        mesas_values = torch.stack([cosine_distances, euclidean_distances, counts, vars, mins, maxs], dim=0)
        
        return mesas_values.detach().cpu().numpy()
    
    
    def _evaluate_server_statistics(self):
        
        dict_1 = super().evaluate_server_statistics()
        
        signs = {key: [0] for key in self.client_with_keys.keys()}
        for i, ac in enumerate(self.active_clients):
            if i in self.selected_states:
                signs[self.clients_keys[ac]].append(1)
            
        for key in signs.keys():
            if len(signs[key]) > 1:
                signs[key] = signs[key][1:]
        
        return {
            **dict_1, 
            **{key+'_r': np.mean(signs[key]) for key in signs.keys()}
        }
        
        
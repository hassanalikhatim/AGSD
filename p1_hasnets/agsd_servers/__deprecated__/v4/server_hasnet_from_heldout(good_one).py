import numpy as np
import torch
import copy

from sklearn.cluster import KMeans, HDBSCAN, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server



class Server_HaSNet_from_HeldOut(Server):
    
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
        
        default_hasnet_configuration = {
            'healing_epochs': 3,
            'healing_set_size': 1000,
            'epsilon': 0.1
        }
        for key in default_hasnet_configuration.keys():
            if key not in self.configuration.keys():
                self.configuration[key] = default_hasnet_configuration[key]
        
        # preparing healing data
        healing_set_size = min(int(self.configuration['healing_set_size']/0.9), self.data.train.__len__())
        healing_data_indices = np.random.choice(self.data.train.__len__(), healing_set_size, replace=False)
        self.healing_data = Client_Torch_SubDataset(self.data, idxs=healing_data_indices, train_size=0.9)
        
        # preparing some useful functions
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        self.norm = torch.norm(p=2, dim=1)
        self.hdb = HDBSCAN(
            # min_cluster_size=int(len(self.clients)*self.configuration['clients_ratio']/2)+1,
            min_cluster_size=2,
            min_samples=1,
            allow_single_cluster=True
        )
        self.spectral = SpectralClustering(n_clusters=3, affinity='precomputed')
        self.kmeans = KMeans(n_clusters=3, n_init='auto')
        self.clusterer = self.spectral
        
        # preparing hasnets variables
        self.diff_avg = 0
        self.rejected_mean, self.selected_mean = -1., 1.
        self.prob_selected, self.prob_rejected = 1., 0.
        self.clients_benign_probabilities = np.array([0.] * len(self.clients))
        self.current_epoch = 0
        
        return
    
    
    def heal_model_from_state(self, model_state, epochs=1):
        
        local_model_configuration = {}
        for key in self.model.model_configuration.keys():
            local_model_configuration[key] = self.model.model_configuration[key]
        local_model_configuration['learning_rate'] = 1e-4
        
        local_model = Torch_Model(self.healing_data, model_configuration=local_model_configuration)
        local_model.model.load_state_dict(model_state)
        local_model.train(epochs=epochs, batch_size=local_model_configuration['batch_size'], verbose=False)
        
        return local_model.model.state_dict()
    
    
    def pairwise_cosine_similarity_torch(self, flattened_clients_states):
        
        normalized_input_a = torch.nn.functional.normalize(flattened_clients_states)
        res = torch.mm(normalized_input_a, normalized_input_a.T)
        res[res==0] = 1e-6
        
        return res
    
    
    def compute_scalers(self, differences):
        
        norms = torch.norm(differences, p=2, dim=1, keepdim=True)
        self.scalers = torch.median(norms) / norms
        
        return
    
    
    def _old_process_and_aggregate(self, clients_states: list[dict], strength=1e-5):
        
        norms = torch.norm(torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_states], dim=0), p=2, dim=1)
        assert len(norms) == len(clients_states), f'len of norms is {len(norms)}, but len of states is {len(clients_states)}'
        scalers = torch.median(norms) / norms
        
        state_t = copy.deepcopy(self.model.model.state_dict())
        for key in self.model.model.state_dict().keys():
            state_t[key] = state_t[key].float()
            
            for i in range(len(clients_states)):
                state_t[key] = state_t[key] + (clients_states[i][key] - self.model.model.state_dict()[key]).clone() * scalers[i] / len(clients_states)
                    
            if not ('bias' in key or 'bn' in key):
                standard_deviation = strength * torch.std(state_t[key].clone().view(-1), unbiased=False)
                try: state_t[key] = state_t[key] + torch.normal(0., standard_deviation, size=state_t[key].shape).to(state_t[key].device)
                except: pass
            
        return state_t
    
    
    def fit_clusters(self, clients_state_dict: list[dict]):
        
        flattened_clients_states = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_state_dict], dim=0)
        self.compute_scalers(flattened_clients_states)
        cosine_similarities = self.pairwise_cosine_similarity_torch(flattened_clients_states).detach().cpu().numpy()
        self.clusterer.fit(cosine_similarities)
        self.gamma_labels_ = self.clusterer.labels_
        self.sorted_labels = np.sort(np.unique(self.gamma_labels_))
        
        return
    
    
    def compute_gammas(self, clients_state_dict: list[dict]):
        
        f_previous = torch.cat([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        f_previous_healed = torch.stack([self.parameter_flatten_client_state_torch(self.heal_model_from_state(self.model.model.state_dict(), epochs=self.configuration['healing_epochs']))])
        
        aggregated_model = self.process_and_aggregate(clients_state_dict)
        healed_model = self.heal_model_from_state(aggregated_model, epochs=self.configuration['healing_epochs'])
        f_ag = torch.stack([self.parameter_flatten_client_state_torch(aggregated_model)])
        f_h = torch.stack([self.parameter_flatten_client_state_torch(healed_model)])
        
        aggregated_models = [self.process_and_aggregate([clients_state_dict[c] for c in np.where(self.gamma_labels_==k)[0]]) for k in self.sorted_labels]
        f_sub_ags = [torch.stack([self.parameter_flatten_client_state_torch(aggregated_models[k])]) for k in self.sorted_labels]
        self.gamma = np.array([self.cosine_similarity(f_h-f_ag, f_sub_ags[i]-f_ag).item() for i, k in enumerate(self.sorted_labels)])
        self.gamma += np.array([self.cosine_similarity(f_previous_healed-f_previous, f_sub_ags[i]-f_previous).item() for i, k in enumerate(self.sorted_labels)])
        self.gamma = self.gamma.reshape(-1)
        
        return
    
    
    def compute_best_label(self):
        
        if np.max(self.gamma) > np.min(self.gamma):
            self.best_gamma_label = self.sorted_labels[np.where(self.gamma==np.max(self.gamma))].reshape(-1)[0]
            self.worst_gamma_label = self.sorted_labels[np.where(self.gamma==np.min(self.gamma))].reshape(-1)[0]
            self.diff = abs(np.max(self.gamma)-np.mean(self.gamma[np.where(self.gamma!=np.max(self.gamma))]))
        else:
            self.best_gamma_label = -2
            self.worst_gamma_label = -2
            self.diff = 0
        
        return
    
    
    def update_selection_probability(self):
        
        def process_prob(x):
            return np.clip(x*x*x, 0.05, 0.3)
        
        max_mean = np.mean(self.gamma[np.where(self.sorted_labels==self.best_gamma_label)]) if self.best_gamma_label != self.worst_gamma_label else -10
        nonmax_mean = np.mean(self.gamma[np.where(self.sorted_labels!=-self.best_gamma_label)]) if self.best_gamma_label != self.worst_gamma_label else 10
        
        epsilon = np.abs(self.selected_mean - self.rejected_mean) * 1e-2
        
        selected_selected = 1 / np.clip(self.selected_mean - max_mean, a_min=epsilon, a_max=None)
        selected_rejected = 1 / np.clip(max_mean - self.rejected_mean, a_min=epsilon, a_max=None)
        self.prob_selected = selected_selected / (selected_selected + selected_rejected)
        
        rejected_selected = 1 / np.clip(self.selected_mean - nonmax_mean, a_min=epsilon, a_max=None)
        rejected_rejected = 1 / np.clip(nonmax_mean - self.rejected_mean, a_min=epsilon, a_max=None)
        self.prob_rejected = rejected_selected / (rejected_selected + rejected_rejected)
        
        message = 'Both sel_prob {} and rej_prob {} should be <= 1.'.format(self.prob_selected, self.prob_rejected)
        assert self.prob_selected <= 1 and self.prob_rejected <= 1, message
        
        r_alpha = process_prob(self.prob_selected-self.prob_rejected)
        r_beta = self.prob_selected-self.prob_rejected
        self.selected_mean = (1-r_alpha)*self.selected_mean + r_alpha*max_mean
        self.rejected_mean = (1-r_alpha)*self.rejected_mean + r_alpha*nonmax_mean
        
        return r_alpha
    
    
    def aggregate(self, clients_state_dict: list[dict], **kwargs):
        
        # self.fit_clusters(clients_state_dict)
        initial_model = torch.stack([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        flattened_clients_states = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_state_dict], dim=0)
        self.compute_scalers(flattened_clients_states-initial_model)
        flattened_clients_states = initial_model + (flattened_clients_states-initial_model)*self.scalers
        aggregated_classifier = torch.mean(flattened_clients_states, dim=0, keepdim=True)
        cosine_similarities = self.pairwise_cosine_similarity_torch(flattened_clients_states-aggregated_classifier).detach().cpu().numpy()
        self.clusterer.fit(cosine_similarities)
        self.gamma_labels_ = self.clusterer.labels_
        self.sorted_labels = np.sort(np.unique(self.gamma_labels_))
        
        self.compute_gammas(clients_state_dict)
        
        self.compute_best_label()
        # r_alpha = self.update_selection_probability()
        
        self.clients_benign_probabilities[self.active_clients[np.where(self.gamma_labels_==self.best_gamma_label)]] += self.diff
        self.clients_benign_probabilities[self.active_clients[np.where(self.gamma_labels_!=self.best_gamma_label)]] -= self.diff
        self.clients_benign_probabilities = np.clip(self.clients_benign_probabilities, -10., 10.)
        
        self.hasnet_indicator = np.array([-1.]*len(self.active_clients))
        self.hasnet_indicator[np.where(self.gamma_labels_==self.best_gamma_label)] = 1.
        # self.clients_benign_probabilities[self.active_clients] += r_alpha * self.hasnet_indicator
        self.hasnet_indicator[np.where(np.array([self.clients_benign_probabilities[c] for c in self.active_clients]) >= 0.)] = 1.
        
        selected_client_states = [clients_state_dict[c] for c in np.where(self.hasnet_indicator==1)[0]]
        self.print_out('Clients selected.')
        
        return self.process_and_aggregate(selected_client_states if len(selected_client_states)>0 else [self.model.model.state_dict()])
    
    
    def evaluate_server_statistics(self):
        
        dict_1 = super().evaluate_server_statistics()
        
        signs = {key: [0] for key in self.client_with_keys.keys()}
        for i, ac in enumerate(self.active_clients):
            signs[self.clients_keys[ac]].append(np.mean(self.hasnet_indicator[i] > 0))
            
        for key in signs.keys():
            if len(signs[key]) > 1:
                signs[key] = signs[key][1:]
        
        return {
            **dict_1, 
            **{key+'_acc_ratio': np.mean(signs[key]) for key in signs.keys()}
        }
        
        
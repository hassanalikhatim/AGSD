import numpy as np
import torch
import copy
import gc

from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
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
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.hdb = HDBSCAN(min_cluster_size=int(len(self.clients)*self.configuration['clients_ratio']/2)+1, min_samples=1, allow_single_cluster=True)
        self.spectral = SpectralClustering(n_clusters=3, affinity='precomputed')
        self.kmeans = KMeans(n_clusters=3, n_init='auto')
        self.clusterer = self.spectral
        
        # preparing hasnets variables
        self.n_dims = 30
        self.rejected_mean, self.selected_mean = -1., 1.
        self.prob_selected, self.prob_rejected = 1., 0.
        self.clients_benign_probabilities = np.array([0.] * len(self.clients))
        
        self.server_name = f'hgsd_(id-{{{healing_set_size}}})'
        
        return
    
    
    def heal_model_from_state(self, model_state, epochs=1):
        
        local_model_configuration = {}
        for key in self.model.model_configuration.keys():
            local_model_configuration[key] = self.model.model_configuration[key]
        local_model_configuration['learning_rate'] = 1e-6
        
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
    
    
    def process_and_aggregate(self, clients_states: list[dict], strength=1e-5, indices: list[int] = None):
        
        if indices is None:
            indices = np.arange(len(clients_states))
        
        # initial_model_state = torch.cat([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        # flattened_clients_states = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_states], dim=0)
        # scalers = self.get_scalers(flattened_clients_states-initial_model_state)
        
        state_t = copy.deepcopy(self.model.model.state_dict())
        delta_state_t = copy.deepcopy(self.model.model.state_dict())
        for key in self.model.model.state_dict().keys():
            # state_t[key] = state_t[key].float()
            
            for i, ind in enumerate(indices):
                if i==0:
                    delta_state_t[key] = (clients_states[ind][key] - self.model.model.state_dict()[key]).clone() * self.scalers[ind] / len(clients_states)
                else:
                    delta_state_t[key] += (clients_states[ind][key] - self.model.model.state_dict()[key]).clone() * self.scalers[ind] / len(clients_states)
            
            state_t[key] = state_t[key].float() + delta_state_t[key]
            if not ('bias' in key or 'bn' in key):
                standard_deviation = torch.std(delta_state_t[key].clone().view(-1), unbiased=False)
                try: state_t[key] += strength * torch.normal(0., standard_deviation, size=state_t[key].shape).to(state_t[key].device)
                except: pass

        return state_t
    
    
    def rescale_and_flatten(self, clients_states: list[dict]):
        
        initial_model_state = torch.cat([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        flattened_clients_states = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_states], dim=0)
        
        self.compute_scalers(flattened_clients_states-initial_model_state)
        # norms = torch.norm(flattened_clients_states - initial_model_state, p=2, dim=1, keepdim=True)
        # assert len(norms) == len(flattened_clients_states), f'len of norms is {len(norms)}, but len of states is {len(flattened_clients_states)}'
        # scalers = torch.median(norms) / norms
        # scalers = self.get_scalers(flattened_clients_states-initial_model_state)
        
        return initial_model_state + (flattened_clients_states-initial_model_state) * self.scalers
    
    
    def compute_gammas(self, clients_state_dict: list[dict]):
        
        initial_model_state = torch.cat([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        healed_initial_state = torch.stack([self.parameter_flatten_client_state_torch(self.heal_model_from_state(self.model.model.state_dict(), epochs=self.configuration['healing_epochs']))])
        
        flattened_clients_states = self.rescale_and_flatten(clients_state_dict)
        client_to_initial = flattened_clients_states - initial_model_state
        
        perturbed_aggregated_state = self.process_and_aggregate(clients_state_dict, strength=0.)
        flattened_perturbed_aggregated_state = torch.stack([self.parameter_flatten_client_state_torch(perturbed_aggregated_state)], dim=0)
        flattened_aggregated_state = torch.mean(flattened_clients_states, dim=0, keepdim=True)
        flattened_healed_states = torch.stack([self.parameter_flatten_client_state_torch(self.heal_model_from_state(perturbed_aggregated_state, epochs=self.configuration['healing_epochs']))], dim=0)
        
        self.cs_values = 1 + self.pairwise_cosine_similarity_torch(flattened_clients_states-flattened_aggregated_state).detach().cpu().numpy()
        # self.cs_values += 1 + self.pairwise_cosine_similarity_torch(client_to_initial).detach().cpu().numpy()
        
        # Trust index for estimating the direction of healing
        self.gamma = self.cosine_similarity( flattened_healed_states-flattened_aggregated_state, flattened_clients_states-flattened_aggregated_state )
        self.gamma += self.cosine_similarity( healed_initial_state-initial_model_state, client_to_initial )
        self.gamma += self.cosine_similarity( flattened_perturbed_aggregated_state-flattened_healed_states, flattened_perturbed_aggregated_state-flattened_clients_states )
        self.gamma = self.gamma.view(len(self.active_clients), 1).detach().cpu().numpy()
        
        self.flattened_clients_states = flattened_clients_states
        self.flattened_aggregated_state = flattened_aggregated_state
        self.flattened_healed_states = flattened_healed_states
        
        return
    
    
    def fit_clusters(self):
        
        self.clusterer.fit(self.cs_values)
        self.gamma_labels_ = self.clusterer.labels_
        
        return
    
    
    def compute_hasnet_indicator(self):
        
        u_labels = np.unique(self.gamma_labels_)
        means = []
        for k in u_labels:
            means.append(np.mean([self.gamma[c, 0] for c in np.where(self.gamma_labels_==k)]))
        means = np.array(means)
        self.best_gamma_label = u_labels[np.where(means==np.max(means))]
        self.worst_gamma_label = u_labels[np.where(means==np.min(means))]
        
        return
    
    
    def update_selection_probability(self):
        
        def process_prob(x):
            return np.clip(x*x*x, 0.05, 0.3)
        
        max_mean = np.mean([self.gamma[c, 0] for c in np.where(self.gamma_labels_==self.best_gamma_label)]) if self.best_gamma_label != self.worst_gamma_label else -10
        nonmax_mean = np.mean([self.gamma[c, 0] for c in np.where(self.gamma_labels_!=-self.best_gamma_label)]) if self.best_gamma_label != self.worst_gamma_label else 10
        self.gamma_diff = '{:.3f}, {:.3f}, {:.3f}'.format(max_mean, nonmax_mean, max_mean - nonmax_mean)
        
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
    
    
    def aggregate_defended(self, clients_state_dict: list[dict], **kwargs):
        
        self.hasnet_indicator = np.array([0.]*len(self.active_clients))
        
        # select the clients that are believed to be non-poisoned
        self.compute_gammas(clients_state_dict); gc.collect()
        self.fit_clusters()
        self.compute_hasnet_indicator()
        r_alpha = self.update_selection_probability()
        
        self.hasnet_indicator[np.where(self.gamma_labels_==self.best_gamma_label)] = 1.
        self.hasnet_indicator[np.where(self.gamma_labels_==self.worst_gamma_label)] = -1.
        self.clients_benign_probabilities[self.active_clients] += r_alpha * self.hasnet_indicator
        self.hasnet_indicator[np.where(np.array([self.clients_benign_probabilities[c] for c in self.active_clients]) < 0.)] = -1.
        
        # self.print_out(
        #     '\r{} - Clean clients: {:.1f}%, \tselected_prob: {:.2f}, \trejected_prob: {:.2f}, \tselected_mean: {:.3f}, \trejected_mean: {:.3f}'.format(
        #         self.server_name, 100.*np.mean([self.clients_keys[c]=='clean' for c in self.active_clients]),
        #         self.prob_selected, self.prob_rejected, self.selected_mean, self.rejected_mean
        #     )
        # )
        
        self.print_out('Clients selected.')
        if np.sum(self.hasnet_indicator==1) > 0:
            selected_client_states = [clients_state_dict[c] for c in np.where(self.hasnet_indicator==1)[0]]
            return self.process_and_aggregate(clients_state_dict, indices=np.where(self.hasnet_indicator==1)[0])
        else:
            return self.model.model.state_dict()
    
    
    def aggregate(self, clients_state_dict, pre_str=''):
        
        return_model = self.aggregate_defended(clients_state_dict, pre_str=pre_str)
        
        # try:
        #     return_model = self.aggregate_defended(clients_state_dict, pre_str=pre_str)
        # except Exception as e:
        #     print('The exception that I encountered is:', e)
        #     return_model = self.model.model.state_dict()
            
        self.good_indicator = self.hasnet_indicator.copy()
            
        return return_model
    
    
    def _evaluate_server_statistics(self):
        
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
        
        
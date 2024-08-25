import numpy as np
import torch
import copy

from sklearn.cluster import KMeans, HDBSCAN
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
        verbose: bool=True
    ):
        
        super().__init__(
            data, model, clients_with_keys=clients_with_keys, configuration=configuration, verbose=verbose
        )
        
        default_hasnet_configuration = {
            'healing_epochs': 1,
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
        self.kmeans = KMeans(n_clusters=3, n_init='auto')
        self.clusterer = self.kmeans
        
        # preparing hasnets variables
        self.n_dims = 30
        self.rejected_mean, self.selected_mean = -1., 1.
        self.prob_selected, self.prob_rejected = 1., 0.
        self.clients_benign_probabilities = np.array([0.] * len(self.clients))
        
        return
    
    
    def heal_model_from_state(self, model_state):
        
        local_model = Torch_Model(self.healing_data, model_configuration=self.model.model_configuration)
        local_model.model.load_state_dict(model_state)
        local_model.train(epochs=self.configuration['healing_epochs'], batch_size=self.model.model_configuration['batch_size'], verbose=False)
        
        return local_model.model.state_dict()
    
    
    def pairwise_cosine_similarity_torch(self, flattened_clients_states):
        
        normalized_input_a = torch.nn.functional.normalize(flattened_clients_states)  
        res = torch.mm(normalized_input_a, normalized_input_a.T)
        res[res==0] = 1e-6
        
        return res
    
    
    def _compute_gammas(self, clients_state_dict: list[dict]):
        
        # initial_model_state = torch.cat([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        # healed_initial_state = torch.stack([self.parameter_flatten_client_state_torch(self.heal_model_from_state(self.model.model.state_dict(), epochs=self.configuration['healing_epochs']))])
        
        flattened_clients_states = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_state_dict], dim=0)
        # client_to_initial = flattened_clients_states - initial_model_state
        
        perturbed_aggregated_state = self.process_and_aggregate([self.super_aggregate(clients_state_dict)], strength=1e-5)
        flattened_perturbed_aggregated_state = torch.stack([self.parameter_flatten_client_state_torch(perturbed_aggregated_state)], dim=0)
        flattened_aggregated_state = torch.mean(flattened_clients_states, dim=0, keepdim=True)
        healed_state = perturbed_aggregated_state
        
        # Trust index for estimating the direction of healing
        self.gamma = 0
        # self.gamma += self.cosine_similarity( healed_initial_state-initial_model_state, client_to_initial )
        for i in range(self.configuration['healing_epochs']):
            healed_state = self.heal_model_from_state(healed_state)
            flattened_healed_states = torch.stack([self.parameter_flatten_client_state_torch(healed_state)], dim=0)
            self.gamma += self.cosine_similarity( flattened_aggregated_state-flattened_healed_states, flattened_aggregated_state-flattened_clients_states )
            self.gamma += self.cosine_similarity( flattened_perturbed_aggregated_state-flattened_healed_states, flattened_perturbed_aggregated_state-flattened_clients_states )
        self.gamma = self.gamma.view(len(self.active_clients), 1).detach().cpu().numpy()
        
        return
    
    
    def compute_gammas(self, clients_state_dict: list[dict], healed_model_state: dict):
        
        healed_initial_state = torch.stack([self.parameter_flatten_client_state_torch(self.heal_model_from_state(self.model.model.state_dict()))])
        initial_model_state = torch.cat([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        flattened_clients_states = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_state_dict], dim=0)
        flattened_healed_states = torch.stack([self.parameter_flatten_client_state_torch(healed_model_state)], dim=0)
        
        flattened_aggregated_state = torch.mean(flattened_clients_states, dim=0, keepdim=True)
        client_to_initial = flattened_clients_states - initial_model_state
        
        # This is method 2 for estimating the direction of healing
        self.gamma = self.cosine_similarity( flattened_aggregated_state-flattened_healed_states, flattened_aggregated_state-flattened_clients_states )
        self.gamma += self.cosine_similarity( healed_initial_state-initial_model_state, client_to_initial )
        self.gamma = self.gamma.view(len(self.active_clients), 1).detach().cpu().numpy()
        
        # self.client_cs_values = self.pairwise_cosine_similarity_torch(flattened_clients_states).detach().cpu().numpy()
        # self.client_cs_values = np.append(self.client_cs_values, self.client_cs_values*self.gamma, axis=1)
        # self.client_cs_values = np.append(self.client_cs_values, self.gamma, axis=1)
        
        return
    
    
    def fit_clusters(self):
        
        # self.hdb.fit(self.client_cs_values)
        # self.cs_labels_ = self.hdb.labels_
        
        self.processed_gamma = np.concatenate((self.gamma, self.selected_mean-self.gamma, self.gamma-self.rejected_mean), axis=1)
        self.clusterer.fit(self.processed_gamma)
        self.gamma_labels_ = self.clusterer.labels_
        
        return
    
    
    def compute_hasnet_indicator(self):
        
        # u_labels = np.unique(self.cs_labels_)
        # means = []
        # for k in u_labels:
        #     means.append(np.mean([self.gamma[c, 0] for c in np.where(self.cs_labels_==k)]))
        # means = np.array(means)
        # self.best_cs_label = u_labels[np.where(means==np.max(means))]
        # self.worst_cs_label = u_labels[np.where(means==np.min(means))]
        
        u_labels = np.unique(self.gamma_labels_)
        means = []
        for k in u_labels:
            means.append(np.mean([self.gamma[c, 0] for c in np.where(self.gamma_labels_==k)]))
        means = np.array(means)
        self.best_gamma_label = u_labels[np.where(means==np.max(means))]
        self.worst_gamma_label = u_labels[np.where(means==np.min(means))]
        
        self.hasnet_indicator = np.array([1.]*len(self.active_clients))
        self.hasnet_indicator[np.where(self.gamma_labels_!=self.best_gamma_label)] = -1.
        # self.hasnet_indicator[np.where(self.cs_labels_!=self.best_cs_label)] = -1.
        
        return
    
    
    def update_selection_probability(self):
        
        def process_prob(x):
            return np.clip(x*x*x, 0.05, 0.3)
        
        max_mean = np.mean([self.gamma[c, 0] for c in np.where(self.hasnet_indicator==1.)]) if 1. in self.hasnet_indicator else -10
        nonmax_mean = np.mean([self.gamma[c, 0] for c in np.where(self.hasnet_indicator==-1.)]) if -1. in self.hasnet_indicator else 10
        
        epsilon = np.abs(self.selected_mean - self.rejected_mean) * 1e-2
        selection_nearness = 1 / np.clip(self.selected_mean - self.gamma, a_min=epsilon, a_max=None)
        rejection_nearness = 1 / np.clip(self.gamma - self.rejected_mean, a_min=epsilon, a_max=None)
        selection_probs = selection_nearness / (selection_nearness + rejection_nearness)
        
        selected_selected = 1 / np.clip(self.selected_mean - max_mean, a_min=epsilon, a_max=None)
        selected_rejected = 1 / np.clip(max_mean - self.rejected_mean, a_min=epsilon, a_max=None)
        self.prob_selected = selected_selected / (selected_selected + selected_rejected)
        
        rejected_selected = 1 / np.clip(self.selected_mean - nonmax_mean, a_min=epsilon, a_max=None)
        rejected_rejected = 1 / np.clip(nonmax_mean - self.rejected_mean, a_min=epsilon, a_max=None)
        self.prob_rejected = rejected_selected / (rejected_selected + rejected_rejected)
        
        message = 'Both sel_prob {} and rej_prob {} should be <= 1.'.format(self.prob_selected, self.prob_rejected)
        assert self.prob_selected <= 1 and self.prob_rejected <= 1, message
        
        r_alpha = process_prob(self.prob_selected-self.prob_rejected)
        r_beta = r_alpha # process_prob(1-self.prob_rejected)
        self.selected_mean = (1-r_alpha)*self.selected_mean + r_alpha*max_mean
        self.clients_benign_probabilities[self.active_clients] += r_alpha * (selection_probs - 0.5).reshape(-1)
        if (self.prob_selected-self.prob_rejected) > 0.1:
            self.rejected_mean = (1-r_beta)*self.rejected_mean + r_beta*nonmax_mean
            
        return
    
    
    def prepare_clean_clients_indicator(self, clients_state_dict, healed_model_state):
        
        self.compute_gammas(clients_state_dict, healed_model_state)
        self.fit_clusters()
        self.compute_hasnet_indicator()
        self.update_selection_probability()
        
        self.hasnet_indicator[np.where(np.array([self.clients_benign_probabilities[c] for c in self.active_clients]) < 0.)] = -1.
        if self.prob_selected <= (0.5 + self.configuration['epsilon']):
            self.hasnet_indicator = -1. * np.ones_like(self.hasnet_indicator)
            
        self.print_out(
            '\rClean clients: {:.1f}%, \tselected_prob: {:.2f}, \trejected_prob: {:.2f}, \tselected_mean: {:.3f}, \trejected_mean: {:.3f}'.format(
                100.*np.mean([self.clients_keys[c]=='clean' for c in self.active_clients]),
                self.prob_selected, self.prob_rejected, self.selected_mean, self.rejected_mean
            )
        )
        
        return
    
    
    def aggregate(self, clients_state_dict: list[dict], **kwargs):
        
        # aggregate and heal
        aggregated_state = self.super_aggregate(clients_state_dict)
        healed_model_state = self.heal_model_from_state(self.process_and_aggregate([aggregated_state]))
        self.print_out('Model healed (1).')
        
        # select the clients that are believed to be non-poisoned
        self.prepare_clean_clients_indicator(clients_state_dict, healed_model_state)
        selected_client_states = [clients_state_dict[c] for c in np.where(self.hasnet_indicator==1)[0]]
        self.print_out('Clients selected.')
        
        return self.process_and_aggregate(selected_client_states if len(selected_client_states)>0 else [self.model.model.state_dict()])
    
    
    def process_and_aggregate(self, clients_states: list[dict], strength=1e-5):
        
        norms = torch.norm(torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_states], dim=0), p=2, dim=1)
        scalers = torch.min(norms) / norms
        
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
        
        
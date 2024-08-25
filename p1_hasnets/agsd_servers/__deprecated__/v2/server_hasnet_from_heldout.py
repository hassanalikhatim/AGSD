import numpy as np
import copy
import torch

from sklearn.cluster import KMeans


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
        configuration: dict={}
    ):
        
        super().__init__(
            data, model, clients_with_keys=clients_with_keys, configuration=configuration
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
        self.kmeans = KMeans(n_clusters=3, n_init='auto')
        
        # preparing hasnets variables
        self.n_dims = 30
        self.rejected_mean, self.selected_mean = -1., 1.
        self.prob_selected, self.prob_rejected = 1., 0.
        
        return
    
    
    def heal_model_from_state(self, model_state, epochs: int=1):
        
        local_model = Torch_Model(
            self.healing_data, model_configuration=self.model.model_configuration
        )
        local_model.model.load_state_dict(model_state)
        local_model.train(
            epochs=epochs, 
            batch_size=self.model.model_configuration['batch_size'], 
            verbose=False
        )
        
        return local_model.model.state_dict()
    
    
    def cluster_gammas(self):
        
        self.kmeans.fit(self.gamma)
        self.labels_ = self.kmeans.labels_
        
        u_labels = np.unique(self.labels_)
        means = []
        for k in u_labels:
            means.append(np.mean([self.gamma[c] for c in np.where(self.labels_==k)]))
        means = np.array(means)
        
        return means, u_labels[np.where(means==np.max(means))]
    
    
    def update_selection_probability(self, means):
        
        def process_prob(x):
            return np.clip(x*x*x, 0.05, 0.3)
        
        max_mean = np.max(means)
        nonmax_mean = np.mean(means[np.where(means<np.max(means))])
        
        epsilon = np.abs(self.selected_mean - self.rejected_mean) * 1e-2
        selected_selected = 1 / np.clip(self.selected_mean - max_mean, a_min=epsilon, a_max=None)
        selected_rejected = 1 / np.clip(max_mean - self.rejected_mean, a_min=epsilon, a_max=None)
        self.prob_selected = selected_selected / (selected_selected + selected_rejected)
        
        rejected_selected = 1 / np.clip(self.selected_mean - nonmax_mean, a_min=epsilon, a_max=None)
        rejected_rejected = 1 / np.clip(nonmax_mean - self.rejected_mean, a_min=epsilon, a_max=None)
        self.prob_rejected = rejected_selected / (rejected_selected + rejected_rejected)
        
        message = 'Both sel_prob {} and rej_prob {} should be <= 1.'
        assert self.prob_selected <= 1 and self.prob_rejected <= 1, message
        
        r_alpha = process_prob(self.prob_selected)
        r_beta = process_prob(1-self.prob_rejected)
        self.selected_mean = (1-r_alpha)*self.selected_mean + r_alpha*max_mean
        if (self.prob_selected-self.prob_rejected) > 0.1:
            self.rejected_mean = (1-r_beta)*self.rejected_mean + r_beta*nonmax_mean
        
        return
    
    
    def prepare_clean_clients_indicator(
        self, clients_state_dict, healed_model_state
    ):
        
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
        # self.gamma = self.gamma.detach().cpu().numpy()
        
        means, best_label = self.cluster_gammas()
        self.update_selection_probability(means)
        
        print(
            '\rClean clients: {:.2f}%, selected_prob: {:.6f}, rejected_prob: {:.6f}'.format(
                100.*np.mean([self.clients_keys[c]=='clean' for c in self.active_clients]),
                self.prob_selected, self.prob_rejected
            )
        )
        
        if self.prob_selected > (0.5 + self.configuration['epsilon']):
            self.hasnet_indicator = np.array([0]*len(self.active_clients))
            self.hasnet_indicator[np.where(self.labels_==best_label)] = 1
        else:
            self.hasnet_indicator = np.array([0]*len(self.active_clients))
            
        return
    
    
    def aggregate(
        self, clients_state_dict: list[dict], **kwargs
    ):
        
        # aggregate and heal
        aggregated_model_state = super().aggregate(clients_state_dict)
        healed_model_state = self.heal_model_from_state(aggregated_model_state, epochs=1)
        self.print_out('Model healed (1).')
        
        # select the clients that are believed to be non-poisoned
        self.prepare_clean_clients_indicator(clients_state_dict, healed_model_state)
        selected_client_states = [clients_state_dict[c] for c in np.where(self.hasnet_indicator==1)[0]]
        if len(selected_client_states) == 0:
            selected_client_states = [self.model.model.state_dict()]
        self.print_out('Clients selected.')
        
        return self.super_aggregate(selected_client_states)
    
    
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
        
        
import numpy as np
import torch
import copy
import gc
from termcolor import colored

from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering, DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server

from .agsd_adv_attack import AGSD_Adversarial_Attack



class AGSD_ID(Server):
    
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
        self.real_healing_data = Client_Torch_SubDataset(self.data, idxs=healing_data_indices, train_size=0.9)
        
        # preparing some useful functions
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.hdb = HDBSCAN(metric='l2', algorithm='auto', min_cluster_size=int(len(self.clients)*self.configuration['clients_ratio']/2)+1, min_samples=1, allow_single_cluster=True)
        self.db = DBSCAN(eps=5, min_samples=5, metric='precomputed')
        self.spectral = SpectralClustering(n_clusters=2, affinity='precomputed')
        self.kmeans = KMeans(n_clusters=4, n_init='auto')
        self.clusterer = self.spectral
        
        # preparing hasnets variables
        self.clients_benign_probabilities = np.array([0.] * len(self.clients))
        
        # self.model.model.to(torch.float)
        self.my_perturber = AGSD_Adversarial_Attack(self.model)
        self.debug = False
        
        return
    
    
    def sample_clients(self):
        
        probs = np.clip(self.clients_benign_probabilities, -1, 1)
        probs = np.exp(probs) / np.sum(np.exp(probs))
        
        self.active_clients = np.random.choice(
            len(self.clients), int(self.clients_ratio*len(self.clients)), 
            replace=False, p=probs
        )
        
        return
    
    
    def heal_model_from_state(self, model_state, epochs=1):
        
        local_model_configuration = {}
        for key in self.model.model_configuration.keys():
            local_model_configuration[key] = self.model.model_configuration[key]
        local_model_configuration['learning_rate'] = 1e-4
        
        local_model = Torch_Model(self.real_healing_data, model_configuration=local_model_configuration)
        local_model.model.load_state_dict(model_state)
        
        local_model.data = self.my_perturber.attack(local_model, self.real_healing_data, epsilon=0.2)
        local_model.train(epochs=self.configuration['healing_epochs'], batch_size=local_model_configuration['batch_size'], verbose=False)
        
        return local_model.model.state_dict()
    
    
    def pairwise_cosine_similarity_torch(self, flattened_clients_states):
        
        normalized_input_a = torch.nn.functional.normalize(flattened_clients_states)
        res = torch.mm(normalized_input_a, normalized_input_a.T)
        res[res==0] = 1e-6
        
        return res
    
    
    def compute_scalers(self, differences):
        
        norms = torch.norm(differences, p=2, dim=1, keepdim=True).view(-1)
        self.median_norm = torch.median(norms)
        self.scalers = self.median_norm / norms
        
        return
    
    
    def rescaled_noisy_aggregate(self, clients_states: list[dict], scalers: np.ndarray=None, strength=1e-5):
        
        if scalers is None:
            # scalers = np.ones((len(clients_states)))
            initial_state = self.model.model.state_dict()
            initial_model_state = torch.stack([self.parameter_flatten_client_state_torch(initial_state)])
            flattened_states = torch.stack([self.parameter_flatten_client_state_torch(cl_s) for cl_s in clients_states], dim=0)
            norms = torch.norm(flattened_states-initial_model_state, p=2, dim=1, keepdim=True).view(-1)
            scalers = torch.median(norms) / norms
        
        state_t = copy.deepcopy(self.model.model.state_dict())
        delta_state_t = copy.deepcopy(self.model.model.state_dict())
        for key in self.model.model.state_dict().keys():
            for i, client_state in enumerate(clients_states):
                if i==0:
                    delta_state_t[key] = (client_state[key] - self.model.model.state_dict()[key]).clone() * scalers[i] / len(clients_states)
                else:
                    delta_state_t[key] += (client_state[key] - self.model.model.state_dict()[key]).clone() * scalers[i] / len(clients_states)
            
            state_t[key] = state_t[key].float() + delta_state_t[key]
            if not ('bias' in key or 'bn' in key):
                standard_deviation = torch.std(delta_state_t[key].clone().view(-1), unbiased=False)
                try: state_t[key] += strength * torch.normal(0., standard_deviation, size=state_t[key].shape).to(state_t[key].device)
                except: pass

        return state_t
    
    
    def get_std_and_confidence_torch(self, all_models: list[dict], **kwargs):
        
        def get_confidence(arr_in: torch.Tensor): 
            arr_ = torch.exp(arr_in) / torch.sum(torch.exp(arr_in), dim=1, keepdim=True)
            arr_ = torch.mean(arr_, dim=0)
            return torch.max(arr_)
        
        def additive_stds(arr_in: torch.Tensor):
            arr_e = torch.exp(arr_in) / torch.sum(torch.exp(arr_in), dim=1, keepdim=True)
            arr_ = torch.argmax(arr_in, dim=1)
            z_ = torch.nn.functional.one_hot(arr_, num_classes=len(self.data.get_class_names())).float()
            return torch.mean(torch.std(z_, dim=0, unbiased=False))
        
        def multiplicative_stds(arr_in: torch.Tensor): 
            _arr = torch.argmax(arr_in, dim=1)
            final_std = 1.
            for k in torch.unique(_arr):
                arr_ = _arr[_arr==k]
                if len(arr_) > 0:
                    z_ = torch.nn.functional.one_hot(arr_, num_classes=len(self.data.get_class_names())).float()
                    final_std *= torch.mean(torch.std(z_, dim=0, unbiased=False))
            return final_std
        
        local_model = Torch_Model(self.real_healing_data, model_configuration=self.model.model_configuration)
        local_model.model.load_state_dict(self.super_aggregate(all_models))
        healing_data = self.my_perturber.attack(local_model, self.real_healing_data, epsilon=0.3, iterations=30)
        a_stds_ = []; conf_ = []
        for m, model_ in enumerate(all_models):
            local_model.model.load_state_dict(model_)
            adv_y, _ = local_model.predict(torch.utils.data.DataLoader(healing_data.train, shuffle=False, batch_size=local_model.model_configuration['batch_size']), verbose=False)
            self.print_out(f'Going to compute stds and confidences for model {m}.')
            a_stds_.append(additive_stds(adv_y)); conf_.append(get_confidence(adv_y))
            self.print_out(f'Computed stds and confidences for model {m}.')
            
        a_stds_ = torch.stack(a_stds_).to(self.model.device).view(-1)
        conf_ = -1 * torch.stack(conf_).to(self.model.device).view(-1)
        
        return a_stds_, conf_
    
    
    def rescale_a_state(self, state: dict):
        
        initial_state = self.model.model.state_dict()
        initial_model_state = torch.stack([self.parameter_flatten_client_state_torch(initial_state)])
        flattened_state = torch.stack([self.parameter_flatten_client_state_torch(state)])
        scaler_of_state = self.median_norm / torch.norm(flattened_state-initial_model_state, p=2, dim=1, keepdim=True).view(-1)
        flattened_state = initial_model_state.clone() + (flattened_state.clone() - initial_model_state.clone()) * scaler_of_state.view(-1, 1)
        
        rescaled_state = copy.deepcopy(initial_state)
        for key in rescaled_state.keys():
            rescaled_state[key] = rescaled_state[key].float() + (state[key] - initial_state[key]).clone() * scaler_of_state[0]
            
        return rescaled_state, flattened_state
    
    
    def aggregate_defended_torch(self, clients_state_dict: list[dict], **kwargs):
        
        # Functions to normalize torch array values
        def normalized_torch(arr_in: torch.Tensor):
            return torch.exp(arr_in)/torch.sum(torch.exp(arr_in))
        def linearly_normalized_torch(arr_in: torch.Tensor):
            return (arr_in-torch.min(arr_in))/(torch.max(arr_in)-torch.min(arr_in)) if torch.max(arr_in)>torch.min(arr_in) else arr_in/torch.max(arr_in)
        
        
        current_clients = np.array(self.clients_keys)[self.active_clients]
        
        
        ######################################
        # Step 1: Preliminary Aggregation and Clustering
        ######################################
        
        initial_model_state = torch.stack([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        
        flattened_clients_states = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_state_dict], dim=0)
        self.compute_scalers(flattened_clients_states-initial_model_state)
        flattened_clients_states = initial_model_state.clone() + (flattened_clients_states.clone()-initial_model_state.clone()) * self.scalers.view(-1, 1)
        
        # Preliminary aggregation
        perturbed_aggregated_state = self.rescaled_noisy_aggregate(clients_state_dict, scalers=self.scalers, strength=0.)
        perturbed_aggregated_state, flattened_perturbed_aggregated_state = self.rescale_a_state(perturbed_aggregated_state)
        
        # Preliminary clustering of the clients submissions based on the difference from the aggregate model
        self_cs_values = 0
        self_cs_values += self.pairwise_cosine_similarity_torch(flattened_clients_states-flattened_perturbed_aggregated_state)
        self_cs_values += self.pairwise_cosine_similarity_torch(flattened_clients_states-initial_model_state)
        
        self.kmeans.fit(linearly_normalized_torch(self_cs_values).detach().cpu().numpy())
        preliminary_labels_ = self.kmeans.labels_
        preliminary_sorted_labels_ = np.sort(np.unique(preliminary_labels_))
        
        
        ######################################
        # Step 2: Computing the Trust-Index --- gamma (PyTorch Implementation)
        ######################################
        
        # Extend an array based on clusters to the clients
        def extend_to_clients(arr_: np.ndarray, labels: np.ndarray, srt_un_labels: np.ndarray):
            return torch.stack([arr_[srt_un_labels==c] for c in labels]).view(-1)
        
        # Computing the standard deviation, confidence and loss of each cluster's aggregated model
        agg_cl = [self.super_aggregate([clients_state_dict[c] for c in np.where(preliminary_labels_==k)[0]]) for k in preliminary_sorted_labels_]
        a_stds_, conf_ = self.get_std_and_confidence_torch(agg_cl)
        a_stds_ = extend_to_clients(a_stds_, preliminary_labels_, preliminary_sorted_labels_)
        conf_ = extend_to_clients(conf_, preliminary_labels_, preliminary_sorted_labels_)
        
        # Computing the trust index --- gamma --- that we use to identify the right cluster
        gamma_1 = normalized_torch(a_stds_)
        gamma_2 = torch.exp( (torch.min(gamma_1)-torch.max(gamma_1))/(torch.mean(gamma_1)-torch.min(gamma_1)+1e-4) ) * normalized_torch(conf_)
        self_gamma = linearly_normalized_torch(gamma_1 + gamma_2)
        self.gamma = self_gamma.detach().cpu().numpy()
        
        
        ######################################
        # Step 3: Guided Clustering (PyTorch Implementation)
        ######################################
        
        # If the difference from the aggregate model is not very effective, use the guiding of trust index
        gamma_scaling = torch.exp( (torch.min(self_cs_values)-torch.max(self_cs_values))/(torch.mean(self_cs_values)-torch.min(self_cs_values)+1e-4) )
        gamma_gamma_transpose = torch.matmul(self_gamma.view(-1, 1), torch.t(self_gamma.view(-1, 1)))
        self_cs_values +=  gamma_scaling * gamma_gamma_transpose
        self_cs_values = linearly_normalized_torch(self_cs_values)
        
        # convert Pytorch values into numpy for later use
        self.cs_values = self_cs_values.detach().cpu().numpy()
        
        # cluster using spectral clustering
        self.clusterer.fit(self.cs_values)
        self.labels_ = self.clusterer.labels_
        sorted_labels = np.sort(np.unique(self.labels_))
        
        
        ######################################
        # Step 4: Stateful Selection of the Clusters and Final Aggregation
        ######################################
        
        # Finding the best cluster based on the trust index --- gamma --- to make updates
        def get_best_label(values_: np.ndarray):
            _a_means = np.array([np.mean(values_[np.where(self.labels_==label)]) for label in sorted_labels])
            return sorted_labels[np.argmax(_a_means).reshape(-1)[0]]
        
        # Identify the best label
        best_label = get_best_label(self.gamma)
        
        self.good_indicator = np.zeros_like(self.good_indicator)
        if np.sum(np.isnan(self.gamma)) == 0:
            # update the trust state history --- clients with the best label improve, while clients with worst label degrade
            self.clients_benign_probabilities[self.active_clients[np.where(self.labels_==best_label)]] += abs(self.gamma[np.where(self.labels_==best_label)])
            self.clients_benign_probabilities[self.active_clients[np.where(self.labels_!=best_label)]] -= abs(1 - self.gamma[np.where(self.labels_!=best_label)])
            
            # select good clients based on the best label
            self.good_indicator[np.where(self.labels_==best_label)] = 1.
            # stateful filtering of the selected clients --- those clients which have bad history are ruled out of the aggregation
            self.good_indicator[np.where(self.clients_benign_probabilities[self.active_clients] < 0.)] = 0.
        
        if 1 in self.good_indicator:
            return_model = self.rescaled_noisy_aggregate([clients_state_dict[c] for c in np.where(self.good_indicator==1)[0]], strength=5e-5)
        else: return_model = self.model.model.state_dict()
        
        
        ######################################
        # Here you can type things that you want to print along with the original output.
        ######################################
        
        self.a_msg_that_i_need_to_print = ''
        
        def get_max_mean_min(values_: np.ndarray, str_: str, default_color: str='green'):
            return_str_ = ''
            if 'clean' in current_clients and np.mean(current_clients=='clean') < 1:
                c_max_s, c_mean_s, c_min_s = np.max(values_[current_clients=='clean']), np.mean(values_[current_clients=='clean']), np.min(values_[current_clients=='clean'])
                b_max_s, b_mean_s, b_min_s = np.max(values_[current_clients!='clean']), np.mean(values_[current_clients!='clean']), np.min(values_[current_clients!='clean'])
                color = 'light_red' if c_min_s<=b_max_s else default_color
                return_str_ = colored(f'{str_}:{c_max_s:5.2f},', default_color) + colored(f'{c_min_s:5.2f}|{b_max_s:5.2f}', color) + colored(f',{b_min_s:5.2f})', default_color)
            return return_str_
        
        def get_clean_selected_ratio(arr_: np.ndarray):
            selected_clients = np.array(self.clients_keys)[self.active_clients]
            selected_clients = selected_clients[np.where((self.labels_==get_best_label(arr_)))]
            return np.mean(selected_clients=='clean') if len(selected_clients)>0 else -1
        
        colors = ['light_cyan', 'light_red', 'green']
        values_dict = {
            'g': self.gamma,
            'a': a_stds_.detach().cpu().numpy(),
            'c': conf_.detach().cpu().numpy(),
        }
        values = [get_clean_selected_ratio(values_dict[key]) for key in values_dict.keys()]
        self.a_msg_that_i_need_to_print += '|'.join([colored(f'({key}-{values[k]:+.2f})', colors[int(values[k])+1]) for k, key in enumerate(values_dict.keys())])
        
        self.a_msg_that_i_need_to_print += '| ' + get_max_mean_min(self.gamma, 'G')
        self.a_msg_that_i_need_to_print += ' | ' + get_max_mean_min(self.clients_benign_probabilities[self.active_clients], 'H')
        
        if self.debug:
            self.a_msg_that_i_need_to_print += '\n-----------------------------------'
            self.a_msg_that_i_need_to_print += '\n' + get_max_mean_min(a_stds_, 'st')
            self.a_msg_that_i_need_to_print += '\n' + get_max_mean_min(conf_, 'cf')
            self.a_msg_that_i_need_to_print += '\n\n'
        
        return return_model
    
    
    def aggregate(self, clients_state_dict, pre_str=''):
        
        return_model = self.aggregate_defended_torch(clients_state_dict, pre_str=pre_str)
        
        # try:
        #     return_model = self.aggregate_defended(clients_state_dict, pre_str=pre_str)
        # except Exception as e:
        #     print('The exception that I encountered is:', e)
        #     return_model = self.model.model.state_dict()
        
        return return_model
    
    
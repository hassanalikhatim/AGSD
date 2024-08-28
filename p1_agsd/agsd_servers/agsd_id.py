import numpy as np
import torch
import copy
import gc
from termcolor import colored
import time

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
            'epsilon': 0.1,
            'n_clusters': 2
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
        self.spectral = SpectralClustering(n_clusters=self.configuration['n_clusters'], affinity='precomputed')
        self.kmeans = KMeans(n_clusters=self.configuration['n_clusters'], n_init='auto')
        self.clusterer = self.spectral
        
        # preparing hasnets variables
        self.clients_benign_probabilities = np.array([0.] * len(self.clients))
        
        # self.model.model.to(torch.float)
        self.my_perturber = AGSD_Adversarial_Attack(self.model)
        self.debug = False
        
        return
    
    
    def sample_clients(self):
        
        # probs = np.clip(self.clients_benign_probabilities, -1, 1)
        probs = np.zeros_like(self.clients_benign_probabilities)
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
        
        # local_model.data = self.my_perturber.attack(local_model, self.real_healing_data, epsilon=0.2)
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
                delta = (client_state[key] - self.model.model.state_dict()[key]).clone() * scalers[i]
                try: delta += strength * torch.normal(0., torch.std(delta.view(-1), unbiased=False), size=state_t[key].shape).to(state_t[key].device)
                except: pass
                if i==0:
                    delta_state_t[key] = delta / len(clients_states)
                else:
                    delta_state_t[key] += delta / len(clients_states)
            
            # deltas = torch.stack([(client_state[key]-self.model.model.state_dict()[key]).clone()*scalers[i] for i, client_state in enumerate(clients_states)], dim=0)
            # delta_state_t[key] = torch.median(torch.stack([delta for delta in deltas], dim=0), dim=0)[0]
            # delta_state_t[key] = torch.mean(torch.stack([delta for delta in deltas], dim=0), dim=0)
            
            state_t[key] = state_t[key].float() + delta_state_t[key]
            # if not ('bias' in key or 'bn' in key):
            #     standard_deviation = torch.std(delta_state_t[key].clone().view(-1), unbiased=False)
            #     standard_deviation_state = torch.std(state_t[key].clone().view(-1), unbiased=False)
            #     try: state_t[key] += strength * torch.normal(0., standard_deviation, size=state_t[key].shape).to(state_t[key].device)
            #     except: pass
            #     # try: state_t[key] += 1e-6 * torch.normal(0., standard_deviation_state, size=state_t[key].shape).to(state_t[key].device)
            #     # except: pass
                
        return state_t
    
    
    def get_std_and_confidence_torch(self, all_models: list[dict], **kwargs):
        
        def get_confidence(arr_in: torch.Tensor): 
            arr_ = torch.exp(arr_in) / torch.sum(torch.exp(arr_in), dim=1, keepdim=True)
            # arr_ = torch.nn.functional.softmax(arr_in, dim=1)
            arr_ = torch.mean(arr_, dim=0)
            return torch.max(arr_)
        
        def additive_stds(arr_in: torch.Tensor): 
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
        
        def diff_norm(arr_1: torch.Tensor, arr_2: torch.Tensor):
            _arr_1 = torch.exp(arr_1) / torch.sum(torch.exp(arr_1), dim=1, keepdim=True)
            _arr_2 = torch.exp(arr_2) / torch.sum(torch.exp(arr_2), dim=1, keepdim=True)
            return torch.norm(_arr_1-_arr_2, p=2)
            # return torch.norm(arr_1-arr_2, p=1)
        
        local_model = Torch_Model(self.real_healing_data, model_configuration=self.model.model_configuration)
        local_model.model.load_state_dict(self.super_aggregate(all_models))
        healing_data = self.my_perturber.attack(local_model, self.real_healing_data, epsilon=0.3, iterations=30)
        a_stds_ = []; conf_ = []; loss_ = []
        for m, model_ in enumerate(all_models):
            local_model.model.load_state_dict(model_)
            # healing_data = self.my_perturber.attack(local_model, self.real_healing_data, epsilon=0.3, iterations=30)
            # act_y, _ = local_model.predict(torch.utils.data.DataLoader(self.real_healing_data.train, shuffle=False, batch_size=local_model.model_configuration['batch_size']), verbose=False)
            adv_y, _ = local_model.predict(torch.utils.data.DataLoader(healing_data.train, shuffle=False, batch_size=local_model.model_configuration['batch_size']), verbose=False)
            
            if torch.sum(torch.isnan(adv_y))>0: 
                a_stds_.append(torch.tensor(-1.).to(local_model.device)); conf_.append(torch.tensor(-1.).to(local_model.device))
            else:
                a_stds_.append(additive_stds(adv_y.clone())); conf_.append(get_confidence(adv_y.clone()))
        
        a_stds_ = torch.stack(a_stds_).to(self.model.device).view(-1)
        conf_ = -1 * torch.stack(conf_).to(self.model.device).view(-1)
        # loss_ = -1 * torch.stack(loss_).to(self.model.device).view(-1)
        
        return a_stds_, conf_, loss_
    
    
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
        
        self.good_indicator = np.zeros_like(self.good_indicator)
        
        
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
        
        # Clustering the clients submissions based on the difference from the aggregate model
        def dot_vec_vec_t(arr_in: torch.Tensor):
            return torch.matmul(arr_in.view(-1, 1), torch.t(arr_in.view(-1, 1)))
        self_cs_values = 0
        self_cs_values += self.pairwise_cosine_similarity_torch( flattened_clients_states-flattened_perturbed_aggregated_state )
        self_cs_values += self.pairwise_cosine_similarity_torch( flattened_clients_states-initial_model_state )
        self_cs_values = linearly_normalized_torch(self_cs_values)
        
        # convert Pytorch values into numpy for later use
        self.cs_values = self_cs_values.detach().cpu().numpy()
        
        # cluster using spectral clustering
        self.clusterer.fit(self.cs_values)
        self.labels_ = self.clusterer.labels_
        sorted_labels = np.sort(np.unique(self.labels_))
        
        
        ######################################
        # Step 2: Computing the Trust-Index --- gamma (PyTorch Implementation)
        ######################################
        
        def non_weight_of_this_value(arr_in: torch.Tensor):
            return torch.exp( (torch.min(arr_in)-torch.max(arr_in))/(torch.mean(arr_in)-torch.min(arr_in)+1e-4) )
        
        # Computing the standard deviation, confidence and loss of each client
        a_stds_, conf_, loss_ = self.get_std_and_confidence_torch(clients_state_dict)
        
        # Computing the trust index --- gamma --- that we use to identify the right cluster
        gamma_1 = normalized_torch(a_stds_)
        gamma_2 = non_weight_of_this_value(gamma_1) * normalized_torch(conf_)
        self_gamma = linearly_normalized_torch(gamma_1 + gamma_2)
        self.gamma = self_gamma.detach().cpu().numpy()
        
        assert torch.sum(torch.isnan(a_stds_)) == 0, 'There should not be any nan values in stds array.'
        assert torch.sum(torch.isnan(conf_)) == 0, 'There should not be any nan values in conf array.'
        assert torch.sum(torch.isnan(self_gamma)) == 0, 'There should not be any nan values in trust index array.'
        
        
        ######################################
        # Step 4: Stateful Selection of the Clusters and Final Aggregation
        ######################################
        
        # Finding the best cluster based on the trust index --- gamma --- to make updates
        def get_best_condition(values_: np.ndarray):
            _a_means = np.array([np.mean(values_[np.where(self.labels_==label)]) for label in sorted_labels])
            return self.labels_ == sorted_labels[np.argmax(_a_means).reshape(-1)[0]]
        def get_worst_condition(values_: np.ndarray):
            _a_means = np.array([np.mean(values_[np.where(self.labels_==label)]) for label in sorted_labels])
            return self.labels_ == sorted_labels[np.argmin(_a_means).reshape(-1)[0]]
        
        # Identify the best label, also identify the best and the worst conditions for cluster selection and rejection respectively
        best_condition = get_best_condition(self.gamma)
        worst_condition = get_worst_condition(self.gamma)
        
        self.good_indicator = np.zeros_like(self.good_indicator)
        if np.max(self.gamma) > np.min(self.gamma):
            
            # update the trust state history --- clients with the best label improve, while clients with worst label degrade
            self.clients_benign_probabilities[self.active_clients[np.where(best_condition)]] += abs(self.gamma[np.where(best_condition)])
            self.clients_benign_probabilities[self.active_clients[np.where(worst_condition)]] -= abs(1 - self.gamma[np.where(worst_condition)])
            
            # select good clients based on the best label
            self.good_indicator[np.where(best_condition)] = 1.
            
            # stateful filtering of the selected clients --- those clients which have bad history are ruled out of the aggregation
            self.good_indicator[np.where(self.clients_benign_probabilities[self.active_clients] < 0.)] = 0.
        
        if 1 in self.good_indicator: return_model = self.rescaled_noisy_aggregate([clients_state_dict[c] for c in np.where(self.good_indicator==1)[0]], strength=1e-5)
        else: return_model = self.model.model.state_dict()
        
        
        ######################################
        # Here you can type things that you want to print along with the original output.
        ######################################
        
        time_out_start = time.time()
        current_clients = np.array(self.clients_keys)[self.active_clients]
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
            best_condition = get_best_condition(arr_)
            selected_clients = current_clients[np.where((best_condition))]
            return np.mean(selected_clients=='clean') if len(selected_clients)>0 else -1
        
        colors = ['light_cyan', 'light_red', 'green']
        values_dict = {
            'g': self.gamma,
            's': a_stds_.detach().cpu().numpy(),
            'c': conf_.detach().cpu().numpy(),
            # 'h1': healing_gamma_1.detach().cpu().numpy(),
            # 'h2': healing_gamma_2.detach().cpu().numpy(),
            # 'H': healing_gamma.detach().cpu().numpy(),
        }
        values = [get_clean_selected_ratio(values_dict[key]) for key in values_dict.keys()]
        self.a_msg_that_i_need_to_print += '|'.join([colored(f'({key}-{values[k]:+.2f})', colors[int(values[k])+1]) for k, key in enumerate(values_dict.keys())])
        
        # self.a_msg_that_i_need_to_print += '|' + colored(f'Fls', 'red') if torch.sum(torch.isnan(self_gamma))>0 else f'| Tru'
        # self.a_msg_that_i_need_to_print += f'|{non_weight_of_this_value(self_cs_values):.3f}'
        self.a_msg_that_i_need_to_print += '|' + get_max_mean_min(self.gamma, 'G')
        # self.a_msg_that_i_need_to_print += '|' + get_max_mean_min(healing_gamma.detach().cpu().numpy(), 'H')
        self.a_msg_that_i_need_to_print += '|' + get_max_mean_min(self.clients_benign_probabilities[self.active_clients], 'H')
        
        if self.debug:
            self.a_msg_that_i_need_to_print += '\n-----------------------------------'
            self.a_msg_that_i_need_to_print += '\n' + get_max_mean_min(a_stds_, 'st')
            self.a_msg_that_i_need_to_print += '\n' + get_max_mean_min(conf_, 'cf')
            self.a_msg_that_i_need_to_print += '\n\n'
            
        self.time_out = time.time() - time_out_start
        
        return return_model
    
    
    def aggregate(self, clients_state_dict, pre_str=''):
        
        return_model = self.aggregate_defended_torch(clients_state_dict, pre_str=pre_str)
        
        # try:
        #     return_model = self.aggregate_defended_torch(clients_state_dict, pre_str=pre_str)
        # except Exception as e:
        #     print('The exception that I encountered is:', e)
        #     return_model = self.model.model.state_dict()
        
        return return_model
    
    
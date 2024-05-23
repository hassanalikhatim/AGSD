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

from ...hgsd_adv_attack import HGSD_Adversarial_Attack



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
        self.real_healing_data = Client_Torch_SubDataset(self.data, idxs=healing_data_indices, train_size=0.9)
        
        # preparing some useful functions
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.hdb = HDBSCAN(metric='l2', algorithm='auto', min_cluster_size=int(len(self.clients)*self.configuration['clients_ratio']/2)+1, min_samples=1, allow_single_cluster=True)
        self.db = DBSCAN(eps=5, min_samples=5, metric='precomputed')
        self.spectral = SpectralClustering(n_clusters=2, affinity='precomputed')
        self.kmeans = KMeans(n_clusters=2, n_init='auto')
        self.clusterer = self.spectral
        self.clusterer_2 = self.spectral
        
        # preparing hasnets variables
        self.n_dims = 30
        self.rejected_mean, self.selected_mean = -1., 1.
        self.prob_selected, self.prob_rejected = 1., 0.
        self.clients_benign_probabilities = np.array([0.] * len(self.clients))
        
        self.current_round = 0
        self.server_name = f'hgsd_(id-{{{healing_set_size}}})'
        self.model.model.to(torch.float)
        self.end_shape = (len(self.parameter_flatten_client_state_np(self.model.model.state_dict()).reshape(-1))//2)*2
        self.my_perturber = HGSD_Adversarial_Attack(self.model)
        self.good_agg_h = np.zeros((1, 2))
        self.bad_agg_h = np.zeros((1, 2))
        self.agg_h = np.zeros((1, 2))
        
        self.n_stored_models = 10
        self.last_n_models = []
        self.delta = None
        
        return
    
    
    def sample_clients(self):
        
        probs = np.clip(self.clients_benign_probabilities, a_min=0, a_max=None)
        probs = np.exp(self.clients_benign_probabilities) / np.sum(np.exp(self.clients_benign_probabilities))
        
        self.active_clients = np.random.choice(
            len(self.clients), int(self.clients_ratio*len(self.clients)), 
            replace=False, p=probs
        )
        
        return
    
    
    def heal_model_from_state(self, model_state, epochs=1):
        
        local_model_configuration = {}
        for key in self.model.model_configuration.keys():
            local_model_configuration[key] = self.model.model_configuration[key]
        local_model_configuration['learning_rate'] = 1e-5
        
        local_model = Torch_Model(self.real_healing_data, model_configuration=local_model_configuration)
        local_model.model.load_state_dict(model_state)
        
        healing_data = self.my_perturber.attack(local_model, self.real_healing_data, epsilon=0.3)
        local_model.data = healing_data
        local_model.train(epochs=self.configuration['healing_epochs'], batch_size=local_model_configuration['batch_size'], verbose=False)
        
        return local_model.model.state_dict()
    
    
    def pairwise_cosine_similarity_torch(self, flattened_clients_states):
        
        normalized_input_a = torch.nn.functional.normalize(flattened_clients_states)
        res = torch.mm(normalized_input_a, normalized_input_a.T)
        res[res==0] = 1e-6
        
        return res
    
    
    def compute_scalers(self, differences):
        
        norms = torch.norm(differences, p=2, dim=1, keepdim=True).view(-1)
        self.scalers = torch.median(norms) / norms
        
        return
    
    
    def process_and_aggregate(
        self, 
        clients_states: list[dict], indices: list[int]=None, strength=1e-5, 
        use_hgsd_indicator: bool=False, compute_local_scales: bool=False
    ):
        
        if indices is None:
            indices = np.arange(len(clients_states))
        
        hgsd_indicator = self.scalers.clone()
        if compute_local_scales:
            initial_model_state = torch.stack([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
            flattened_clients_states = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in [clients_states[k] for k in indices]], dim=0)
            norms = torch.norm(flattened_clients_states-initial_model_state, p=2, dim=1, keepdim=True).view(-1)
            hgsd_indicator[indices] = torch.median(norms) / norms
        
        if use_hgsd_indicator:
            hgsd_indicator *= torch.tensor(np.exp(self.hasnet_indicator-np.max(self.hasnet_indicator))).to(self.scalers.device)
        
        state_t = copy.deepcopy(self.model.model.state_dict())
        delta_state_t = copy.deepcopy(self.model.model.state_dict())
        for key in self.model.model.state_dict().keys():
            for i, ind in enumerate(indices):
                if i==0:
                    delta_state_t[key] = (clients_states[ind][key] - self.model.model.state_dict()[key]).clone() * hgsd_indicator[ind] / len(indices)
                else:
                    delta_state_t[key] += (clients_states[ind][key] - self.model.model.state_dict()[key]).clone() * hgsd_indicator[ind] / len(indices)
            
            if not ('bias' in key or 'bn' in key):
                standard_deviation = torch.std(delta_state_t[key].clone().view(-1), unbiased=False)
                try: state_t[key] += delta_state_t[key] + strength * torch.normal(0., standard_deviation, size=state_t[key].shape).to(state_t[key].device)
                except: pass

        return state_t
    
    
    def compute_gammas(self, clients_state_dict: list[dict]):
        
        initial_model_state = torch.stack([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        flattened_clients_states = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_state_dict], dim=0)
        self.compute_scalers(flattened_clients_states-initial_model_state)
        flattened_clients_states = initial_model_state.clone() + (flattened_clients_states.clone()-initial_model_state.clone()) * self.scalers.view(-1, 1)
        
        healed_initial_state = torch.stack([self.parameter_flatten_client_state_torch(self.heal_model_from_state(self.model.model.state_dict(), epochs=self.configuration['healing_epochs']))])
        client_to_initial = flattened_clients_states - initial_model_state
        
        perturbed_aggregated_state = self.process_and_aggregate(clients_state_dict, strength=0.)
        flattened_perturbed_aggregated_state = torch.stack([self.parameter_flatten_client_state_torch(perturbed_aggregated_state)], dim=0)
        flattened_aggregated_state = torch.mean(flattened_clients_states, dim=0, keepdim=True)
        flattened_healed_states = torch.stack([self.parameter_flatten_client_state_torch(self.heal_model_from_state(perturbed_aggregated_state, epochs=self.configuration['healing_epochs']))], dim=0)
        
        self.cs_values = self.pairwise_cosine_similarity_torch(flattened_clients_states-flattened_aggregated_state).detach().cpu().numpy()
        self.l2_values = torch.mean(flattened_clients_states-flattened_aggregated_state, dim=1, keepdim=True).detach().cpu().numpy()
        self.l2_values = np.dot(self.l2_values, self.l2_values.T)
        # self.cs_values += self.pairwise_cosine_similarity_torch(client_to_initial).detach().cpu().numpy()
        
        # Trust index for estimating the direction of healing
        self.gamma_1 = 1 + self.cosine_similarity( flattened_healed_states-flattened_aggregated_state, flattened_clients_states-flattened_aggregated_state )
        self.gamma_2 = 1 + self.cosine_similarity( healed_initial_state-initial_model_state, client_to_initial )
        self.gamma_3 = 1 + self.cosine_similarity( flattened_perturbed_aggregated_state-flattened_healed_states, flattened_perturbed_aggregated_state-flattened_clients_states )
        self.gamma = self.gamma_1 + self.gamma_2 + self.gamma_3
        
        # if self.delta is not None:
        #     self.gamma += 1 + self.cosine_similarity( client_to_initial, self.delta )
        #     self.cs_values += 1 + self.pairwise_cosine_similarity_torch(client_to_initial-self.delta).detach().cpu().numpy()
        
        self.gamma = self.gamma.view(len(self.active_clients)).detach().cpu().numpy()
        
        # self.cs_values *= np.dot(self.gamma.reshape(-1, 1), self.gamma.reshape(-1, 1).T)
        self.cs_values = (self.cs_values-np.min(self.cs_values)) / (np.max(self.cs_values)-np.min(self.cs_values))
        self.l2_values = (self.l2_values-np.min(self.l2_values)) / (np.max(self.l2_values)-np.min(self.l2_values))
        # self.cs_values = normalize(self.cs_values, norm='l2')
        
        return
    
    
    def fit_clusters(self):
        
        self.clusterer.fit(self.cs_values)
        self.gamma_labels_ = self.clusterer.labels_
        
        self.clusterer_2.fit(self.l2_values)
        self.labels_2 = self.clusterer_2.labels_ + 1
        
        self.agreement = np.ones_like(self.gamma_labels_)
        labels_consistency = np.mean(self.gamma_labels_*self.labels_2)
        labels_inverted_consistency = np.mean(self.gamma_labels_*(1-self.labels_2))
        if labels_consistency > labels_inverted_consistency:
            self.agreement[np.where(self.gamma_labels_ != self.labels_2)] = 0
        elif labels_consistency < labels_inverted_consistency:
            self.agreement[np.where(self.gamma_labels_ == self.labels_2)] = 0
        self.sorted_labels = np.sort(np.unique(self.gamma_labels_))
        
        return
    
    
    def deprecated_compute_hasnet_indicator(self):
        
        self.best_gamma_label = self.sorted_labels[np.where(self.means==np.max(self.means))[0][0]]
        self.worst_gamma_label = self.sorted_labels[np.where(self.means==np.min(self.means))[0][0]]
        
        return
    
    
    def deprecated_update_selection_probability(self):
        
        def process_prob(x):
            return np.clip(x*x*x, 0.05, 0.3)
        
        max_mean = np.mean([self.gamma[c] for c in np.where(self.gamma_labels_==self.best_gamma_label)]) if self.best_gamma_label != self.worst_gamma_label else -10
        nonmax_mean = np.mean([self.gamma[c] for c in np.where(self.gamma_labels_!=-self.best_gamma_label)]) if self.best_gamma_label != self.worst_gamma_label else 10
        # self.gamma_diff = '{:.3f}, {:.3f}, {:.3f}'.format(max_mean, nonmax_mean, max_mean - nonmax_mean)
        
        epsilon = np.abs(self.selected_mean - self.rejected_mean) * 1e-2
        
        selected_selected = 1 / np.clip(self.selected_mean - max_mean, a_min=epsilon, a_max=None)
        selected_rejected = 1 / np.clip(max_mean - self.rejected_mean, a_min=epsilon, a_max=None)
        self.prob_selected = selected_selected / (selected_selected + selected_rejected)
        
        rejected_selected = 1 / np.clip(self.selected_mean - nonmax_mean, a_min=epsilon, a_max=None)
        rejected_rejected = 1 / np.clip(nonmax_mean - self.rejected_mean, a_min=epsilon, a_max=None)
        self.prob_rejected = rejected_selected / (rejected_selected + rejected_rejected)
        
        message = f'Both sel_prob {self.prob_selected} and rej_prob {self.prob_rejected} should be <= 1.'
        assert self.prob_selected <= 1 and self.prob_rejected <= 1, message
        
        r_alpha = process_prob(self.prob_selected-self.prob_rejected)
        r_beta = self.prob_selected-self.prob_rejected
        self.selected_mean = (1-r_alpha)*self.selected_mean + r_alpha*max_mean
        self.rejected_mean = (1-r_alpha)*self.rejected_mean + r_alpha*nonmax_mean
        
        return r_alpha
    
    
    def get_best_model(self, all_models: list[dict], **kwargs):
        
        def hl_prob(arr_in: torch.Tensor):
            arr_in = arr_in.clone().detach().cpu().numpy()
            # arr_ = np.argmax(arr_in, axis=1)
            # z_ = np.zeros((len(arr_), len(self.data.get_class_names())))
            # z_[np.arange(arr_.size), arr_] = 1
            return np.max(np.mean(np.exp(arr_in)/np.sum(np.exp(arr_in), axis=1, keepdims=True), axis=0))
        def hl_std(arr_in: torch.Tensor):
            arr_in = arr_in.clone().detach().cpu().numpy()
            arr_ = np.argmax(arr_in, axis=1)
            z_ = np.zeros((len(arr_), len(self.data.get_class_names())))
            z_[np.arange(arr_.size), arr_] = 1
            return np.exp(np.mean(np.std(arr_in, axis=0)))
        def hl_norm(arr_in: torch.Tensor):
            return torch.norm(arr_in, p=2).item()
        
        local_model = Torch_Model(self.real_healing_data, model_configuration=self.model.model_configuration)
        stds_ = []; norms_ = []; losses_ = []; diffs_ = []
        for model_ in all_models:
            local_model.model.load_state_dict(model_)
            healing_data = self.my_perturber.attack(local_model, self.real_healing_data, epsilon=0.3, iterations=30)
            # current_pred, _ = self.model.predict(torch.utils.data.DataLoader(healing_data.train, shuffle=False, batch_size=local_model.model_configuration['batch_size']))
            adv_y, _ = local_model.predict(torch.utils.data.DataLoader(healing_data.train, shuffle=False, batch_size=local_model.model_configuration['batch_size']))
            cl_y, _ = local_model.predict(torch.utils.data.DataLoader(self.real_healing_data.train, shuffle=False, batch_size=local_model.model_configuration['batch_size']))
            l, _ = local_model.test_shot(torch.utils.data.DataLoader(healing_data.train), verbose=False)
            stds_.append(hl_std(adv_y)); norms_.append(hl_prob(adv_y)); losses_.append(l); diffs_.append(hl_norm(adv_y-cl_y))
        stds_ = np.array(stds_); norms_ = 1/np.array(norms_); losses_ = 1/np.array(losses_); diffs_ = 1/np.array(diffs_)
        
        def processed_sigmoid(x, z: int=1):
            values = 1 / (1 + np.exp(-x*z))
            return (values-0.5)*2
        # sigmoided_std_diff = processed_sigmoid(np.max(stds_), z=6)# - np.min(self.stds_), z=5)
        initial_model_state = torch.stack([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        f_agg_cl = [torch.stack([self.parameter_flatten_client_state_torch(agcl)]) for agcl in all_models]
        f_agg_cl_cs = np.array([self.cosine_similarity(initial_model_state, fagcl).item() for fagcl in f_agg_cl])
        # conditioned_std = np.log(1 - f_agg_cl_cs)
        
        self.a_stds_ = np.array([stds_[np.where(self.sorted_labels==gl)] for gl in self.gamma_labels_]).reshape(-1)
        self.a_norms_ = np.array([norms_[np.where(self.sorted_labels==gl)] for gl in self.gamma_labels_]).reshape(-1)
        self.a_cs_ = np.array([f_agg_cl_cs[np.where(self.sorted_labels==gl)] for gl in self.gamma_labels_]).reshape(-1)
        self.a_losses_ = np.array([losses_[np.where(self.sorted_labels==gl)] for gl in self.gamma_labels_]).reshape(-1)
        self.a_diffs_ = np.array([diffs_[np.where(self.sorted_labels==gl)] for gl in self.gamma_labels_]).reshape(-1)
        
        return
    
    
    def update_hasnet_indicator(self, arr_: np.ndarray):
        best_condition = (arr_==np.max(arr_)); worst_condition = (arr_!=np.max(arr_))
        self.hasnet_indicator[np.where(best_condition)] += 1; self.hasnet_indicator[np.where(worst_condition)] -= 1
        return
    
    
    def aggregate_defended(self, clients_state_dict: list[dict], **kwargs):
        
        self.compute_gammas(clients_state_dict); gc.collect()
        self.fit_clusters()
        
        agg_cl = [self.process_and_aggregate(clients_state_dict, indices=np.where(self.gamma_labels_==k)[0]) for k in self.sorted_labels]
        self.get_best_model(agg_cl); gc.collect()
        
        means = []
        for k in self.sorted_labels:
            means.append(np.mean([self.gamma[c] for c in np.where(self.gamma_labels_==k)]))
        means = np.array(means)
        self.a_means = np.array([means[np.where(self.sorted_labels==gl)] for gl in self.gamma_labels_]).reshape(-1)
        self.a_size_ = np.array([np.sum(self.gamma_labels_==gl) for gl in self.sorted_labels])
        self.a_size_ = np.array([self.a_size_[np.where(self.sorted_labels==gl)] for gl in self.gamma_labels_]).reshape(-1)
        
        these_clients_probs = self.clients_benign_probabilities[self.active_clients].copy()
        m_trust = np.array([np.mean(these_clients_probs[np.where(self.gamma_labels_==k)]) for k in self.sorted_labels])
        self.a_trust_ = np.array([m_trust[np.where(self.sorted_labels==gl)] for gl in self.gamma_labels_]).reshape(-1)
        # r_alpha = self.update_selection_probability()
        
        self.hasnet_indicator = np.array([0.]*len(self.active_clients))
        metrics_to_use = [self.a_means] #*self.a_stds_*self.a_norms_*self.a_losses_, self.a_size_] #, self.a_cs_, self.a_diffs_]
        _ = [self.update_hasnet_indicator(metric.copy()) for metric in metrics_to_use]
        self.hasnet_indicator /= len(metrics_to_use)
        # self.hasnet_indicator *= self.agreement
        
        def process_prob(x):
            return np.sign(x) * np.clip(np.abs(x**3), 0.05, 0.1)
        self.clients_delta = np.zeros_like(self.hasnet_indicator)
        if np.max(self.hasnet_indicator) > np.min(self.hasnet_indicator):
            # self.clients_delta = -1 + 2 * (self.hasnet_indicator - np.min(self.hasnet_indicator)) / (np.max(self.hasnet_indicator) - np.min(self.hasnet_indicator))
            # self.clients_delta = self.hasnet_indicator - np.mean(self.hasnet_indicator)
            self.clients_delta = self.hasnet_indicator.copy()
            # self.clients_delta[np.where(self.hasnet_indicator==np.max(self.hasnet_indicator))] = 1
            # self.clients_delta = process_prob(self.clients_delta)
        else:
            self.hasnet_indicator *= 0.
            
        self.prepare_print_message()
        self.print_out('Clients selected.')
        
        self.clients_benign_probabilities[self.active_clients] += self.clients_delta
        self.hasnet_indicator[np.where(self.clients_benign_probabilities[self.active_clients] < 0.)] = 0.
        if np.max(self.hasnet_indicator)>np.min(self.hasnet_indicator): 
            return_model = self.process_and_aggregate(
                clients_state_dict, indices=np.where(self.hasnet_indicator==np.max(self.hasnet_indicator))[0], 
                use_hgsd_indicator=True, compute_local_scales=False
            )
            self.delta = torch.stack([self.parameter_flatten_client_state_torch(return_model)]) - torch.stack([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        else: return_model = self.model.model.state_dict()
        
        return return_model
    
    
    def prepare_print_message(self):
        
        colors = ['light_cyan', 'light_red', 'green']
        def get_clean_selected_ratio(arr_in: list[np.ndarray]):
            extra_condition = (self.agreement > -10)
            arr_ = arr_in[0].reshape(-1)
            selected_clients = np.array(self.clients_keys)[self.active_clients]
            if len(arr_in) == 1:
                selected_clients = selected_clients[np.where((arr_==np.max(arr_) * extra_condition))]
            elif len(arr_in) == 2:
                arr_2 = arr_in[1].reshape(-1)
                selected_clients = selected_clients[np.where((arr_==np.max(arr_)) * (arr_2==np.max(arr_2) * extra_condition))]
            return np.mean(selected_clients=='clean') if len(selected_clients)>0 else -1
        
        values_dict = {
            # 'm': [self.a_stds_ * self.a_norms_ * self.a_losses_ * self.a_means],
            # 'n': [self.a_norms_],
            'g': [self.a_means],
            # 'l': [self.a_losses_],
            # 'c': [self.a_cs_],
            # 'd': [self.a_diffs_],
            's': [self.a_size_],
            't': [self.a_trust_],
        }
        values = [get_clean_selected_ratio(values_dict[key]) for key in values_dict.keys()]
        self.a_msg_that_i_need_to_print = '|'.join([colored(f'({key}-{values[k]:+.2f})', colors[int(values[k])+1]) for k, key in enumerate(values_dict.keys())])
        
        # self.a_msg_that_i_need_to_print += f'b_max: {np.max(self.clients_benign_probabilities[np.where(np.array(self.clients_keys)!='clean')]):+.2f}, '
        # self.a_msg_that_i_need_to_print += f'b_mean: {np.mean(self.clients_benign_probabilities[np.where(np.array(self.clients_keys)!='clean')]):+.2f}, '
        
        selected_clients = np.array(self.clients_keys)[self.active_clients]
        
        acc_ = np.array([np.mean(selected_clients[np.where((self.gamma_labels_==k))]=='clean') for k in self.sorted_labels])
        acc_ *= (1-acc_); prefix = ' | cl_acc'
        self.a_msg_that_i_need_to_print += colored(f'{prefix}: {1-np.mean(acc_):.2f}', 'red') if np.mean(acc_)>0 else colored(f'{prefix}: {1-np.mean(acc_):.2f}', 'white')
        
        # self.a_msg_that_i_need_to_print += f' | clean: {np.mean(self.hasnet_indicator[selected_clients=='clean']):+.2f}, '
        # self.a_msg_that_i_need_to_print += f'bad: {np.mean(self.hasnet_indicator[selected_clients!='clean']):+.2f}, '
        # self.a_msg_that_i_need_to_print += f'sel: {np.sum(self.hasnet_indicator==np.max(self.hasnet_indicator))}, '
        # self.a_msg_that_i_need_to_print += f'ds:{np.max(self.stds_)-np.min(self.stds_):+.4f}, '
        # self.a_msg_that_i_need_to_print += f'dn:{np.max(self.norms_)-np.min(self.norms_):+.4f}, '
        # self.a_msg_that_i_need_to_print += f'dg:{np.max(self.a_means)-np.min(self.a_means):+.4f}, '
        # self.a_msg_that_i_need_to_print += f'dg:{np.max(self.losses_)-np.min(self.losses_):+.4f}, '
        # self.a_msg_that_i_need_to_print += f'dc:{np.max(self.a_c_std)-np.min(self.a_c_std):+.4f}'
        
        return
    
    
    def stack_model(self, model_state: list[dict]):
        
        self.last_n_models = self.last_n_models[-self.n_stored_models+1:]
        self.last_n_models.append({k: v.cpu() for k, v in model_state.items()})
        
        return
    
    
    def aggregate(self, clients_state_dict, pre_str=''):
        
        return_model = self.aggregate_defended(clients_state_dict, pre_str=pre_str)
        
        # try:
        #     return_model = self.aggregate_defended(clients_state_dict, pre_str=pre_str)
        # except Exception as e:
        #     print('The exception that I encountered is:', e)
        #     return_model = self.model.model.state_dict()
        
        self.good_indicator = np.zeros_like(self.clients_delta).astype('float')
        if np.max(self.hasnet_indicator) > np.min(self.hasnet_indicator):
            self.good_indicator[np.where(self.hasnet_indicator==np.max(self.hasnet_indicator))] = 1
        
        return return_model # self.last_n_models[np.random.randint(0, len(self.last_n_models))]
    
    
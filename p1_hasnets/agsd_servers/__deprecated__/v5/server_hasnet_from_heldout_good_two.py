import numpy as np
import torch
import copy
import gc
from termcolor import colored

from sklearn.cluster import KMeans, HDBSCAN, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server

from ...agsd_adv_attack import HGSD_Adversarial_Attack



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
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        # self.norm = torch.norm(p=2, dim=1)
        self.hdb = HDBSCAN(
            # min_cluster_size=int(len(self.clients)*self.configuration['clients_ratio']/2)+1,
            min_cluster_size=2,
            min_samples=1,
            allow_single_cluster=True
        )
        self.spectral = SpectralClustering(n_clusters=2, affinity='precomputed')
        self.kmeans = KMeans(n_clusters=3, n_init='auto')
        self.clusterer = self.spectral
        
        # preparing hasnets variables
        self.diff_avg = 0
        self.rejected_mean, self.selected_mean = -1., 1.
        self.prob_selected, self.prob_rejected = 1., 0.
        self.clients_benign_probabilities = np.array([0.] * len(self.clients))
        self.current_epoch = 0
        
        self.model.model.to(torch.float)
        self.my_perturber = HGSD_Adversarial_Attack(self.model)
        
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
    
    
    def rescale_noisy_aggregate(self, clients_states: list[dict], indices: list[int]=None, strength: float=1e-5, ):
        
        if indices is None:
            indices = np.arange(len(clients_states))
        
        state_t = copy.deepcopy(self.model.model.state_dict())
        delta_state_t = copy.deepcopy(self.model.model.state_dict())
        for key in self.model.model.state_dict().keys():
            for i, ind in enumerate(indices):
                if i==0:
                    delta_state_t[key] = (clients_states[ind][key] - self.model.model.state_dict()[key]).clone() * self.scalers[ind] / len(indices)
                else:
                    delta_state_t[key] += (clients_states[ind][key] - self.model.model.state_dict()[key]).clone() * self.scalers[ind] / len(indices)
            
            if not ('bias' in key or 'bn' in key):
                standard_deviation = torch.std(delta_state_t[key].clone().view(-1), unbiased=False)
                try: state_t[key] += delta_state_t[key] + strength * torch.normal(0., standard_deviation, size=state_t[key].shape).to(state_t[key].device)
                except: pass

        return state_t
    
    
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
    
    
    def get_best_model(self, all_models: list[dict], **kwargs):
        
        def hl_std(arr_in: torch.Tensor):
            arr_in = arr_in.clone().detach().cpu().numpy()
            arr_ = np.argmax(arr_in, axis=1)
            z_ = np.zeros((len(arr_), len(self.data.get_class_names())))
            z_[np.arange(arr_.size), arr_] = 1
            assert len(np.std(z_, axis=0)) == z_.shape[1]
            return np.exp(np.mean(np.std(z_, axis=0))) / np.mean(np.abs(arr_in))
        
        local_model = Torch_Model(self.real_healing_data, model_configuration=self.model.model_configuration)
        stds_ = []
        for model_ in all_models:
            local_model.model.load_state_dict(model_)
            healing_data = self.my_perturber.attack(local_model, self.real_healing_data, epsilon=0.3, iterations=30)
            adv_y, _ = local_model.predict(torch.utils.data.DataLoader(healing_data.train, shuffle=False, batch_size=local_model.model_configuration['batch_size']))
            stds_.append( hl_std(adv_y) )
        self.stds_ = np.array(stds_)
        
        return self.sorted_labels[np.where(self.stds_==np.max(self.stds_))[0][0]], all_models[np.argmax(stds_)]
    
    
    def aggregate(self, clients_state_dict: list[dict], **kwargs):
        
        self.configuration['healing_epochs'] = 5
        
        initial_model = torch.stack([self.parameter_flatten_client_state_torch(self.model.model.state_dict())])
        flattened_clients_states = torch.stack([self.parameter_flatten_client_state_torch(cs) for cs in clients_state_dict], dim=0)
        self.compute_scalers(flattened_clients_states-initial_model)
        flattened_clients_states = initial_model.clone() + (flattened_clients_states.clone()-initial_model.clone())*self.scalers
        flattened_aggregated_classifier = torch.mean(flattened_clients_states, dim=0, keepdim=True)
        
        cosine_similarities = 1 + self.pairwise_cosine_similarity_torch(flattened_clients_states-flattened_aggregated_classifier).detach().cpu().numpy()
        cosine_similarities += 1 + self.pairwise_cosine_similarity_torch(flattened_clients_states-initial_model).detach().cpu().numpy()
        self.clusterer.fit(cosine_similarities)
        self.gamma_labels_ = self.clusterer.labels_
        self.sorted_labels = np.sort(np.unique(self.gamma_labels_))
        
        """
        There are a total of three clusters:
        1. f(x+tau) === f(x) is the clean cluster
        2. f(x+tau) =/= f(x) is the backdoored cluster
        3. Third cluster may be a mix of the first two.
        
        We know that when backdoored cluster will be healed, it will drastically move away from it.
        On the other hand when the clean cluster is healed, it does not move away from itself that drastically.
        """
        agg_cl = [self.rescale_noisy_aggregate(clients_state_dict, indices=np.where(self.gamma_labels_==k)[0]) for k in self.sorted_labels]
        flattened_agg_cl = [self.parameter_flatten_client_state_torch(agg_cl_) for agg_cl_ in agg_cl]
        best_std_label, _ = self.get_best_model(agg_cl); gc.collect()
        
        agg_cl_h = [self.heal_model_from_state(agg_cl_) for agg_cl_ in agg_cl]
        flattened_agg_cl_h = [self.parameter_flatten_client_state_torch(agg_cl_h_) for agg_cl_h_ in agg_cl_h]
        self.gamma = []
        for k1 in self.sorted_labels:
            gamma_ = 0
            for k2 in self.sorted_labels:
                if k1 == k2: pass
                else: gamma_ += 1 + self.cosine_similarity( (flattened_agg_cl[k1]-flattened_agg_cl[k2]).view(1, -1), (flattened_agg_cl_h[k2]-flattened_agg_cl[k2]).view(1, -1) ).item()    
            self.gamma.append(gamma_)
        self.gamma = np.array(self.gamma)
        self.actual_gamma = self.gamma.copy()
        self.compute_best_label()
        
        best_condition = np.where(np.multiply(self.gamma_labels_==self.best_gamma_label, self.gamma_labels_==best_std_label))
        worst_condition = np.where(np.multiply(self.gamma_labels_!=self.best_gamma_label, self.gamma_labels_!=best_std_label))
        self.hasnet_indicator = np.array([0.]*len(self.active_clients))
        # self.clients_benign_probabilities[self.active_clients[best_condition]] += np.clip(self.diff, 0.05, 0.3)
        # self.clients_benign_probabilities[self.active_clients[worst_condition]] -= np.clip(self.diff, 0.05, 0.3)
        # self.clients_benign_probabilities = np.clip(self.clients_benign_probabilities, -10., 10.)    
        self.hasnet_indicator[best_condition] = 1.
        self.hasnet_indicator[worst_condition] = -1
        self.clients_benign_probabilities[self.active_clients] += np.clip(self.diff, 0.05, 0.3) * self.hasnet_indicator
        self.hasnet_indicator[np.where(np.array([self.clients_benign_probabilities[c] for c in self.active_clients]) < 0.)] = 0.
        
        def get_clean_selected_ratio(arr_in: list[np.ndarray]):
            arr_ = arr_in[0].reshape(-1)
            selected_clients = np.array(self.clients_keys)[self.active_clients]
            clients_probabilities = self.clients_benign_probabilities[self.active_clients]
            if len(arr_in) == 1:
                selected_clients = selected_clients[np.where((arr_==np.max(arr_)) * (clients_probabilities>=0))]
            elif len(arr_in) == 2:
                arr_2 = arr_in[1].reshape(-1)
                selected_clients = selected_clients[np.where((arr_==np.max(arr_)) * (arr_2==np.max(arr_2)) * (clients_probabilities>=0))]
            return np.mean(selected_clients=='clean') if len(selected_clients)>0 else -1
        analysis_stds_ = np.array([self.stds_[np.where(self.sorted_labels==gl)] for gl in self.gamma_labels_]).reshape(-1)
        analysis_gammas = np.array([self.gamma[np.where(self.sorted_labels==gl)] for gl in self.gamma_labels_]).reshape(-1)
        colors = ['light_cyan', 'light_red', 'green']
        values = [get_clean_selected_ratio(arr_) for arr_ in [[analysis_stds_], [analysis_gammas], [analysis_stds_*analysis_gammas], [analysis_gammas+analysis_stds_], [analysis_gammas, analysis_stds_]]]
        self.a_msg_that_i_need_to_print = '|'.join([colored(f'({str_}-{values[s]:.2f})', colors[int(values[s])+1]) for s, str_ in enumerate(['s', 'g', 's*g', 's+g', 's,g'])])
        
        self.print_out('Clients selected.')
        if 1 in self.hasnet_indicator:
            return self.rescale_noisy_aggregate(clients_state_dict, indices=np.where(self.hasnet_indicator==1)[0])
        else:
            return self.model.model.state_dict()
    
    
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
        
        
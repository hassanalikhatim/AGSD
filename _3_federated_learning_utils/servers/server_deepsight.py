import numpy as np
import copy
import torch
from sklearn.cluster import DBSCAN


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _3_federated_learning_utils.servers.server import Server



class Server_Deepsight(Server):
    """
    DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection
    URL: https://www.ndss-symposium.org/wp-content/uploads/2022-156-paper.pdf
    
    @article{rieger2022deepsight,
        title={Deepsight: Mitigating backdoor attacks in federated learning through deep model inspection},
        author={Rieger, Phillip and Nguyen, Thien Duc and Miettinen, Markus and Sadeghi, Ahmad-Reza},
        journal={arXiv preprint arXiv:2201.00763},
        year={2022}
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
        
        return
    
    
    def ensemble_cluster(self, neups, ddifs, biases):
        biases = np.array([bias.cpu().numpy() for bias in biases])
        N = len(neups)
        cosine_labels = DBSCAN(min_samples=3,metric='cosine').fit(biases).labels_
        neup_labels = DBSCAN(min_samples=3).fit(neups).labels_
        # print("neup_cluster:{}".format(neup_labels))
        ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_
        # print("ddif_cluster:{}".format(ddif_labels))

        dists_from_cluster = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
                    neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j]))/3.0
                dists_from_cluster[j, i] = dists_from_cluster[i, j]
                
        # print("dists_from_clusters:")
        # print(dists_from_cluster)
        ensembled_labels = DBSCAN(min_samples=3,metric='precomputed').fit(dists_from_cluster).labels_

        return ensembled_labels
    
    
    def aggregate(
        self, clients_state_dict,
        **kwargs
    ):
        
        chosen_ids = self.active_clients
        state_t_minus_1 = self.model.model.state_dict()
        
        global_weight = list(self.model.model.state_dict().values())[-2]
        global_bias = list(self.model.model.state_dict().values())[-1]
        
        biases = [(list(clients_state_dict[i].values())[-1] - global_bias) for i, _ in enumerate(chosen_ids)]
        weights = [list(clients_state_dict[i].values())[-2] for i, _ in enumerate(chosen_ids)]
        
        n_client = len(chosen_ids)
        cosine_similarity_dists = np.array((n_client, n_client))
        neups = list()
        n_exceeds = list()

        # calculate neups
        sC_nn2 = 0
        for i in range(len(chosen_ids)):
            C_nn = torch.sum(weights[i]-global_weight, dim=[1]) + biases[i]-global_bias
            # print("C_nn:",C_nn)
            C_nn2 = C_nn * C_nn
            neups.append(C_nn2)
            sC_nn2 += C_nn2
            
            C_max = torch.max(C_nn2).item()
            threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else 1 / len(biases) * C_max
            n_exceed = torch.sum(C_nn2 > threshold).item()
            n_exceeds.append(n_exceed)
        # normalize
        neups = np.array([(neup/sC_nn2).cpu().numpy() for neup in neups])
        # print("n_exceeds:{}".format(n_exceeds))
        
        rand_input = torch.randn(
            [self.model.model_configuration['batch_size']]+list(self.data.train.__getitem__(0)[0].shape)
        ).to(self.model.device)
        # if isinstance(task, Cifar10FederatedTask):
        #     # 256 can be replaced with smaller value
        #     rand_input = torch.randn((256, 3, 32, 32)).to(self.model.device)
        # elif isinstance(task, TinyImagenetFederatedTask):
        #     # 256 can be replaced with smaller value
        #     rand_input = torch.randn((256, 3, 64, 64)).to(self.model.device)
        
        global_ddif = torch.mean(torch.softmax(self.model.model(rand_input), dim=1), dim=0)
        client_ddifs = []
        for i, _ in enumerate(chosen_ids):
            self.model.model.load_state_dict(clients_state_dict[i])
            client_ddifs.append(
                torch.mean(torch.softmax(self.model.model(rand_input), dim=1), dim=0) / global_ddif
            )
        client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])
        self.model.model.load_state_dict(state_t_minus_1)
        
        # use n_exceed to label
        classification_boundary = np.median(np.array(n_exceeds)) / 2
        
        identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
        # print("identified_mals:{}".format(identified_mals))
        clusters = self.ensemble_cluster(neups, client_ddifs, biases)
        # print("ensemble clusters:{}".format(clusters))
        cluster_ids = np.unique(clusters)

        deleted_cluster_ids = list()
        for cluster_id in cluster_ids:
            n_mal = 0
            cluster_size = np.sum(cluster_id == clusters)
            for identified_mal, cluster in zip(identified_mals, clusters):
                if cluster == cluster_id and identified_mal:
                    n_mal += 1
            # print("cluser size:{} n_mal:{}".format(cluster_size,n_mal))        
            if (n_mal / cluster_size) >= (1 / 3):
                deleted_cluster_ids.append(cluster_id)
        
        # temp_chosen_ids = copy.deepcopy(chosen_ids)
        # for i in range(len(chosen_ids)-1, -1, -1):
        #     if clusters[i] in deleted_cluster_ids:
        #         del chosen_ids[i]
        temp_chosen_ids = copy.deepcopy(chosen_ids)
        chosen_ids = []
        for i in range(len(temp_chosen_ids)-1, -1, -1):
            if clusters[i] not in deleted_cluster_ids:
                chosen_ids.append(clusters[i])
        
        if len(chosen_ids)==0:
            chosen_ids = temp_chosen_ids
        # print("final clients length:{}".format(len(chosen_ids)))
        
        w_avg = {}
        for key in state_t_minus_1.keys():
            for i, _ in enumerate(chosen_ids):
                if key not in w_avg.keys():
                    w_avg[key] = copy.deepcopy(clients_state_dict[i][key])
                else:
                    w_avg[key] += clients_state_dict[i][key]
            
            w_avg[key] = torch.div(w_avg[key], len(chosen_ids))
            
        self.good_indicator = -1. * np.ones((len(self.active_clients)))
        self.good_indicator[chosen_ids] = 1.
        
        return w_avg
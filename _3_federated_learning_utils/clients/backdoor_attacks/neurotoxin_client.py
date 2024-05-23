"""
    Neurotoxin: Durable Backdoors in Federated Learning
    URL: https://proceedings.mlr.press/v162/zhang22w/zhang22w.pdf
    @inproceedings{zhang2022neurotoxin,
        title={Neurotoxin: Durable backdoors in federated learning},
        author={Zhang, Zhengming and Panda, Ashwinee and Song, Linyue and Yang, Yaoqing and Mahoney, Michael and Mittal, Prateek and Kannan, Ramchandran and Gonzalez, Joseph},
        booktitle={International Conference on Machine Learning},
        pages={26429--26446},
        year={2022},
        organization={PMLR}
    }
"""



import numpy as np
import torch
import copy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from .simple_backdoor_client import Simple_Backdoor_Client



class Neurotoxin_Client(Simple_Backdoor_Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        super().__init__(
            data, global_model_configuration,
            client_configuration=client_configuration
        )
        
        self.clean_data = data
        
        self.client_type = 'neurotoxin_backdoor'
        self.last_epoch_weights = None
        
        default_configuration = {
            'mask_ratio': 0.02
        }
        for key in default_configuration.keys():
            if key not in self.local_model_configuration.keys():
                self.local_model_configuration[key] = default_configuration[key]
        
        return
    
    
    def weight_updates(self, global_model_state_dict: dict, verbose=True, **kwargs):
        
        if not self.last_epoch_weights:
            self.last_epoch_weights = copy.deepcopy(global_model_state_dict)
            return super().weight_updates(global_model_state_dict, verbose=verbose)
        
        else:
            self.differences = {}
            for key in self.last_epoch_weights.keys():
                self.differences[key] = global_model_state_dict[key] - self.last_epoch_weights[key]
            
            # calculate where the difference is smallest - these neurons will likely not be updated 
            # in the future by honest clients.
            flattened_differences = self.flatten_state(self.differences)
            self.threshold = np.sort( np.abs(flattened_differences) )[int(self.local_model_configuration['mask_ratio']*len(flattened_differences))]
            
            local_model = Torch_Model(data = self.data, model_configuration=self.local_model_configuration)
            local_model.model.load_state_dict(global_model_state_dict)
            train_loader, test_loader = local_model.data.prepare_data_loaders(batch_size=self.local_model_configuration['batch_size'])
            for epoch in range(1, self.local_model_configuration['local_epochs']+1):
                train_loss, train_acc = self.train_shot(local_model, train_loader, epoch, verbose=verbose)
            
            # poisoned_state_dict = copy.deepcopy(local_model.model.state_dict())
            # local_model.data = self.clean_data
            # local_model.model.load_state_dict(global_model_state_dict)
            # local_model.train(
            #     epochs=self.local_model_configuration['local_epochs'], 
            #     batch_size=self.local_model_configuration['batch_size'],
            #     verbose=verbose
            # )
            # clean_state_dict = local_model.model.state_dict()
            # final_state = {key: poisoned_state_dict[key] for key in poisoned_state_dict.keys()}
            
            self.last_epoch_weights = copy.deepcopy(global_model_state_dict)
            
            return local_model.model.state_dict()
    
    
    def train_shot(
        self, local_model: Torch_Model, train_loader, epoch,
        log_interval=None, verbose=True,
        **kwargs
    ):
        
        len_train_dataset = len(train_loader.dataset)
        
        local_model.model.train()
        
        loss_over_data = 0
        acc_over_data = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(local_model.device), target.to(local_model.device)
            
            local_model.optimizer.zero_grad()
            output = local_model.model(data)
            loss = local_model.loss_function(output, target)
            loss.backward()
            local_model.optimizer.step()
            
            updated_weights = local_model.model.state_dict()
            for key in updated_weights.keys():
                updated_weights[key] = torch.where(self.differences[key]<self.threshold, updated_weights[key], self.differences[key]+self.last_epoch_weights[key])
            local_model.model.load_state_dict(updated_weights)
            
            loss_over_data += loss.data
            pred = output.argmax(1, keepdim=True)
            acc_over_data += pred.eq(target.view_as(pred)).sum().item()
            
            if verbose:
                print(
                    '\r[Epoch: {} ({:3.1f}%)] | train_loss: {:.6f} | train_acc: {:.3f}% | '.format(
                        epoch, 100. * batch_idx / len(train_loader), 
                        loss_over_data/len_train_dataset, 100.*acc_over_data/len_train_dataset
                        ), 
                    end=''
                )
        
        return loss_over_data/len_train_dataset, acc_over_data/len_train_dataset
    
    
    def flatten_state(
        self, client_state_dict: dict
    ):
        
        # Flatten all the weights into a 1-D array
        flattened_client_state = []
        for key in client_state_dict.keys():
            flattened_client_state += client_state_dict[key].cpu().flatten().tolist()
        
        return np.array(flattened_client_state)
    
    
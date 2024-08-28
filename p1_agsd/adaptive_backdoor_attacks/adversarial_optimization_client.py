import numpy as np
import torch
import copy
import gc


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.adversarial_attacks.fgsm import FGSM

from _3_federated_learning_utils.clients.backdoor_attacks.simple_backdoor_client import Simple_Backdoor_Client



class Adversarial_Optimization_Backdoor_Client(Simple_Backdoor_Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        self.clean_data = data
        
        super().__init__(
            data, global_model_configuration,
            client_configuration=client_configuration
        )
        
        self.client_type = 'adversarial_optimization_client'
        
        default_client_configuration = {
            'threshold_scaler': 0.1
        }
        for key in default_client_configuration:
            if key not in self.local_model_configuration.keys():
                self.local_model_configuration[key] = default_client_configuration[key]
        
        return
    
    
    def weight_updates(self, global_model_state_dict: dict, verbose=True, **kwargs):
        
        local_model = Torch_Model(data=self.clean_data, model_configuration=self.local_model_configuration)
        local_model.model.load_state_dict(global_model_state_dict)
        local_model.train(
            epochs=self.local_model_configuration['local_epochs'], 
            batch_size=self.local_model_configuration['batch_size'],
            verbose=verbose
        )
        
        self.epsilon = torch.median(torch.abs(self.parameter_flatten_model_state(global_model_state_dict)-self.parameter_flatten_model_state(local_model.model.state_dict()))).item()
        self.epsilon *= self.local_model_configuration['threshold_scaler']
        
        local_model.data = self.data
        train_loader, test_loader = local_model.data.prepare_data_loaders(batch_size=self.local_model_configuration['batch_size'])
        for epoch in range(1, self.local_model_configuration['local_epochs']+1):
            train_loss, train_acc = self.train_shot(local_model, train_loader, epoch, verbose=verbose)
            gc.collect()
            
        return local_model.model.state_dict()
    
    
    def train_shot(
        self, local_model: Torch_Model, train_loader, epoch,
        log_interval=None, verbose=True,
        **kwargs
    ):
        
        current_global_state = local_model.model.state_dict()
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
                if ('weight' in key) or ('bias' in key):
                    difference = torch.clamp(updated_weights[key]-current_global_state[key], -self.epsilon, self.epsilon)
                    updated_weights[key] = current_global_state[key] + difference
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
    
    
    def parameter_flatten_model_state(self, client_state_dict: dict):

        flattened_client_state = []
        for key in client_state_dict.keys():
            if ('weight' in key) or ('bias' in key):
                flattened_client_state += [client_state_dict[key].clone().view(-1)]

        return torch.cat(flattened_client_state)
    

    
    
    
    
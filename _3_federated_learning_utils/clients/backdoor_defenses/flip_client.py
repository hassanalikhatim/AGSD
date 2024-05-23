import torch
import numpy as np


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.model_utils.torch_model import Torch_Model

from ..client import Client

from .flip_trigger_inversion import Flip_Trigger_Inversion



class Flip_Client(Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        super().__init__(
            data, global_model_configuration, client_configuration,
            client_type='flip'
        )
        
        self.classes = np.sort(np.unique(
            np.array([data.train.__getitem__(i)[1] for i in range(data.train.__len__())])
        ))
        
        self.distance_matrix = np.zeros( (len(self.classes), len(self.classes)) )
        self.variance_matrix = np.zeros( (len(self.classes), len(self.classes)) )
        
        self.promising_pair = None
        
        return
    
    
    def reset_client(self, data=None, client_configuration: dict={}):
        
        default_client_configuration = {
            'trigger_inversion_iterations': 1000
        }
        for key in default_client_configuration.keys():
            if key not in client_configuration.keys():
                client_configuration[key] = default_client_configuration[key]
        
        return super().reset_client(data, client_configuration)
    
    
    def trigger_inversion(self, data_in, data_out, local_model):
        
        x_input, x_output = data_in.detach().cpu().numpy(), data_out.detach().cpu().numpy()
        
        attack = Flip_Trigger_Inversion(local_model, self.local_model_configuration['loss_fn'])
        
        if not self.promising_pair:
            trigger_inversion_classes = self.classes
        else:
            trigger_inversion_classes = self.promising_pair#[i[0] for i in self.promising_pair]
        
        perturbations = np.zeros( [len(self.classes)] + list(x_input.shape[1:]) ).astype(np.float32)
        masks = np.zeros_like(perturbations)
        for t, target_class in enumerate(trigger_inversion_classes):
            perturbations[target_class], masks[target_class] = attack.attack(
                x_input, np.array([target_class]*len(data_in)), 
                iterations=self.local_model_configuration['trigger_inversion_iterations'],
                verbose=False
            )
            
            attack.last_run_loss_values = np.transpose(attack.last_run_loss_values)
            
            for s, source_class in enumerate(trigger_inversion_classes):
                source_class_indices = np.where(x_output == source_class)[0]
                # measure mean source to target distance for each source and save in the distances matrix
                self.distance_matrix[source_class, target_class] = np.mean( np.abs(masks[t]) )
                # measure mean source to target loss variance for each source and save in the variances matrix
                self.variance_matrix[source_class, target_class] = np.mean( np.std(attack.last_run_loss_values[source_class_indices]) )
        
        self.promising_pair = [i[0] for i in np.where(self.variance_matrix==np.max(self.variance_matrix))]
        
        self.perturbation = perturbations[self.promising_pair]
        self.mask = masks[self.promising_pair]
        
        data_01 = (1-self.mask[0])*x_input + self.mask[0]*self.perturbation[0]
        data_10 = (1-self.mask[1])*x_input + self.mask[1]*self.perturbation[1]
        
        return data_01, data_10
    
    
    def weight_updates(self, global_model_state_dict, verbose=True, **kwargs):
        
        local_model = Torch_Model(
            data = self.data,
            model_configuration=self.local_model_configuration
        )
        local_model.model.load_state_dict(global_model_state_dict)
        
        local_model = self.train_local_model(local_model, verbose=verbose)
        
        return local_model.model.state_dict()
    
    
    def test(self, model: Torch_Model):

        test_data = self.data.sample_data(self.data.test, batch_size=self.local_model_configuration['batch_size'])
        model.test_shot(test_data, verbose=True)
        
        return
    
    
    def train_local_model(self, local_model: Torch_Model, verbose=True):
        
        train_loader = torch.utils.data.DataLoader(self.data.train, shuffle=True, batch_size=self.local_model_configuration['batch_size'])
        test_loader = torch.utils.data.DataLoader(self.data.test, shuffle=True, batch_size=self.local_model_configuration['batch_size'])
        
        for epoch in range(1, self.local_model_configuration['epochs']+1):
            local_model, train_loss, train_acc, _ = self.train_shot(local_model, train_loader, epoch, verbose=verbose)
            test_loss, test_acc = local_model.test_shot(test_loader, verbose=verbose)
            
        return local_model
    
    
    def train_shot(self, local_model: Torch_Model, train_loader, epoch, verbose=True, pre_str='', **kwargs):
        
        local_model.model.train()
        
        loss_over_data = 0
        acc_over_data = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(local_model.device), target.to(local_model.device)
            
            data_01, data_10 = self.trigger_inversion(data, target, local_model)
            data_01 = torch.tensor(data_01).to(local_model.device)
            data_10 = torch.tensor(data_10).to(local_model.device)
            
            local_model.optimizer.zero_grad()
            output = local_model.model(data)
            loss = local_model.loss_function(output, target) 
            loss += local_model.loss_function(local_model.model(data_01), target)
            loss += local_model.loss_function(local_model.model(data_10), target)
            loss.backward()
            local_model.optimizer.step()
            
            loss_over_data += loss.item()
            pred = output.argmax(1, keepdim=True)
            acc_over_data += pred.eq(target.view_as(pred)).sum().item()
            
            if verbose:
                print_str = 'Epoch: {}({:3.1f}%] | tr_loss: {:.5f} | tr_acc: {:.2f}% | '.format(
                    epoch, 100. * batch_idx / len(train_loader), 
                    loss_over_data / min( (batch_idx+1) * train_loader.batch_size, len(train_loader.dataset) ), 
                    100. * acc_over_data / min( (batch_idx+1) * train_loader.batch_size, len(train_loader.dataset) )
                )
                print('\r' + pre_str + print_str, end='')
        
        local_model.model.eval()
        
        n_samples = min( len(train_loader)*train_loader.batch_size, len(train_loader.dataset) )
        return local_model, loss_over_data/n_samples, acc_over_data/n_samples, print_str
    
    
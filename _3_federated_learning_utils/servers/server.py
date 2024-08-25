import numpy as np
import copy
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from .server_client_plugin import Server_Client_Plugin



class Server(Server_Client_Plugin):
    
    def __init__(
        self, 
        data: Torch_Dataset,
        model: Torch_Model,
        clients_with_keys: dict={}, 
        configuration: dict={},
        verbose: bool=True,
        **kwargs
    ):
        
        super().__init__(clients_with_keys)
        
        self.model = model
        self.data = data
        
        self.configuration = {
            'clients_ratio': 0.1,
            'patience': self.model.model_configuration['epochs']
        }
        if configuration:
            for key in configuration.keys():
                self.configuration[key] = configuration[key]
        if 'patience' in self.model.model_configuration.keys():
            if self.model.model_configuration['patience'] is not None:
                self.configuration['patience'] = self.model.model_configuration['patience']
        
        self.clients_ratio = self.configuration['clients_ratio']
        
        self.good_indicator = np.ones(( int(self.clients_ratio*len(self.clients)) ))
        self.best_test_acc = 0.; self.still_patient = True
        self.saved_flattened_model = None
        
        self.verbose = verbose; self.show_msg = ''; self.a_msg_that_i_need_to_print = ''
        self.time_out = 0
        
        return
    
    
    def print_out(self, print_statement: str, overwrite=True):
        
        if self.verbose:
            if overwrite:
                print('\r' + ' ' * len(self.show_msg) + '\r' + print_statement, end='')
            else:
                print('\n' + print_statement, end='')
            self.show_msg = print_statement
        
        return
    
    
    def get_updated_models_from_clients(self, pre_str='', **kwargs):
        
        clients_state_dict = []
        for i, idx in enumerate(self.active_clients):
            this_str = 'Client {}/{}'.format(i+1, len(self.active_clients))
            self.print_out(pre_str + this_str, overwrite=True)
            
            clients_state_dict.append(
                self.clients[idx].weight_updates(self.model.model.state_dict(), verbose=False)
            )
        
        return clients_state_dict
    
    
    def run(self, num_rounds=1):
        
        for round in range(num_rounds):
            self.shot(round=round)
        
        return
    
    
    def shot(self, round=-1, pre_str=''):
        
        if self.still_patient:
            pre_str += 'Round {}, '.format(round)
            
            self.sample_clients()
            clients_state_dict = self.get_updated_models_from_clients(pre_str=pre_str)
                
            assert len(clients_state_dict) == len(self.active_clients)
            
            try: 
                aggregated_weights = self.aggregate(clients_state_dict, pre_str=pre_str)
            except Exception as e: 
                print(f'\n\nThis should not happen often. The exception is {e}')
                aggregated_weights = self.model.model.state_dict()
            self.model.model.load_state_dict(aggregated_weights)
            
            self.update_patience()
            
        else:
            self.model.model.load_state_dict(self.key_unflatten_client_state_np(self.saved_flattened_model))
        
        return
    
    
    def aggregate(self, clients_state_dict, pre_str=''):
        return self.super_aggregate(clients_state_dict, pre_str=pre_str)
    
    
    def super_aggregate(
        self, clients_state_dict,
        pre_str=''
    ):
        
        w_avg = copy.deepcopy(clients_state_dict[0])
        for key in w_avg.keys():
            for i in range(1, len(clients_state_dict)):
                w_avg[key] += clients_state_dict[i][key]
            
            w_avg[key] = torch.div(w_avg[key], len(clients_state_dict))
            
        return w_avg
    
    
    def update_patience(self):
        
        train_loader = torch.utils.data.DataLoader(self.data.train, batch_size=self.model.model_configuration['batch_size'])
        self.train_loss, self.train_acc = self.model.test_shot(train_loader, verbose=False)
        
        test_loader = torch.utils.data.DataLoader(self.data.test, batch_size=self.model.model_configuration['batch_size'])
        self.test_loss, self.test_acc = self.model.test_shot(test_loader, verbose=False)
        
        # know when to stop training based on patience
        if self.best_test_acc == 0 or self.test_acc > self.best_test_acc:
            self.patience_variable = 0
            
            self.best_train_loss = self.train_loss
            self.best_train_acc = self.train_acc
            self.best_test_loss = self.test_loss
            self.best_test_acc = self.test_acc
            
            self.saved_flattened_model = self.key_flatten_client_state_np(self.model.model.state_dict())
        else:
            self.patience_variable += 1
            if self.patience_variable > self.configuration['patience']:
                self.still_patient = False
                
                self.train_loss = self.best_train_loss
                self.train_acc = self.best_train_acc
                self.test_loss = self.best_test_loss
                self.test_acc = self.best_test_acc
        
        return
    
    
    def evaluate_server_statistics(self):
        
        signs = {key: [0] for key in self.client_with_keys.keys()}
        for i, ac in enumerate(self.active_clients):
            signs[self.clients_keys[ac]].append(np.mean(self.good_indicator[i] > 0))
            
        for key in signs.keys():
            if len(signs[key]) > 1:
                signs[key] = signs[key][1:]
        
        return {
            'train_loss': self.train_loss, 
            'train_acc': self.train_acc, 
            'test_loss': self.test_loss, 
            'test_acc': self.test_acc,
             **{key+'_acc_ratio': np.mean(signs[key]) for key in signs.keys()}
        }
        
        
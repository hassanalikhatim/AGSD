"""
    IBA: Towards Irreversible Backdoor Attacks in Federated Learning
    URL: https://openreview.net/pdf?id=cemEOP8YoC
    @inproceedings{nguyen2023iba,
        title={IBA: Towards Irreversible Backdoor Attacks in Federated Learning},
        author={Nguyen, Dung Thuy and Nguyen, Tuan Minh and Tran, Anh Tuan and Doan, Khoa D and WONG, KOK SENG},
        booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
        year={2023}
    }
"""

import numpy as np
import torch


from ..client import Client

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.poisonable_class import Poisonable_Data

from .iba_client_backdoor import Irreversible_Backdoor, Simple_Backdoor



class Irreversible_Backdoor_Client(Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        super().__init__(
            data, global_model_configuration,
            client_configuration=client_configuration,
            client_type='irreversible_backdoor'
        )
        
        self.data = Irreversible_Backdoor(self.data, backdoor_configuration=client_configuration)
        
        return
    
    
    def weight_updates(self, global_model_state_dict, verbose=True, **kwargs):
        
        local_model = Torch_Model(
            data = self.data,
            model_configuration=self.local_model_configuration
        )
        local_model.model.load_state_dict(global_model_state_dict)
        
        self.data.reset_trigger()
        self.data.compute_optimal_trigger(local_model)
        
        local_model.train(
            epochs=self.local_model_configuration['local_epochs'], 
            batch_size=self.local_model_configuration['batch_size'],
            verbose=verbose
        )
        
        weights = local_model.model.state_dict()
        
        # allow weight scaling in the updates
        if 'scale' in self.local_model_configuration.keys():
            if self.local_model_configuration['scale']:
                for key in weights.keys():
                    gradients = weights[key] - global_model_state_dict[key]
                    weights[key] = global_model_state_dict[key] + self.local_model_configuration['scale']*gradients
        
        return weights
    
    
    def _deprecated_test_server_model(self, poisoned_data: Simple_Backdoor, model: Torch_Model):
        '''
        This function is just for analysis and results.
        '''
        
        batch_size = model.model_configuration['batch_size']
        
        self.data.reset_trigger()
        self.data.compute_optimal_trigger(model)
        
        poisoned_data.trigger = self.data.trigger
        test_data_loader = torch.utils.data.DataLoader(poisoned_data.poisoned_test, batch_size=batch_size)
        b_loss, b_acc = model.test_shot(test_data_loader, verbose=False)
        
        return {'poisoned_loss': b_loss, 'poisoned_acc': b_acc}
    
    
    def _deprecated_test_server_model(self, poisoned_data: Simple_Backdoor, model: Torch_Model):
        '''
        This function is just for analysis and results.
        '''
        
        batch_size = model.model_configuration['batch_size']
        poisoned_data.trigger = self.data.visible_trigger
        
        test_data_loader = torch.utils.data.DataLoader(poisoned_data.test, batch_size=batch_size, shuffle=False)
        outputs, ground_truths = model.predict(test_data_loader)
        outputs = outputs.argmax(1, keepdims=True).detach().cpu().numpy().reshape(-1)
        ground_truths = ground_truths.detach().cpu().numpy().reshape(-1)
        indices = np.array([i for i in np.where((outputs==ground_truths) & (ground_truths!=poisoned_data.target))[0]]).reshape(-1)
        
        b_loss = 0; b_acc = 0
        filtered_poisoned_data = Client_SubDataset(poisoned_data.poisoned_test, indices=indices)
        if filtered_poisoned_data.__len__() > 0:
            test_data_loader = torch.utils.data.DataLoader(filtered_poisoned_data, batch_size=batch_size)
            b_loss, b_acc = model.test_shot(test_data_loader, verbose=False)
        
        return {'poisoned_loss': b_loss, 'poisoned_acc': b_acc}
    
    
    def test_server_model(self, input_data: Simple_Backdoor, model: Torch_Model):
        '''
        This function is just for analysis and results.
        '''
        
        batch_size = model.model_configuration['batch_size']
        
        test_data_loader = torch.utils.data.DataLoader(input_data.test, batch_size=batch_size, shuffle=False)
        outputs, ground_truths = model.predict(test_data_loader)
        outputs = outputs.argmax(1, keepdims=True).detach().cpu().numpy().reshape(-1)
        ground_truths = ground_truths.detach().cpu().numpy().reshape(-1)
        
        
        def poison(x, y, **kwargs):
            return torch.clamp(x+self.trigger_, 0., 1.), self.target_
        
        poisoned_data = Poisonable_Data(input_data.test)
        poisoned_data.poisoner_fn = poison
        poisoned_data.poison_indices = np.arange(poisoned_data.__len__())
        
        b_loss = []; b_acc = []
        for t, (trigger, target) in enumerate(zip(self.data.visible_triggers, self.data.targets)):
            self.trigger_ = trigger; self.target_ = target
            
            indices = np.array([i for i in np.where((outputs==ground_truths) & (ground_truths!=target))[0]]).reshape(-1)
            filtered_poisoned_data = Client_SubDataset(poisoned_data, indices=indices)
            _b_loss = 0; _b_acc = 0
            if filtered_poisoned_data.__len__() > 0:
                test_data_loader = torch.utils.data.DataLoader(filtered_poisoned_data, batch_size=batch_size)
                _b_loss, _b_acc = model.test_shot(test_data_loader, verbose=False)
            b_loss.append(_b_loss); b_acc.append(_b_acc)
        
        return {'poisoned_loss': np.mean(b_loss), 'poisoned_acc': np.mean(b_acc)}
    
    
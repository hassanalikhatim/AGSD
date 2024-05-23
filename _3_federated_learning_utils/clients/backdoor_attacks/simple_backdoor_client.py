import numpy as np
import torch


from ..client import Client

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor
from _1_adversarial_ML.backdoor_attacks.poisonable_class import Poisonable_Data



class Simple_Backdoor_Client(Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        super().__init__(
            data, global_model_configuration,
            client_configuration=client_configuration,
            client_type='simple_backdoor'
        )
        
        self.data = Simple_Backdoor(data, backdoor_configuration=client_configuration)
        
        return
    
    
    def _deprecated_reset_client(self, data=None, client_configuration: dict={}):
        
        if data:
            self.data = Simple_Backdoor(
                data, backdoor_configuration=client_configuration
            )
            
        self.local_model_configuration = {
            'local_epochs': 1
        }
        for key in self.global_model_configuration.keys():
            self.local_model_configuration[key] = self.global_model_configuration[key]
        if client_configuration:
            for key in client_configuration.keys():
                self.local_model_configuration[key] = client_configuration[key]
        
        return
    
    
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
        for t, (trigger, target) in enumerate(zip(self.data.triggers, self.data.targets)):
            self.trigger_ = trigger; self.target_ = target
            
            indices = np.array([i for i in np.where((outputs==ground_truths) & (ground_truths!=target))[0]]).reshape(-1)
            filtered_poisoned_data = Client_SubDataset(poisoned_data, indices=indices)
            _b_loss = 0; _b_acc = 0
            if filtered_poisoned_data.__len__() > 0:
                test_data_loader = torch.utils.data.DataLoader(filtered_poisoned_data, batch_size=batch_size)
                _b_loss, _b_acc = model.test_shot(test_data_loader, verbose=False)
            b_loss.append(_b_loss); b_acc.append(_b_acc)
        
        return {'poisoned_loss': np.mean(b_loss), 'poisoned_acc': np.mean(b_acc)}
    
    
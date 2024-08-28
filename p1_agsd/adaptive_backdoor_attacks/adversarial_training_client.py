import numpy as np
import torch
import copy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.adversarial_attacks.fgsm import FGSM

from _3_federated_learning_utils.clients.backdoor_attacks.simple_backdoor_client import Simple_Backdoor_Client



class Adversarial_Training_Backdoor_Client(Simple_Backdoor_Client):
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        super().__init__(data, global_model_configuration, client_configuration=client_configuration)
        
        self.client_type = 'adversarial_training_client'
        default_client_configuration = {
            'adv_epsilon': 0.2,
            'adv_iterations': 100
        }
        for key in default_client_configuration:
            if key not in self.local_model_configuration.keys():
                self.local_model_configuration[key] = default_client_configuration[key]
        
        return
    
    
    def adversarial_attack(self, model: Torch_Model, data: Torch_Dataset, epsilon=0.2, iterations=500) -> Torch_Dataset:
        
        x, y = [], []
        for i in range(data.train.__len__()):
            _x, _y = data.train.__getitem__(i)
            x.append(_x.detach().cpu().numpy()); y.append(_y)
        x = np.array(x); y = np.array(y)
        
        delta = np.max(x)-np.min(x)
        attacker = FGSM(model)
        perturbed_x_inputs = np.random.uniform(-0.05*delta, 0.05*delta, size=x.shape).astype(np.float32)
        adv_x = attacker.attack(perturbed_x_inputs, y, epsilon=epsilon*delta, targeted=False, iterations=iterations)
        
        adversarial_data = Torch_Dataset(data_name=data.data_name)
        adversarial_data.train = torch.utils.data.TensorDataset(torch.tensor(adv_x).float(), torch.tensor(y))
        adversarial_data.test = data.test
        
        return adversarial_data
    
    
    def weight_updates(self, global_model_state_dict, verbose=True, **kwargs) -> dict:
        
        local_model = Torch_Model(data=self.data, model_configuration=self.local_model_configuration)
        local_model.model.load_state_dict(copy.deepcopy(global_model_state_dict))
        
        local_model.data = self.adversarial_attack(
            local_model, self.data, 
            epsilon=self.local_model_configuration['adv_epsilon'], 
            iterations=self.local_model_configuration['adv_iterations']
        )
        
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
    
    
    
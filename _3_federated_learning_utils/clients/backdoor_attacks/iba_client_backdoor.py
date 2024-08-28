import numpy as np
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor
from _1_adversarial_ML.adversarial_attacks.universal_adversarial_perturbation import Universal_Adversarial_Perturbation



class Irreversible_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration: dict={}, **kwargs
    ):
        
        super().__init__(
            data, 
            backdoor_configuration=backdoor_configuration
        )
        
        return
    
    
    def configure_backdoor(
        self, backdoor_configuration: dict={},
    ):
        
        self.backdoor_configuration = {
            'poison_ratio': 0.2,
            'target': 0,
            'trigger_inversion_iterations': 200
        }
        for key in backdoor_configuration.keys():
            self.backdoor_configuration[key] = backdoor_configuration[key]
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        
        self.reset_trigger()
        self.targets = [self.backdoor_configuration['target']]
        
        self.visible_trigger = torch.zeros_like(self.train.__getitem__(0)[0])
        self.visible_trigger[0, :5, :5] = 1.
        self.visible_triggers = [self.visible_trigger]
        
        return
    
    
    def reset_trigger(self):
        self.triggers = [torch.zeros_like(self.train.__getitem__(0)[0])]
        return
    
    
    def compute_optimal_trigger(self, local_model: Torch_Model):
        
        x_input, random_indices = [], np.random.choice(self.train.__len__(), size=512)
        for k in random_indices:
            x_input.append(self.train.__getitem__(k)[0])
        x_input = torch.stack(x_input, dim=0).detach().cpu().numpy()
        
        attack = Universal_Adversarial_Perturbation(local_model, local_model.model_configuration['loss_fn'])
        
        np_trigger = attack.attack(
            x_input, np.array([self.targets[0]]*len(x_input)), 
            iterations=self.backdoor_configuration['trigger_inversion_iterations'],
            verbose=False
        )
        self.triggers = [torch.clamp( torch.tensor(np_trigger[0])+self.visible_trigger, np.min(x_input), np.max(x_input))]
        
        return
    
    
    
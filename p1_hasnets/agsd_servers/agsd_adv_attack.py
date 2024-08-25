import numpy as np
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.adversarial_attacks.adversarial_attack import Adversarial_Attack
from _1_adversarial_ML.adversarial_attacks.fgsm import FGSM
from _1_adversarial_ML.adversarial_attacks.ifgsm import i_FGSM

from .agsd_attack import AGSD



class AGSD_Adversarial_Attack:
    
    def __init__(
        self, 
        model: Torch_Model,
        loss='crossentropy',
        input_mask=None, output_mask=None
    ):
        
        # super().__init__(model, loss, input_mask, output_mask)
        
        # self.num_classes = model.data.get_output_shape()[-1]
        # self.model_configuration = model.model_configuration
        
        return
    
    
    def attack(self, model: Torch_Model, data: Torch_Dataset, epsilon=0.2, iterations=500):
        
        x, y = [], []
        for i in range(data.train.__len__()):
            _x, _y = data.train.__getitem__(i)
            x.append(_x.detach().cpu().numpy()); y.append(_y)
        x = np.array(x); y = np.array(y)
        
        adv_x = self.free_attack(model, x, targets=y, epsilon=epsilon, iterations=iterations)
        
        healing_data_adv = Torch_Dataset(data_name=data.data_name)
        healing_data_adv.train = torch.utils.data.TensorDataset(torch.tensor(adv_x).float(), torch.tensor(y))
        healing_data_adv.test = data.test
        
        return healing_data_adv
    
    
    def bla_attack(self, model: Torch_Model, x_inputs: np.ndarray, targets=None, epsilon=0.2, iterations=500):
        
        attacker = i_FGSM(model)
        perturbed_x_inputs = np.random.normal(scale=1e-2*np.std(x_inputs), size=x_inputs.shape).astype(np.float32)
        adversarial_x = attacker.attack(perturbed_x_inputs, targets, epsilon=epsilon, targeted=False, iterations=10)
        
        return adversarial_x
    
    
    def free_attack(self, model: Torch_Model, x_inputs: np.ndarray, targets=None, epsilon=0.2, iterations=500):
        
        delta = np.max(x_inputs)-np.min(x_inputs)
        
        attacker = AGSD(model)
        perturbed_x_inputs = np.random.uniform(-0.05*delta, 0.05*delta, size=x_inputs.shape).astype(np.float32)
        adversarial_x = attacker.attack(perturbed_x_inputs, targets, epsilon=epsilon*delta, targeted=False, iterations=iterations)
        # adversarial_x = np.clip(adversarial_x, np.min(x_inputs), np.max(x_inputs))
        
        return adversarial_x
    
    
    def untargeted_attack(self, model: Torch_Model, x_inputs: np.ndarray, targets=None, epsilon=0.2, iterations=500):
        
        delta = np.max(x_inputs)-np.min(x_inputs)
        
        attacker = FGSM(model)
        perturbed_x_inputs = np.random.uniform(-0.05*delta, 0.05*delta, size=x_inputs.shape).astype(np.float32)
        adversarial_x = attacker.attack(perturbed_x_inputs, targets, epsilon=epsilon*delta, targeted=False, iterations=iterations)
        
        return adversarial_x
    
    
    def targeted_attack(self, model: Torch_Model, x_inputs: np.ndarray, targets=None, epsilon=0.2, iterations=500):
        
        delta = np.max(x_inputs)-np.min(x_inputs)
        
        attacker = FGSM(model)
        adversarial_x = attacker.attack(x_inputs, targets, epsilon=epsilon*delta, targeted=True, iterations=iterations)
        
        return adversarial_x
    
    
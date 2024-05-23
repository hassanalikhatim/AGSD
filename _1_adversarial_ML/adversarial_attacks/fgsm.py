import numpy as np
import torch


from _0_general_ML.model_utils.torch_model import Torch_Model

from .adversarial_attack import Adversarial_Attack



class FGSM(Adversarial_Attack):
    
    def __init__(self, model: Torch_Model, loss='crossentropy', input_mask=None, output_mask=None):
        super().__init__(model, loss=loss, input_mask=input_mask, output_mask=output_mask)
        return
    
    
    def attack(self, x_input, y_input, epsilon=0.1, targeted=False, **kwargs):
        
        self.last_run_loss_values = []
        x_perturbation = np.zeros_like(x_input).astype(np.float32)
        x_perturbation = self.fgsm_step(x_input, y_input, x_perturbation, epsilon=epsilon, targeted=targeted)
        
        return np.clip(x_input + x_perturbation, np.min(x_input), np.max(x_input))
    
    
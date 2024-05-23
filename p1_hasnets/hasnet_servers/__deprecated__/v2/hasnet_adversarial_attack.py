import numpy as np
import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.adversarial_attacks.adversarial_attack import Adversarial_Attack



class HaSNet_Adversarial_Attack(Adversarial_Attack):
    
    def __init__(
        self, 
        model: Torch_Model,
        loss='crossentropy', 
        unflatten_client_state = None,
        input_mask=None, output_mask=None
    ):
        
        super().__init__(model, loss, input_mask, output_mask)
        
        self.num_classes = model.data.get_output_shape()[-1]
        self.model_configuration = model.model_configuration
        
        if unflatten_client_state:
            self.unflatten_client_state = unflatten_client_state
        
        return
    
    
    def attack(
        self,
        x_inputs, clients_state_dict: list[dict], 
        targets=None,
        epsilon=1, iterations=500
    ):
        
        epsilon_per_iteration = 2 * epsilon / iterations
        
        if not targets:
            targets = np.random.randint(0, self.num_classes, size=len(x_inputs)).astype(np.uint8)
        perturbation = np.random.uniform(-0.1, 0.1, size=x_inputs.shape).astype(np.float32)
        for i in range(iterations):
            print('\rIteration: {}'.format(i), end='')
            delta_perturbation = np.zeros_like(perturbation)
            for client_state_dict in clients_state_dict:
                # define model using clients state dict
                self.model.load_state_dict(self.unflatten_client_state(client_state_dict))
                
                updated_perturbation = self.batch_wise_fgsm_step(
                    x_inputs, targets, 
                    perturbation, 
                    epsilon=epsilon_per_iteration, 
                    targeted=True
                )
                delta_perturbation += (updated_perturbation - perturbation) / len(clients_state_dict)    
            perturbation = np.clip(delta_perturbation, -epsilon, epsilon)
            perturbation = np.clip(x_inputs + perturbation, 0, 1) - x_inputs
        
        return np.clip(x_inputs+perturbation, 0, 1), targets
    
    
    def adversarial_attack(
        self, 
        x_inputs, y_inputs,
        local_model: Torch_Model, clients_state_dict: list[dict], 
        epsilon=0.2, iterations=500
    ):
        
        epsilon_per_iteration = 2 * epsilon / iterations
        
        targets = y_inputs.astype(np.uint8)
        perturbation = np.random.uniform(-0.1, 0.1, size=x_inputs.shape).astype(np.float32)
        for i in range(iterations):
            print('\rAdv Iteration: {}'.format(i), end='')
            delta_perturbation = np.zeros_like(perturbation)
            for client_state_dict in clients_state_dict:
                # define model using clients state dict
                self.model.load_state_dict(self.unflatten_client_state(client_state_dict))
                
                updated_perturbation = self.fgsm_step(
                    x_inputs, targets, 
                    perturbation, 
                    epsilon=epsilon_per_iteration, 
                    targeted=False
                )
                delta_perturbation += (updated_perturbation - perturbation) / len(clients_state_dict)    
            perturbation = np.clip(delta_perturbation, -epsilon, epsilon)
            perturbation = np.clip(x_inputs+perturbation, 0, 1) - x_inputs
        
        return np.clip(x_inputs+perturbation, 0, 1), targets
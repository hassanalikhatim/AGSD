import numpy as np
import torch


from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.adversarial_attacks.adversarial_attack import Adversarial_Attack

from _0_general_ML.model_utils.optimizer_utils.torch_optimizer import Torch_Optimizer



class Trigger_Inversion(Adversarial_Attack):
    
    def __init__(self, model: Torch_Model, loss='crossentropy', input_mask=None, output_mask=None):
        super().__init__(model, loss=loss, input_mask=input_mask, output_mask=output_mask)
        
        self.optimizer = Torch_Optimizer(
            name='sgd', lr=1e-3, momentum=0.5
        )
        
        return
    
    
    def attack(
        self, x_input, y_input,
        iterations=1000,
        **kwargs
    ):
        
        self.last_run_loss_values = []
        
        x_perturbation = np.random.standard_normal(
            size=[1]+list(x_input.shape[1:])
        ).astype(np.float32)
        mask = np.zeros(
            ([1]+[1]+list(x_input.shape[2:]))
        ).astype(np.float32)
        for iteration in range(iterations):
            x_perturbation, mask, loss = self.step(
                x_input, y_input, x_perturbation, mask
            )
            
            self.last_run_loss_values.append(loss)
            
            print('\rIteration: {:3d}/{:3d}, Loss: {}'.format(iteration, iterations, loss), end='')
            
        return x_perturbation, mask
    
    
    def sigmoid(self, x_in):
        return 1 / ( 1 + torch.exp(-x_in) )
    
    
    def step(
        self, x_input, y_input, 
        x_perturbation, mask, epsilon=0.05
    ):
        
        x_v = torch.tensor(x_input.astype(np.float32)).to(self.device)
        y_in = torch.tensor(y_input).to(self.device)
        
        x_delta = torch.autograd.Variable(torch.tensor(x_perturbation)).to(self.device)
        x_mask = torch.autograd.Variable(torch.tensor(mask)).to(self.device)
        x_delta.requires_grad=True; x_mask.requires_grad=True
        
        threshold_lower, threshold_upper = 0., 1.
        while (threshold_upper - threshold_lower) > 0.02:
            self.threshold = (threshold_lower + threshold_upper)/2
            
            sigmoided_mask = torch.clamp(self.sigmoid(x_mask-self.threshold) - 0.5, 0, 1).sign()
            
            if torch.mean(sigmoided_mask) <= 0.03:
                # if torch.mean(sigmoided_mask) approaches 0, threshold is too large, 
                threshold_upper = self.threshold
            else:
                # if torch.mean(sigmoided_mask) approaches 1, threshold is too small
                threshold_lower = self.threshold
        
        thresholded_mask = self.sigmoid(x_mask - self.threshold)
        
        loss = self.adv_loss_outputs(
            self.model(
                torch.clamp(
                    (1-thresholded_mask)*x_v + thresholded_mask*self.sigmoid(x_delta),
                    0, 1
                )
            ), 
            y_in
        )
        loss += torch.mean(self.sigmoid(x_mask))
        loss.backward()
        
        # optimizer.step()
        x_perturbation -= epsilon * x_delta.grad.data.sign().detach().cpu().numpy()
        mask -= epsilon * x_mask.grad.data.sign().detach().cpu().numpy()
        
        return x_perturbation, mask, loss.detach().cpu().numpy()
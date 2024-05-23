import numpy as np
import torch


from _0_general_ML.model_utils.torch_model import Torch_Model

from _0_general_ML.model_utils.optimizer_utils.torch_optimizer import Torch_Optimizer

from .adversarial_attack import Adversarial_Attack



class Trigger_Inversion(Adversarial_Attack):
    
    def __init__(self, model: Torch_Model, loss='crossentropy', input_mask=None, output_mask=None):
        super().__init__(model, loss=loss, input_mask=input_mask, output_mask=output_mask)
        
        self.optimizer = Torch_Optimizer(name='sgd', lr=1e-3, momentum=0.5)
        
        return
    
    
    def attack(self, x_input, y_input, iterations=1000, verbose=True, **kwargs):
        
        self.last_run_loss_values = []
        
        x_perturbation = np.random.standard_normal( size=[1]+list(x_input.shape[1:]) ).astype(np.float32)
        mask = np.zeros( ([1]+[1]+list(x_input.shape[2:])) ).astype(np.float32)
        for iteration in range(iterations):
            x_perturbation, mask = self.step(x_input, y_input, x_perturbation, mask)
            
            if verbose:
                print(
                    '\rIteration: {:3d}/{:3d}, Loss: {}'
                    ''.format(iteration, iterations, np.mean(self.last_run_loss_values[-1:])), 
                    end=''
                )

        self.last_run_loss_values = np.array(self.last_run_loss_values)
            
        return x_perturbation, mask
    
    
    def step(self, x_input, y_input, x_perturbation, mask, epsilon=0.05):
        
        def sigmoid(x_in):
            return 1 / ( 1 + torch.exp(-x_in) )
        
        x_v = torch.tensor(x_input.astype(np.float32)).to(self.device)
        y_in = torch.tensor(y_input).to(self.device)
        
        x_delta = torch.autograd.Variable(torch.tensor(x_perturbation)).to(self.device)
        x_mask = torch.autograd.Variable(torch.tensor(mask)).to(self.device)
        x_delta.requires_grad=True; x_mask.requires_grad=True
        
        sigmoided_mask = sigmoid(x_mask)
        loss = self.adv_loss_outputs(
            self.model(torch.clamp( (1-sigmoided_mask)*x_v + sigmoided_mask*torch.clamp(x_delta, 0, 1), 0, 1 )), 
            y_in
        )
        loss += torch.mean(sigmoided_mask)
        torch.mean(loss).backward()
        
        x_perturbation -= epsilon * x_delta.grad.data.sign().detach().cpu().numpy()
        mask -= epsilon * x_mask.grad.data.sign().detach().cpu().numpy()
        
        self.last_run_loss_values.append(torch.mean(loss).item())
        
        return x_perturbation, mask
    
import numpy as np
import torch


from .simple_backdoor_client import Simple_Backdoor_Client

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _1_adversarial_ML.backdoor_attacks.multitrigger_multitarget_backdoor_attack import MultiTrigger_MultiTarget_Backdoor



class MultiTrigger_MultiTarget_Backdoor_Client(Simple_Backdoor_Client):
    """
    Code for:
    
    Multi-Trigger Backdoor Attacks: More Triggers, More Threats
    URL: https://arxiv.org/pdf/2401.15295v1
    
    @article{li2024multi,
        title={Multi-Trigger Backdoor Attacks: More Triggers, More Threats},
        author={Li, Yige and Ma, Xingjun and He, Jiabo and Huang, Hanxun and Jiang, Yu-Gang},
        journal={arXiv preprint arXiv:2401.15295},
        year={2024}
    }
    """
    
    def __init__(
        self, data: Torch_Dataset, 
        global_model_configuration: dict,
        client_configuration: dict={}
    ):
        
        super().__init__(
            data, global_model_configuration,
            client_configuration=client_configuration
        )
        
        self.client_type = 'multitrigger_multitarget_backdoor'
        self.data = MultiTrigger_MultiTarget_Backdoor(data, backdoor_configuration=client_configuration)
        
        return
    
    
    
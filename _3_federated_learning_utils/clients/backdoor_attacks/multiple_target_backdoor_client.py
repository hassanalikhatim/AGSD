import numpy as np
import torch


from .simple_backdoor_client import Simple_Backdoor_Client

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _1_adversarial_ML.backdoor_attacks.multiple_target_backdoor_attack import Multiple_Target_Backdoor



class Multiple_Target_Backdoor_Client(Simple_Backdoor_Client):
    """
    Code for:
    
    Dynamic Backdoor Attacks Against Machine Learning Models
    URL: https://arxiv.org/pdf/2003.03675
    
    @inproceedings{salem2022dynamic,
        title={Dynamic backdoor attacks against machine learning models},
        author={Salem, Ahmed and Wen, Rui and Backes, Michael and Ma, Shiqing and Zhang, Yang},
        booktitle={2022 IEEE 7th European Symposium on Security and Privacy (EuroS\&P)},
        pages={703--718},
        year={2022},
        organization={IEEE}
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
        
        self.client_type = 'multiple_target_backdoor'
        self.data = Multiple_Target_Backdoor(data, backdoor_configuration=client_configuration)
        
        return
    
    
    
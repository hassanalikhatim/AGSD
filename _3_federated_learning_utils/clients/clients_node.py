import torch


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from .client import Client



class Client_Node:
    
    def __init__(
        self, clients :list[Client]=[]
    ):
        
        self.clients = clients
        
        return
    
    
    def weight_updates(
        self, global_model_state_dict: dict, 
        verbose=True
    ) -> dict:
        
        clients_state_dict = []
        for client in self.clients:
            clients_state_dict.append(
                client.weight_updates(
                    global_model_state_dict, verbose=verbose
                )
            )
            
        node_state_dict = {}
        for key in global_model_state_dict.keys():
            for client_state_dict in clients_state_dict:
                if key in node_state_dict:
                    node_state_dict[key] += client_state_dict[key]
                else:
                    node_state_dict[key] = client_state_dict[key]
            
            node_state_dict[key] = torch.div(node_state_dict[key], len(clients_state_dict))
        
        return node_state_dict
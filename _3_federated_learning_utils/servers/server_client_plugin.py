import numpy as np
import copy
import torch



class Server_Client_Plugin:
    
    def __init__(self, clients_with_keys):
        
        self.prepare_clients(clients_with_keys)
        self.clients_ratio = 0.1
        
        return
    
    
    def prepare_clients(self, clients_with_keys: dict={}):
        
        self.client_with_keys = clients_with_keys
        
        self.clients, self.clients_keys = [], []
        for key in self.client_with_keys.keys():
            self.clients += self.client_with_keys[key]
            self.clients_keys += [key] * len(self.client_with_keys[key])
        
        return
    
    
    def add_client(self, client):
        return self.clients.append(client)
    
    
    def sample_clients(self):
        
        self.active_clients = np.random.choice(
            len(self.clients), int(self.clients_ratio*len(self.clients)), 
            replace=False
        )
        
        return
    
    
    def parameter_flatten_client_state_torch(self, client_state_dict: dict):

        flattened_client_state = []
        for key in client_state_dict.keys():
            if ('weight' in key) or ('bias' in key):
                flattened_client_state += [client_state_dict[key].clone().view(-1)]

        return torch.cat(flattened_client_state)
    

    def parameter_flatten_client_state_np(self, client_state_dict: dict):
        
        flattened_client_state = []
        for key in client_state_dict.keys():
            if ('weight' in key) or ('bias' in key):
                flattened_client_state += client_state_dict[key].clone().cpu().flatten().tolist()
        
        return np.array(flattened_client_state)
    
    
    def key_flatten_client_state_np(self, client_state_dict: dict):
        
        flattened_client_state = []
        for key in client_state_dict.keys():
            flattened_client_state += client_state_dict[key].clone().cpu().flatten().tolist()
        
        return np.array(flattened_client_state)
    
    
    def key_flatten_client_state_torch(self, client_state_dict: dict):
        
        flattened_client_state = []
        for key in client_state_dict.keys():
            flattened_client_state += [client_state_dict[key].clone().view(-1)]

        return torch.cat(flattened_client_state)
    
    
    def key_unflatten_client_state_np(self, flattened_client_state):
        
        client_state_dict_ = copy.deepcopy(self.model.model.state_dict())
        
        flattened_client_state_copy = torch.tensor(flattened_client_state.copy())
        unflattened_client_state = {}
        for key in client_state_dict_.keys():
            np_state_key = client_state_dict_[key].cpu().numpy()
            unflattened_client_state[key] = flattened_client_state_copy[:len(np_state_key.flatten())].reshape(np_state_key.shape)
            flattened_client_state_copy = flattened_client_state_copy[len(np_state_key.flatten()):]
        
        return unflattened_client_state
    
    
    def np_cosine_similarity(self, a, b_s):
        return np.array([np.dot(a, b)/max(np.linalg.norm(a, ord=2)*np.linalg.norm(b, ord=2), 1e-8) for b in b_s])
    
    
    
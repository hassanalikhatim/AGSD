import os
import numpy as np
import pandas as pd
import copy


from utils_.general_utils import confirm_directory

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _3_federated_learning_utils.servers.server import Server
from _3_federated_learning_utils.clients.client import Client



class Helper_Paths:
    
    def __init__(
        self,
        my_model_configuration: dict,
        my_server_configuration: dict,
        my_clients_distribution: dict,
        versioning: bool=True,
        verbose: bool=True
    ):
        
        self.my_model_configuration = my_model_configuration
        self.my_server_configuration = my_server_configuration
        self.my_clients_distribution = my_clients_distribution
        
        self.versioning = versioning
        
        return
    
    
    def print_out(self, *statement, end='\n'):
        if self.verbose:
            print(*statement, end=end)
        return
    
    
    def prepare_identifier_path(self):
        
        self.identifier_path = ''
        for key in self.my_server_configuration.keys():
            self.identifier_path += '_({}-{})'.format(key, self.my_server_configuration[key])
        self.identifier_path += '/'
        for key in self.my_clients_distribution.keys():
           self.identifier_path += '_({}-{})'.format(key, self.my_clients_distribution[key])
        self.identifier_path += '/'
        
        return
    
    
    def prepare_model_name(self, model_name_prefix='federated'):
        
        self.model_name = model_name_prefix
        for key in self.my_model_configuration.keys():
            if key == 'gpu_number':
                self.model_name += '_(gpu_number-0)'
            else:
                self.model_name += '_({}-{})'.format(key, self.my_model_configuration[key])
                
        return
    
    
    def prepare_csv_things(self, csv_file_path: str, filename: str=''):
        
        if csv_file_path[-1] != '/':
            csv_file_path += '/'
        self.csv_file_path = csv_file_path
        confirm_directory(self.csv_file_path)
        
        self.col_name_identifier = self.identifier_path + self.model_name
        
        version = 0
        if self.versioning and (not self.experiment_conducted):
            self.csv_path_and_filename = '{}v{}_{}'.format(self.csv_file_path, version, filename)
            while os.path.isfile('{}v{}_{}'.format(self.csv_file_path, version, filename)):
                version += 1
                self.csv_path_and_filename = '{}v{}_{}'.format(self.csv_file_path, version, filename)
            
            df = pd.DataFrame({'None': [-1.]})
            df.to_csv(self.csv_path_and_filename, index=False)
        
        else:
            self.csv_path_and_filename = '{}{}'.format(self.csv_file_path, filename)
        if filename[-4:] != '.csv':
            self.csv_path_and_filename += '.csv'
            
        self.print_out('csv file path is: {}'.format(self.csv_path_and_filename))
        
        return
    
    
    def prepare_paths_and_names(
        self, 
        results_path: str, 
        csv_file_path: str, 
        model_name_prefix: str='federated', 
        filename: str='_'
    ):
        
        self.prepare_identifier_path()
        
        self.save_path = '{}{}/'.format(results_path, self.my_model_configuration['dataset_name'])
        self.save_path += '{}'.format(self.identifier_path)
        confirm_directory(self.save_path)
        
        self.prepare_model_name(model_name_prefix=model_name_prefix)
        self.experiment_conducted = self.check_conducted()
        self.prepare_csv_things(csv_file_path, filename)
        
        if not self.experiment_conducted:
            print('WARNING: Experiment has not been conducted.')
            
        return
        
        
    def check_conducted(self):
        
        # check if experiment has already been conduction
        experiment_conducted = False
        save_model_path = '{}{}/torch_models/{}.pth'.format(
            self.save_path, self.my_model_configuration['dataset_name'], self.model_name
        )
        if os.path.isfile(save_model_path):
            self.print_out('Hurray...! Model file found at: {}'.format(save_model_path))
            experiment_conducted = True
        else:
            self.print_out('Model file not found at: {}'.format(save_model_path))
        
        return experiment_conducted
    
    
    
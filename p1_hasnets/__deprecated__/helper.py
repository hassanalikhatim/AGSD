import os
import numpy as np
import pandas as pd
import copy


from utils_.general_utils import confirm_directory

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _3_federated_learning_utils.servers.server import Server
from _3_federated_learning_utils.clients.client import Client



class Helper_Hasnets:
    
    def __init__(
        self,
        dataset_name: str,
        my_model_configuration: dict,
        my_server_configuration: dict,
        my_clients_distribution: dict,
        verbose :bool=True
    ):
        
        self.my_model_configuration = my_model_configuration
        self.my_server_configuration = my_server_configuration
        self.my_clients_distribution = my_clients_distribution
        
        self.dictionary_to_save = {}
        
        self.last_client_results = {}
        self.re_evaluated_on_non_patient_server = False
        
        self.verbose = verbose
        
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
    
    
    def prepare_save_path_and_model_name(
        self, 
        results_path: str, 
        csv_file_path: str, 
        model_name_prefix: str='federated', 
        filename: str='_'
    ):
        
        self.csv_file_path = csv_file_path
        confirm_directory(self.csv_file_path)
        
        self.prepare_identifier_path()
        self.save_path = '{}{}/{}'.format(
            results_path, self.my_model_configuration['dataset_name'], self.identifier_path
        )
        confirm_directory(self.save_path)
        
        self.prepare_model_name(model_name_prefix=model_name_prefix)
        
        self.csv_path_and_filename = self.csv_file_path + filename
        self.col_name_identifier = self.identifier_path + self.model_name
        self.extension = '.' + filename.split('.')[-1]
        if self.extension not in ['.xlsx', '.csv']:
            self.extension = '.csv'
            self.csv_path_and_filename += self.extension
        
        experiment_conducted = False
        save_model_path = '{}{}/torch_models/{}.pth'.format(
            self.save_path, self.my_model_configuration['dataset_name'], self.model_name
        )
        if os.path.isfile(save_model_path):
            experiment_conducted = True
        
        return experiment_conducted
    
    
    def prepare_excel_file_to_save_results(self):
        
        if os.path.isfile(self.csv_path_and_filename):
            self.reload_dataframe()
            
            if len(self.df) < self.my_model_configuration['epochs']:
                for i in range(self.my_model_configuration['epochs'] - len(self.df)):
                    self.df.loc[len(self.df)] = [-1. for column in self.df.columns]
                    # self.df.loc[len(self.df)] = [self.df.loc[len(self.df)-1, column] for column in self.df.columns]
            
        else:
            self.df = pd.DataFrame()
            
        show_msg = 'Going to save file at {}'.format(self.csv_path_and_filename)
        show_msg += '\nGoing to save model at {}{}/torch_models/{}'.format(
            self.save_path, self.my_model_configuration['dataset_name'], self.model_name
        )
        self.print_out(show_msg)
        
        return
    
    
    def deprecated_prepare_excel_file_to_save_results(self):
        
        if os.path.isfile(self.csv_path_and_filename):
            self.reload_dataframe()
            
            if len(self.df) < self.my_model_configuration['epochs']:
                for i in range(self.my_model_configuration['epochs'] - len(self.df)):
                    self.df.loc[len(self.df)] = [-1. for column in self.df.columns]
                    # self.df.loc[len(self.df)] = [self.df.loc[len(self.df)-1, column] for column in self.df.columns]
            
        else:
            self.df = pd.DataFrame()
            
        show_msg = 'Going to save file at {}'.format(self.csv_path_and_filename)
        show_msg += '\nGoing to save model at {}{}/torch_models/{}'.format(
            self.save_path, self.my_model_configuration['dataset_name'], self.model_name
        )
        self.print_out(show_msg)
        
        experiment_already_conducted = False
        if self.col_name_identifier+'_test_acc' in self.df.columns:
            self.print_out('Experiment has already been conducted. Let\'s see if it was completed or not.')
            
            current_epoch = self.my_model_configuration['epochs'] - 1
            clean_acc = self.df.loc[current_epoch, self.col_name_identifier+'_test_acc']
            self.print_out('The clean accuracy at the maximum epoch is {:.2f}.'.format(clean_acc))
            
            if clean_acc != -1.:
                self.print_out('Because clean accuracy is not -1, we assume that experiment was completed.')
                experiment_already_conducted = True
        
        return experiment_already_conducted
    
    
    def reload_dataframe(self):
        
        if self.extension == '.xlsx':
            self.df = pd.read_excel(self.csv_path_and_filename, engine='openpyxl')
        if self.extension == '.csv':
            self.df = pd.read_csv(self.csv_path_and_filename)
            
        return
    
    
    def save_dataframe(self):
        
        if self.extension == '.xlsx':
            self.df.to_excel(self.csv_path_and_filename, index=False)
        if self.extension == '.csv':
            self.df.to_csv(self.csv_path_and_filename, index=False)
            
        return
        
    
    def evaluate_all_clients_on_test_set(
        self, epoch: int, 
        clients: dict, server: Server, 
        poisoned_data: Torch_Dataset
    ):
        
        self.reload_dataframe()
        
        for key in self.my_clients_distribution.keys():
            if len(clients[key])>0:
                
                if not self.re_evaluated_on_non_patient_server:
                    slice_of_dict_results = clients[key][0].test_server_model(poisoned_data, server.model)
                    if not server.still_patient:
                        self.re_evaluated_on_non_patient_server = True
                        self.last_client_results = copy.deepcopy(slice_of_dict_results)
                else:
                    slice_of_dict_results = copy.deepcopy(self.last_client_results)
                        
                    
                for key in slice_of_dict_results.keys():
                    active_key = self.col_name_identifier + '_' + key
                    if active_key not in self.df.columns:
                        self.df[active_key] = [-1.] * len(self.df)
                    
                    self.df.loc[epoch, active_key] = slice_of_dict_results[key]
                
                self.save_dataframe()
            
        return
    
    
    def evaluate_server_statistics(self, epoch: int, server: Server):
        
        self.reload_dataframe()
        
        server_stats = server.evaluate_server_statistics()
        for key in server_stats.keys():
            active_key = self.col_name_identifier + '_' + key
            if active_key not in self.df.columns:
                self.df[active_key] = [-1.] * len(self.df)
                
            self.df.loc[epoch, active_key] = server_stats[key]
            
        self.save_dataframe()
        
        return server_stats
    
    
    def prepare_all_clients_given_keys(
        self,
        different_clients_configured,
        client_configurations,
        implemented_clients,
        my_data,
        splits,
        global_model
    ):
        
        index, clients = 0, []
        for _key in self.my_clients_distribution.keys():
            
            key = different_clients_configured[_key]['type']
            
            for k in range( self.my_clients_distribution[_key] ):
                # load configuration from the configuration file
                this_client_configuration = client_configurations[key]
                for updating_key in different_clients_configured[_key].keys():
                    this_client_configuration[updating_key] = different_clients_configured[_key][updating_key]
                
                clients.append(
                    implemented_clients[key](
                        Client_Torch_SubDataset(my_data, idxs=splits.all_client_indices[index]), 
                        global_model.model_configuration,
                        client_configuration=this_client_configuration
                    )
                ); self.print_out('\rPreparing clients:', index, end=''); index += 1
            
            self.print_out()
            
        return clients
    
    
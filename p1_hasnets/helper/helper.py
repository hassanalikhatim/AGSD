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
        my_model_configuration: dict,
        my_server_configuration: dict,
        my_clients_distribution: dict,
        verbose :bool=True,
        versioning :bool=True
    ):
        
        self.my_model_configuration = my_model_configuration
        self.my_server_configuration = my_server_configuration
        self.my_clients_distribution = my_clients_distribution
        
        self.dictionary_to_save = {}
        self.dictionary_to_load = {}
        
        self.last_client_results = {}
        self.re_evaluated_on_non_patient_server = False
        
        self.verbose = verbose
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
        self.prepare_csv_things(csv_file_path, filename)
        
        return
        
        
    def check_conducted(self, data_name: str=''):
        
        if data_name == '':
            data_name = self.my_model_configuration['dataset_name']
        
        # check if experiment has already been conduction
        self.experiment_conducted = False
        save_model_path = '{}{}/torch_models/{}.pth'.format(
            self.save_path, data_name, self.model_name
        )
        if os.path.isfile(save_model_path):
            self.print_out('Hurray...! Model file found at: {}'.format(save_model_path))
            self.experiment_conducted = True
        else:
            self.print_out('Model file not found at: {}'.format(save_model_path))
            print('WARNING: Experiment has not been conducted.')
            
        return
    
    
    def save_dataframe(self, force_overwrite: bool=False):
        
        # get maximum length of the current dictionary
        max_len_of_dict = 0
        for key in self.dictionary_to_save.keys():
            if len(self.dictionary_to_save[key]) > max_len_of_dict:
                max_len_of_dict = len(self.dictionary_to_save[key])
        for key in self.dictionary_to_save.keys():
            self.dictionary_to_save[key] += [self.dictionary_to_save[key][-1]] * (max_len_of_dict-len(self.dictionary_to_save[key]))
        
        # load the df file
        if os.path.isfile(self.csv_path_and_filename):
            df = pd.read_csv(self.csv_path_and_filename)
        else:
            df = pd.DataFrame({'None': [-1]})
            
        # adjust the length of either the dataframe or the dictionary to match each other
        if len(df) > max_len_of_dict:
            diff_len = len(df) - max_len_of_dict
            for key in self.dictionary_to_save.keys():
                self.dictionary_to_save[key] += [self.dictionary_to_save[key][-1]] * diff_len
        elif len(df) < max_len_of_dict:
            diff_len = max_len_of_dict - len(df)
            for i in range(diff_len):
                df.loc[len(df)] = [-1. for column in df.columns]
        
        # copy dictionary to the dataframe
        for key in self.dictionary_to_save.keys():
            if (key not in df.columns) or (force_overwrite):
                if key in df.columns:
                    self.print_out('Overwriting due to force overwrite.')
                assert len(df) == len(self.dictionary_to_save[key]), f'Length of dataframe is {len(df)}, but the length of array is {len(self.dictionary_to_save[key])}'
                df[key] = self.dictionary_to_save[key]
            
        # save the dataframe
        df.to_csv(self.csv_path_and_filename, index=False)
        
        return
    
    
    def load_columns_in_dictionary(self, load_columns):
        
        # load the df file
        df = pd.read_csv(self.csv_path_and_filename)
        
        for column in load_columns:
            if column in df.columns:
                self.dictionary_to_load[column] = df[column].tolist()
        
        return
    
    
    def evaluate_all_clients_on_test_set(
        self, epoch: int, 
        clients: dict, server: Server, 
        poisoned_data: Torch_Dataset
    ):
        
        for key in self.my_clients_distribution.keys():
            if len(clients[key]) > 0:
                
                if not self.re_evaluated_on_non_patient_server:
                    slice_of_dict_results = clients[key][0].test_server_model(poisoned_data, server.model)
                    if not server.still_patient:
                        self.re_evaluated_on_non_patient_server = True
                        self.last_client_results = copy.deepcopy(slice_of_dict_results)
                else:
                    slice_of_dict_results = copy.deepcopy(self.last_client_results)

                for key in slice_of_dict_results.keys():
                    active_key = self.col_name_identifier + '_' + key
                    if active_key not in self.dictionary_to_save.keys():
                        self.dictionary_to_save[active_key] = []
                    
                    self.dictionary_to_save[active_key].append(slice_of_dict_results[key])

        return
    
    
    def evaluate_server_statistics(self, epoch: int, server: Server):
        
        server_stats = server.evaluate_server_statistics()
        for key in server_stats.keys():
            active_key = self.col_name_identifier + '_' + key
            if active_key not in self.dictionary_to_save.keys():
                self.dictionary_to_save[active_key] = []
                
            self.dictionary_to_save[active_key].append(server_stats[key])
            
        return server_stats
    
    
    def get_display_string(self, server_stats: dict):
        
        color = 'white'
        model_str, clean_str, attack_str, eval_str = '', '', '', ''
        for key in server_stats.keys():
            _str = '{:s}-{:s}'.format('{:.3f}'.format(server_stats[key]) if server_stats[key]!=0 else '()', key)[:10]
            _str += ' ' * (10-len(_str)) + ' | '
            if 'train' in key or 'test' in key: model_str += _str
            elif 'clean' in key: clean_str += _str
            elif 'ratio' in key: attack_str += _str; color = 'red' if server_stats[key] != 0 else 'white'
            
        for key in self.dictionary_to_save.keys():
            _str = '{:.2f}-P_Acc'.format(self.dictionary_to_save[key][-1])[:10]
            _str += ' ' * (10-len(_str)) + ' | '
            if '_poisoned_acc' in key: eval_str += _str
            
        return f'{model_str}{clean_str}{attack_str}{eval_str}', color
    
    
    def prepare_csv_things_with_versioning(self, csv_file_path: str, filename: str=''):
        
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
    
    
    
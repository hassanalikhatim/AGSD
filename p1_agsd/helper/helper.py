import os
import numpy as np
import pandas as pd
import copy
from termcolor import colored


from utils_.general_utils import confirm_directory

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.model_utils.torch_model import Torch_Model

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
        self.re_evaluated_on_non_patient_server = 0
        
        self.verbose = verbose
        self.versioning = versioning
        
        return
    
    
    def print_out(self, *statement, end='\n', local_verbose: bool=False):
        if self.verbose or local_verbose:
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
        
        model_name_split_alpha_prefix = ''
        if 'split_type' in self.my_model_configuration.keys() and self.my_model_configuration['split_type'] != 'iid':
            model_name_split_alpha_prefix += f'_(split_type-{self.my_model_configuration['split_type']})'
        
        if 'alpha' in self.my_model_configuration.keys() and self.my_model_configuration['alpha'] != 0:
            model_name_split_alpha_prefix += f'_(alpha-{self.my_model_configuration['alpha']})'
        
        if len(model_name_split_alpha_prefix) > 1: model_name_split_alpha_prefix += '/'
        
        self.model_name = f'{model_name_split_alpha_prefix}{model_name_prefix}'
        for key in self.my_model_configuration.keys():
            if key == 'gpu_number': self.model_name += '_(gpu_number-0)'
            elif (key == 'split_type') or (key == 'alpha'): pass
            else: self.model_name += '_({}-{})'.format(key, self.my_model_configuration[key])
            
        self.model_name_cont = self.model_name + '_continued'
        # print(self.model_name)
        
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
        
        
    def check_conducted(self, data_name: str='', count_continued_as_conducted: bool=True, local_verbose: bool=False):
        
        if data_name == '':
            data_name = self.my_model_configuration['dataset_name']
        
        # check if experiment has already been conduction
        self.experiment_conducted = False
        save_model_path = f'{self.save_path}{data_name}/torch_models/{self.model_name}.pth'
        save_model_path_cont = f'{self.save_path}{data_name}/torch_models/{self.model_name_cont}.pth'
        if os.path.isfile(save_model_path):
            self.print_out(f'{'#'*50}\nHurray...! Model file found at: {save_model_path}\n{'#'*50}', local_verbose=local_verbose)
            self.experiment_conducted = True
        elif count_continued_as_conducted and os.path.isfile(save_model_path_cont):
            self.print_out(f'{'#'*50}\nDidn\'t find model file, but found continued model file found at: {save_model_path}\n{'#'*50}', local_verbose=local_verbose)
            self.experiment_conducted = True
        else:
            self.print_out(f'{'#'*50}\nModel file not found at: {save_model_path}.\n{'#'*50}', local_verbose=local_verbose)
            print('WARNING: Experiment has not been conducted.')
            
        return
    
    
    def save_dataframe(self, force_overwrite: bool=False):
        
        print(force_overwrite)
        
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
        
        
    def load_all_columns_in_dictionary_with_this_string(self, string_: str, give_columns_if_no_data: bool=False):
        
        # load the df file
        df = pd.read_csv(self.csv_path_and_filename)
        found_a_column = False
        
        for column in df.columns:
            if string_ in column:
                self.dictionary_to_load[column] = df[column].tolist()
                found_a_column = True
        
        if give_columns_if_no_data and (not found_a_column):
            self.dictionary_to_load = {key: [] for key in df.columns}
        
        return
    
    
    def evaluate_all_clients_on_test_set(
        self, epoch: int, 
        clients: dict, server: Server, 
        poisoned_data: Torch_Dataset
    ):
        
        for key in self.my_clients_distribution.keys():
            if len(clients[key]) > 0:
                
                if self.re_evaluated_on_non_patient_server < 5:
                    slice_of_dict_results = clients[key][0].test_server_model(poisoned_data, server.model)
                    if not server.still_patient:
                        self.re_evaluated_on_non_patient_server += 1
                        self.last_client_results = copy.deepcopy(slice_of_dict_results)
                else:
                    slice_of_dict_results = copy.deepcopy(self.last_client_results)

                for key in slice_of_dict_results.keys():
                    active_key = self.col_name_identifier + '_' + key
                    if active_key not in self.dictionary_to_save.keys():
                        self.dictionary_to_save[active_key] = []
                    
                    self.dictionary_to_save[active_key].append(slice_of_dict_results[key])

        return
    
    
    def evaluate_server_statistics(self, epoch: int, server: Server, time: float=0):
        
        server_stats = {'time': time, **server.evaluate_server_statistics()}
        for key in server_stats.keys():
            active_key = self.col_name_identifier + '_' + key
            if active_key not in self.dictionary_to_save.keys():
                self.dictionary_to_save[active_key] = []
                
            self.dictionary_to_save[active_key].append(server_stats[key])
            
        return server_stats
    
    
    def get_display_string(self, server_stats: dict):
        
        color = 'white'
        model_str, clean_str, attack_str, eval_str, time_str = '', '', '', '', ''
        for key in server_stats.keys():
            _str = '{:s}-{:s}'.format('{:.3f}'.format(server_stats[key]) if server_stats[key]!=0 else '()', key)[:10]
            _str += ' ' * (10-len(_str)) + ' | '
            if ('train_' in key) or ('test_' in key): model_str += _str
            elif 'clean' in key: clean_str += _str; color = 'light_cyan' if server_stats[key]==0 and color!='light_red' else color
            elif 'ratio' in key: attack_str += _str; color = 'light_red' if server_stats[key]!=0 else color
            elif 'time' in key: time_str += _str
            
        for key in self.dictionary_to_save.keys():
            _str = '{:.2f}-P_Acc'.format(self.dictionary_to_save[key][-1])[:10]
            _str += ' ' * (10-len(_str)) + ' | '
            if '_poisoned_acc' in key: eval_str = eval_str+colored(_str, 'yellow') if self.dictionary_to_save[key][-1]>0.1 else eval_str+_str
            # if '_poisoned_acc' in key and self.dictionary_to_save[key][-1] > 0.1: color = 'yellow' if color=='white' else color
            
        return f'{model_str}{clean_str}{time_str}{attack_str}{eval_str}', color
    
    
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
    
    
    
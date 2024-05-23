import os
import pandas as pd
import copy


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _3_federated_learning_utils.servers.server import Server

from .helper_server import Helper_Server



class Helper_Hasnets(Helper_Server):
    
    def __init__(
        self,
        my_model_configuration: dict,
        my_server_configuration: dict,
        my_clients_distribution: dict,
        verbose :bool=True,
        versioning :bool=True
    ):
        
        super().__init__(
            my_model_configuration, 
            my_server_configuration, 
            my_clients_distribution, 
            versioning=versioning,
            verbose=verbose
        )
        
        self.dictionary_to_save = {}
        self.dictionary_to_load = {}
        
        self.last_client_results = {}
        self.re_evaluated_on_non_patient_server = False
        
        return
    
    
    def save_dataframe(self):
        
        # get maximum length of the current dictionary
        max_len_of_dict = 0
        for key in self.dictionary_to_save.keys():
            if len(self.dictionary_to_save[key]) > max_len_of_dict:
                max_len_of_dict = len(self.dictionary_to_save[key])
        
        # load the df file
        if os.path.isfile(self.csv_path_and_filename):
            df = pd.read_csv(self.csv_path_and_filename)
            
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
    
    
    def evaluate_server_statistics(self, epoch: int, server: Server):
        
        server_stats = server.evaluate_server_statistics()
        for key in server_stats.keys():
            active_key = self.col_name_identifier + '_' + key
            if active_key not in self.dictionary_to_save.keys():
                self.dictionary_to_save[active_key] = []
                
            self.dictionary_to_save[active_key].append(server_stats[key])
            
        return server_stats
    
    
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

                for key_ in slice_of_dict_results.keys():
                    active_key = self.col_name_identifier + '_' + key_
                    if active_key not in self.dictionary_to_save.keys():
                        self.dictionary_to_save[active_key] = []
                    
                    self.dictionary_to_save[active_key].append(slice_of_dict_results[key_])

        return
    
    
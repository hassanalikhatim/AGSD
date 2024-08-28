from p1_agsd.config import visible_gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu

import argparse


from p1_agsd.scripts.train_federated_multi import main as main_federated_mp
from p1_agsd.scripts.prepare_federated_csvs import main as main_csvs

from p1_agsd.scripts.update_names import main as update_names

from p1_agsd.scripts.generate_results import generate_all_results_and_tables



if __name__ == '__main__':
    
    download_pretrained_models_help = 'Download pretrained models from google drive. This option won\'t be available for long.'
    
    parser = argparse.ArgumentParser(description='AGSD')
    parser.add_argument('--download_pretrained_models', action='store_true', default=False, help=download_pretrained_models_help)
    parser.add_argument('--train_models', action='store_true', default=False, help='Trains new models if not found in the directory.')
    parser.add_argument('--compile_results', action='store_true', default=False, help='Compile results in a csv. This won\'t show results.')
    parser.add_argument('--generate_results', action='store_true', default=False, help='Prints tables in the terminal in latex format and saves results figures.')
    all_arguments = parser.parse_args()
    
    if all_arguments.download_pretrained_models:
        print('Write a script for downloading pretrained models using gdown.')
        # git clone 
        
    if all_arguments.train_models:
        main_federated_mp()
        
    if all_arguments.compile_results:
        main_csvs()
        
    if all_arguments.generate_results:
        generate_all_results_and_tables()
    
    # update_names()
    
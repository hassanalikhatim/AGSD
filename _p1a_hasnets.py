from p1_hasnets.config import visible_gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu


from p1_hasnets.scripts.train_federated_multi import main as main_federated_mp
from p1_hasnets.scripts.prepare_federated_csvs import main as main_csvs

from p1_hasnets.scripts.update_names import main as update_names

from p1_hasnets.scripts.generate_results import generate_all_results_and_tables



if __name__ == '__main__':
    
    main_federated_mp()
    # main_csvs()
    # generate_all_results_and_tables()
    
    # update_names()
    
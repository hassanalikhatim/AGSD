from p1_hasnets.config import visible_gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu


from p1_hasnets.scripts.train_federated import main as main_federated
from p1_hasnets.scripts.train_federated_mp import main as main_federated_mp
from p1_hasnets.scripts.prepare_federated_csvs import main as main_csvs



if __name__ == '__main__':
    
    # main_federated()
    main_federated_mp()
    # main_csvs()
    
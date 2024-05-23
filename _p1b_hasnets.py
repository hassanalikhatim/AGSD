from p1_hasnets.config import visible_gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


from p1_hasnets.scripts.train_federated import main as main_federated
from p1_hasnets.scripts.train_federated_mp import main as main_federated_mp



if __name__ == '__main__':
    
    # main_federated(orientation=1)
    main_federated_mp(orientation=1)
    
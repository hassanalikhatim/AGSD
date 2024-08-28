import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


from p1_agsd.scripts.train_federated_multi import main as main_federated_mp



if __name__ == '__main__':
    
    main_federated_mp(orientation=1)
    
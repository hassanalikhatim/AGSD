import torch


from ..config import *

from _0_general_ML.data_utils.datasets import MNIST, GTSRB, CIFAR10

from _0_general_ML.model_utils.torch_model import Torch_Model



datasets = {
    'mnist': MNIST,
    'gtsrb': GTSRB,
    'cifar10': CIFAR10
}


def main():
    
    dataset_name = 'cifar10'
    
    my_model_configuration = model_configurations[dataset_name].copy()
    my_model_configuration['dataset_name'] = dataset_name
                
    data = datasets[dataset_name]()
    
    global_model = Torch_Model(
        data, my_model_configuration,
        path=results_path
    )
    print(global_model.model)
    print(
        'Learnable Parameters: {}'
        ''.format(
            sum(p.numel() for p in global_model.model.parameters() if p.requires_grad)
        )
    )
    
    for i in range(my_model_configuration['epochs']):
        show_str = ''
        
        global_model.train(
            epochs=1, 
            batch_size=my_model_configuration['batch_size']
        )
        show_str += 'Epochs {}, '.format(i)
        
        train_loader, test_loader = data.prepare_data_loaders(batch_size=my_model_configuration['batch_size'])
        train_loss, train_acc = global_model.test_shot(train_loader)
        test_loss, test_acc = global_model.test_shot(test_loader)
        show_str += 'train_loss: {:.5f} | train_acc: {:.2f} | test_loss: {:.5f} | test_acc: {:.2f}'.format(
            train_loss, train_acc, test_loss, test_acc
        )
        
        print('\r' + show_str)
    
    return


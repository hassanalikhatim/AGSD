import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.datasets import MNIST, GTSRB, CIFAR10, CIFAR100
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset

from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor
from _1_adversarial_ML.backdoor_attacks.class_specific_backdoor_attack import Class_Specific_Backdoor
from _1_adversarial_ML.adversarial_attacks.fgsm import FGSM
from _1_adversarial_ML.adversarial_attacks.ifgsm import i_FGSM
from _1_adversarial_ML.adversarial_attacks.pgd import PGD

from ..agsd_servers.agsd_adv_attack import AGSD_Adversarial_Attack
from .healing_direction import prepare_id_healing_data, prepare_ood_healing_data, prepare_noise_random_healing_data



class My_Model_for_Analysis(Torch_Model):

    def __init__(self, data: Torch_Dataset, model_configuration: dict={}, path: str='', **kwargs):
        super().__init__(data, model_configuration, path=path, **kwargs)

    def parameter_flatten_client_state(self):
        client_state_dict = self.model.state_dict().copy()

        flattened_client_state = []
        for key in client_state_dict.keys():
            if ('weight' in key) or ('bias' in key):
              flattened_client_state += [client_state_dict[key].view(-1)]

        return torch.cat(flattened_client_state).view(1, -1)
    

def perform_std_and_conf_analysis(
    analysis_type: str='id', 
    attack_type: str='fgsm', attack_iterations: int=50, attack_epsilon: float=0.2,
    dataframe_name: str='std_conf_analysis.csv'
):
    
    healing_data_loader = {
        'id': prepare_id_healing_data,
        'ood': prepare_ood_healing_data,
        'noise_random': prepare_noise_random_healing_data
    }
    
    available_attacks = {
        'fgsm': FGSM,
        'ifgsm': i_FGSM,
        'pgd': PGD
    }

    my_data = MNIST()
    
    my_poisoned_data_1 = Simple_Backdoor(my_data, backdoor_configuration={'poison_ratio': 0.001})
    my_poisoned_data_2 = Simple_Backdoor(my_data, backdoor_configuration={'poison_ratio': 0.003})
    my_poisoned_data_3 = Simple_Backdoor(my_data, backdoor_configuration={'poison_ratio': 0.005})
    my_poisoned_data_4 = Simple_Backdoor(my_data, backdoor_configuration={'poison_ratio': 0.01})
    # poisoned_test_loader = torch.utils.data.DataLoader(my_poisoned_data_1.poisoned_test, batch_size=1024)
    
    healing_data = healing_data_loader[analysis_type](my_data, size=500)
    
    gc.collect()

    model_architectures = ['resnet50_gtsrb', 'cnn_gtsrb', 'mnist_cnn', 'cifar10_vgg11']
    model_configuration = {
        'model_architecture': model_architectures[2],
        'learning_rate': 1e-3,
        'batch_size': 1024,
        'loss_fn': 'crossentropy',
        'gpu_number': 0
    }
    model_configuration_healing = model_configuration.copy()
    model_configuration_healing['learning_rate'] = 1e-5

    f_plus = My_Model_for_Analysis(my_data, model_configuration=model_configuration)
    f_minus_1 = My_Model_for_Analysis(my_poisoned_data_1, model_configuration=model_configuration)
    f_minus_1.model.load_state_dict(f_plus.model.state_dict().copy())
    f_minus_2 = My_Model_for_Analysis(my_poisoned_data_2, model_configuration=model_configuration)
    f_minus_2.model.load_state_dict(f_plus.model.state_dict().copy())
    f_minus_3 = My_Model_for_Analysis(my_poisoned_data_3, model_configuration=model_configuration)
    f_minus_3.model.load_state_dict(f_plus.model.state_dict().copy())
    f_minus_4 = My_Model_for_Analysis(my_poisoned_data_4, model_configuration=model_configuration)
    f_minus_4.model.load_state_dict(f_plus.model.state_dict().copy())
    
    f_neutral = My_Model_for_Analysis(my_data, model_configuration=model_configuration)
    
    
    def get_std_and_confidence(model: Torch_Model, data: Torch_Dataset, epsilon: float=0.2, iterations: int=50) -> np.ndarray:
        
        def get_confidence(arr_in: torch.Tensor): 
            arr_ = torch.exp(arr_in) / torch.sum(torch.exp(arr_in), dim=1, keepdim=True)
            arr_ = torch.mean(arr_, dim=0)
            return torch.max(arr_)
        
        def additive_stds(arr_in: torch.Tensor): 
            arr_ = torch.argmax(arr_in, dim=1)
            z_ = torch.nn.functional.one_hot(arr_, num_classes=len(data.get_class_names())).float()
            return torch.mean(torch.std(z_, dim=0, unbiased=False))
        
        def attack_dataset(model: Torch_Model, data: Torch_Dataset, epsilon: float=0.2, iterations: int=50) -> Torch_Dataset:
            x, y = [], []
            for i in range(data.train.__len__()):
                _x, _y = data.train.__getitem__(i)
                x.append(_x.detach().cpu().numpy()); y.append(_y)
            x = np.array(x); y = np.array(y)
            
            delta = np.max(x)-np.min(x)
            
            attacker = available_attacks[attack_type](model)
            perturbed_x_inputs = np.random.uniform(-0.05*delta, 0.05*delta, size=x.shape).astype(np.float32)
            adv_x = attacker.attack(perturbed_x_inputs, y, epsilon=epsilon*delta, targeted=False, iterations=iterations)
            
            healing_data_adv = Torch_Dataset(data_name=data.data_name)
            healing_data_adv.train = torch.utils.data.TensorDataset(torch.tensor(adv_x).float(), torch.tensor(y))
            healing_data_adv.test = data.test
            
            return healing_data_adv
        
        f_neutral.model.load_state_dict(model.model.state_dict())
        adv_data = attack_dataset(f_neutral, data, epsilon=epsilon, iterations=iterations)
        adv_y, _ = f_neutral.predict(torch.utils.data.DataLoader(adv_data.train, shuffle=False, batch_size=f_neutral.model_configuration['batch_size']), verbose=False)
        
        return additive_stds(adv_y).detach().cpu().numpy().reshape(-1)[0], get_confidence(adv_y).detach().cpu().numpy().reshape(-1)[0]

    std_values = {
        'clean_std': [],
        'poisoned_std_1': [],
        'poisoned_std_2': [],
        'poisoned_std_3': [],
        'poisoned_std_4': [],
        
        'clean_conf': [],
        'poisoned_conf_1': [],
        'poisoned_conf_2': [],
        'poisoned_conf_3': [],
        'poisoned_conf_4': [],
        
    }
    for epoch in range(20):
        print('\n\nEpoch: {}/{}'.format(epoch, 30))
        
        # print('F Plus:')
        # initial_state = f_plus.model.state_dict().copy()
        f_plus.train(epochs=1)
        f_minus_1.train(epochs=1)
        f_minus_2.train(epochs=1)
        f_minus_3.train(epochs=1)
        f_minus_4.train(epochs=1)
        
        s_p, c_p = get_std_and_confidence(f_plus, healing_data, epsilon=attack_epsilon, iterations=attack_iterations)
        s_n1, c_n1 = get_std_and_confidence(f_minus_1, healing_data, epsilon=attack_epsilon, iterations=attack_iterations)
        s_n2, c_n2 = get_std_and_confidence(f_minus_2, healing_data, epsilon=attack_epsilon, iterations=attack_iterations)
        s_n3, c_n3 = get_std_and_confidence(f_minus_3, healing_data, epsilon=attack_epsilon, iterations=attack_iterations)
        s_n4, c_n4 = get_std_and_confidence(f_minus_4, healing_data, epsilon=attack_epsilon, iterations=attack_iterations)
        
        std_values['clean_std'].append(s_p); std_values['clean_conf'].append(c_p)
        std_values['poisoned_std_1'].append(s_n1); std_values['poisoned_conf_1'].append(c_n1)
        std_values['poisoned_std_2'].append(s_n2); std_values['poisoned_conf_2'].append(c_n2)
        std_values['poisoned_std_3'].append(s_n3); std_values['poisoned_conf_3'].append(c_n3)
        std_values['poisoned_std_4'].append(s_n4); std_values['poisoned_conf_4'].append(c_n4)
        
        print()
        print('\n'.join([
            '{}: {}'.format(key, std_values[key])
            for key in std_values.keys()
        ]))
        
    data_dict = {f'{my_data.data_name}-{analysis_type}-{attack_type}_{attack_iterations}_{attack_epsilon}-{key}': std_values[key] for key in std_values.keys()}
    save_dataframe_from_dict(data_dict, dataframe_name)
    
    return


def save_dataframe_from_dict(data_dict: dict, dataframe_name: str, force_overwrite: bool=False):
    
    # get maximum length of the current dictionary
    max_len_of_dict = 0
    for key in data_dict.keys():
        if len(data_dict[key]) > max_len_of_dict:
            max_len_of_dict = len(data_dict[key])
    for key in data_dict.keys():
        data_dict[key] += [data_dict[key][-1]] * (max_len_of_dict-len(data_dict[key]))
    
    # load the df file
    if os.path.isfile(dataframe_name):
        df = pd.read_csv(dataframe_name)
    else:
        df = pd.DataFrame({'None': [-1]})
        
    # adjust the length of either the dataframe or the dictionary to match each other
    if len(df) > max_len_of_dict:
        diff_len = len(df) - max_len_of_dict
        for key in data_dict.keys():
            data_dict[key] += [data_dict[key][-1]] * diff_len
    elif len(df) < max_len_of_dict:
        diff_len = max_len_of_dict - len(df)
        for i in range(diff_len):
            df.loc[len(df)] = [-1. for column in df.columns]
    
    # copy dictionary to the dataframe
    for key in data_dict.keys():
        if (key not in df.columns) or (force_overwrite):
            if key in df.columns:
                print('Overwriting due to force overwrite.')
            assert len(df) == len(data_dict[key]), f'Length of dataframe is {len(df)}, but the length of array is {len(data_dict[key])}'
            df[key] = data_dict[key]
        
    # save the dataframe
    df.to_csv(dataframe_name, index=False)
    
    return
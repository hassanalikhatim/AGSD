import numpy as np
import torch
import gc
import matplotlib.pyplot as plt


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.datasets import MNIST, GTSRB, CIFAR10, CIFAR100
from _0_general_ML.data_utils.torch_subdataset import Client_Torch_SubDataset

from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor



def prepare_ood_healing_data(data: Torch_Dataset, size: int=500):
    channels, height, width = data.train.__getitem__(0)[0].shape
    another_dataset = CIFAR100(preferred_size=height)
    num_classes = len(data.get_class_names())

    samples = []; labels = []; random_sample_indices = []
    while len(random_sample_indices) < size:
        random_index = np.random.randint(another_dataset.train.__len__())
        x, y = another_dataset.train.__getitem__(random_index)
        if y < num_classes:
            samples.append(x.detach().cpu().numpy()); labels.append(y)
            random_sample_indices.append(random_index)
        print('\r Len of healing data:', len(random_sample_indices), end='')
    samples = np.array(samples); labels = np.array(labels)
    
    if samples.shape[1] != channels:
        channeled_samples = []
        for c in range(channels):
            channeled_samples.append(samples[:, 0])
        channeled_samples = np.array(channeled_samples)
        channeled_samples = np.transpose(channeled_samples, axes=(1, 0, 2, 3))
    else:
        channeled_samples = samples
    
    healing_data = Torch_Dataset(data_name=data.data_name)
    healing_data.train = torch.utils.data.TensorDataset(torch.tensor(channeled_samples), torch.tensor(labels))
    healing_data.test = data.test
    
    return healing_data


def prepare_ood_random_healing_data(data: Torch_Dataset, size: int=500):
    
    channels, height, width = data.train.__getitem__(0)[0].shape
    another_dataset = CIFAR100(preferred_size=height)
    num_classes = len(data.get_class_names())

    samples = []; labels = []; random_sample_indices = []
    while len(random_sample_indices) < size:
        random_index = np.random.randint(another_dataset.train.__len__())
        x, y = another_dataset.train.__getitem__(random_index)
        if y < num_classes:
            samples.append(x.detach().cpu().numpy()); labels.append(y)
            random_sample_indices.append(random_index)
        print('\r Len of healing data:', len(random_sample_indices), end='')
    samples = np.array(samples); labels = np.array(labels)
    labels = np.random.randint(np.min(labels), np.max(labels), size=(len(samples)))
    
    if samples.shape[1] != channels:
        channeled_samples = []
        for c in range(channels):
            channeled_samples.append(samples[:, 0])
        channeled_samples = np.array(channeled_samples)
        channeled_samples = np.transpose(channeled_samples, axes=(1, 0, 2, 3))
    else:
        channeled_samples = samples
    
    healing_data = Torch_Dataset(data_name=data.data_name)
    healing_data.train = torch.utils.data.TensorDataset(torch.tensor(channeled_samples), torch.tensor(labels))
    healing_data.test = data.test
    
    return healing_data


def prepare_noise_random_healing_data(data: Torch_Dataset, size: int=500):
    
    channels, height, width = data.train.__getitem__(0)[0].shape
    num_classes = len(data.get_class_names())

    samples = []; labels = []; random_sample_indices = []
    while len(random_sample_indices) < size:
        random_index = np.random.randint(data.train.__len__())
        x, y = data.train.__getitem__(random_index)
        if y < num_classes:
            samples.append(x.detach().cpu().numpy()); labels.append(y)
            random_sample_indices.append(random_index)
        print('\r Len of healing data:', len(random_sample_indices), end='')
    samples = np.array(samples); labels = np.array(labels)

    channeled_samples = np.random.uniform(np.min(samples), np.max(samples), size=samples.shape).astype(samples.dtype)
    labels = np.random.randint(np.min(labels), np.max(labels), size=(len(samples)))
    
    healing_data = Torch_Dataset(data_name=data.data_name)
    healing_data.train = torch.utils.data.TensorDataset(torch.tensor(channeled_samples), torch.tensor(labels))
    healing_data.test = data.test
    
    return healing_data


def prepare_id_healing_data(data: Torch_Dataset, size: int=5000):
    return Client_Torch_SubDataset(data, np.arange(size))


def perturb_model_state(state_in: dict) -> dict:
    
    state_t = state_in.copy()
    for key in state_t.keys():
        state_t[key] = state_t[key].float()
        if not ('bias' in key or 'bn' in key):
            try:
                standard_deviation = 1e-5 * torch.std(state_t[key].float().clone().view(-1), unbiased=False)
                state_t[key] += torch.normal(0., standard_deviation, size=state_t[key].shape).to(state_t[key].device)
            except: pass
    
    return state_t


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
    
    
def perform_analysis_and_save_figure(analysis_type: str='ood'):
    
    healing_data_loader = {
        'id': prepare_id_healing_data,
        'ood': prepare_ood_healing_data,
        'ood_random': prepare_ood_random_healing_data,
        'noise_random': prepare_noise_random_healing_data
    }
    
    my_data = GTSRB()
    my_poisoned_data = Simple_Backdoor(my_data, backdoor_configuration={'poison_ratio': 0.3})
    healing_data = healing_data_loader[analysis_type](my_data, size=500)
    gc.collect()
    
    model_architectures = ['resnet50_gtsrb', 'cnn_gtsrb', 'mnist_cnn', 'cifar10_vgg11']
    model_configuration = {
        'model_architecture': model_architectures[0],
        'learning_rate': 1e-3,
        'batch_size': 1024,
        'loss_fn': 'crossentropy',
        'gpu_number': 0
    }
    model_configuration_healing = model_configuration.copy()
    model_configuration_healing['learning_rate'] = 1e-5

    f_plus = My_Model_for_Analysis(my_data, model_configuration=model_configuration)
    f_minus = My_Model_for_Analysis(my_poisoned_data, model_configuration = model_configuration)

    fh_plus = My_Model_for_Analysis(healing_data, model_configuration=model_configuration_healing)
    fh_minus = My_Model_for_Analysis(healing_data, model_configuration=model_configuration_healing)

    f_minus.model.load_state_dict(f_plus.model.state_dict().copy())
    fh_plus.model.load_state_dict(f_plus.model.state_dict().copy())
    fh_minus.model.load_state_dict(f_plus.model.state_dict().copy())
    
    distance_between_models = {
        'f_plus_f_minus': [],
        'f_plus_fh_minus': [],
        'f_plus_fh_plus': [],
        'f_minus_fh_minus': [],
        'f_minus_fh_plus': [],
        'f_plus_healing_towards_minus': [],
        'f_minus_healing_towards_plus': []
    }
    cs = torch.nn.CosineSimilarity()
    # def cs(a, b):
    #   return torch.norm(a-b, p=2)
    
    for epoch in range(20):
        print('\n\nEpoch: {}/{}'.format(epoch, 30))

        print('F Plus:')
        initial_state = f_plus.model.state_dict().copy()
        f_plus.train(epochs=1)
        fh_plus.model.load_state_dict(perturb_model_state(f_plus.model.state_dict()).copy())
        fh_plus.train(epochs=5)

        print('\nF Minus:')
        f_minus.model.load_state_dict(initial_state.copy())
        f_minus.train(epochs=1)
        fh_minus.model.load_state_dict(perturb_model_state(f_plus.model.state_dict()).copy())
        fh_minus.train(epochs=5)
        gc.collect()

        f_plus_state = f_plus.parameter_flatten_client_state()
        fh_plus_state = fh_plus.parameter_flatten_client_state()
        f_minus_state = f_minus.parameter_flatten_client_state()
        fh_minus_state = fh_minus.parameter_flatten_client_state()

        distance_between_models['f_plus_f_minus'].append( cs(f_plus_state, f_minus_state).item() )
        distance_between_models['f_plus_fh_minus'].append( cs(f_plus_state, fh_minus_state).item() )
        distance_between_models['f_plus_fh_plus'].append( cs(f_plus_state, fh_plus_state).item() )
        distance_between_models['f_minus_fh_minus'].append( cs(f_minus_state, fh_minus_state).item() )
        distance_between_models['f_minus_fh_plus'].append( cs(f_minus_state, fh_plus_state).item() )
        distance_between_models['f_plus_healing_towards_minus'].append(cs(
                fh_plus_state - f_plus_state,
                f_minus_state - f_plus_state
        ).item())
        distance_between_models['f_minus_healing_towards_plus'].append(cs(
                fh_minus_state - f_minus_state,
                f_plus_state - f_minus_state
        ).item())

        print()
        print('\n'.join([
            '{}: {}'.format(key, distance_between_models[key])
            for key in distance_between_models.keys()
        ]))
    
    plt.figure(figsize=(4, 3))
    plt.plot(np.array(distance_between_models['f_plus_fh_minus']) - np.array(distance_between_models['f_plus_f_minus']), label='$f_-$ healing towards $f_+$')
    plt.plot(np.array(distance_between_models['f_minus_fh_plus']) - np.array(distance_between_models['f_plus_f_minus']), label='$f_+$ healing towards $f_-$')
    plt.plot(1 - np.array(distance_between_models['f_plus_fh_plus']), label='$f_+$ healing away from $f_+$')
    # plt.plot(1 - np.array(distance_between_models['f_minus_fh_minus']), label='$f_+$ healing towards $f_+$')
    plt.xticks(np.arange(0, 25, 5))
    plt.yscale('log')
    plt.ylabel('Strength of Healing')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'p1_hasnets/__paper__/figures/poisoned_healing_towards_clean_{analysis_type}.pdf')
    
    return


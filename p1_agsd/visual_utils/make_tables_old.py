import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from .loading import load_results_from_settings



def comparison_with_sota_(dataset_names, results_path_local: str):
    
    nicer_names = {
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
        'clean': 'Clean', 
        'simple_(poison-0.75)_(scale-2)': 'Simple Backdoor Attack',
        'simple_(num_clients-100)_(clients_ratio-0.1)': 'FedAvg',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'HasNet (Heldout)',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'HasNet (Noise)',
        'dp_(num_clients-100)_(clients_ratio-0.1)': 'DP-SGD',
        'krum_(num_clients-100)_(clients_ratio-0.1)': 'Krum',
        'foolsgold_(num_clients-100)_(clients_ratio-0.1)': 'FoolsGold',
        'deepsight_(num_clients-100)_(clients_ratio-0.1)': 'DeepSight',
        'flame_(num_clients-100)_(clients_ratio-0.1)': 'Flame'
    }
    
    clients_distributions = [
        {},
        # simple backdoor analysis with different number of backdoor clients
        {'simple_(poison-0.75)_(scale-2)': 0.1},
        {'simple_(poison-0.75)_(scale-2)': 0.3},
        {'simple_(poison-0.75)_(scale-2)': 0.5},
        {'simple_(poison-0.75)_(scale-2)': 0.7},
        {'simple_(poison-0.75)_(scale-2)': 0.9},
        # # different backdoor clients (one at a time) with 30% backdoor distribution
        # {'simple_(poison-0.75)_(scale-2)': 0.3},
        # {'invisible_(poison-0.75)_(scale-2)': 0.3},
        # {'neurotoxin_(poison-0.75)_(scale-2)': 0.3},
        # {'iba_(poison-0.75)_(scale-2)': 0.3},
    ]
    
    server_types = [
        'simple_(num_clients-100)_(clients_ratio-0.1)',
        'dp_(num_clients-100)_(clients_ratio-0.1)',
        'krum_(num_clients-100)_(clients_ratio-0.1)',
        'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
        'deepsight_(num_clients-100)_(clients_ratio-0.1)',
        'flame_(num_clients-100)_(clients_ratio-0.1)',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(clients_distributions))
    
    table_string = ''
    for s, server_type in enumerate(server_types):
        table_string += '{}'.format(nicer_names[server_type])
        
        for d, dataset_name in enumerate(dataset_names):
            for c, clients_distribution in enumerate(clients_distributions):
                
                result_ = results_arr[d, c, s]
                table_string += ' & {}({})'.format(
                    '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                    '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                )
        
            table_string += '\n'
        table_string += '\\\\\n'
    
    return table_string
    
    
def comparison_with_sota(dataset_names, results_path_local: str):
    
    nicer_names = {
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
        'clean': 'Clean', 
        'simple_(poison-0.75)_(scale-2)': 'Simple Backdoor Attack',
        'simple_(num_clients-100)_(clients_ratio-0.1)': 'FedAvg',
        'dp_(num_clients-100)_(clients_ratio-0.1)': 'DP-SGD',
        'krum_(num_clients-100)_(clients_ratio-0.1)': 'Krum',
        'foolsgold_(num_clients-100)_(clients_ratio-0.1)': 'FoolsGold',
        'deepsight_(num_clients-100)_(clients_ratio-0.1)': 'DeepSight',
        'flame_(num_clients-100)_(clients_ratio-0.1)': 'Flame',
        'mesas_(num_clients-100)_(clients_ratio-0.1)': 'MESAS',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\dfh{}',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\df{}',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\dfo{}',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\dfh{}',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\df{}',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\dfo{}',
    }
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    clients_distributions = [
        {},
        {'simple_(poison-0.25)_(scale-2)': 0.45},
        {'invisible_(poison-0.25)_(scale-2)': 0.45},
        {'neurotoxin_(poison-0.25)_(scale-2)': 0.45},
        {'iba_(poison-0.25)_(scale-2)': 0.45},
        # {'class_specific_(poison-0.25)_(scale-2)': 0.45},
    ]
    
    server_types = [
        'simple_(num_clients-100)_(clients_ratio-0.1)',
        'dp_(num_clients-100)_(clients_ratio-0.1)',
        'krum_(num_clients-100)_(clients_ratio-0.1)',
        'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
        'deepsight_(num_clients-100)_(clients_ratio-0.1)',
        'flame_(num_clients-100)_(clients_ratio-0.1)',
        'mesas_(num_clients-100)_(clients_ratio-0.1)',
        # 'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        # 'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
        # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(clients_distributions))
    
    table_string = ''
    for s, server_type in enumerate(server_types):
        table_string += '{}'.format(nicer_names[server_type])
        
        for d, dataset_name in enumerate(dataset_names):
            for c, clients_distribution in enumerate(clients_distributions):
                
                result_ = results_arr[d, c, s]
                table_string += ' & {}({})'.format(
                    '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                    '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                )
        
            table_string += '\n'
        table_string += '\\\\\n'
    
    return table_string


def non_iid_analysis(dataset_names, results_path_local: str):
    
    nicer_names = {
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
        'clean': 'Clean', 
        'simple_(poison-0.75)_(scale-2)': 'Simple Backdoor Attack',
        'simple_(num_clients-100)_(clients_ratio-0.1)': 'FedAvg',
        'dp_(num_clients-100)_(clients_ratio-0.1)': 'DP-SGD',
        'krum_(num_clients-100)_(clients_ratio-0.1)': 'Krum',
        'foolsgold_(num_clients-100)_(clients_ratio-0.1)': 'FoolsGold',
        'deepsight_(num_clients-100)_(clients_ratio-0.1)': 'DeepSight',
        'flame_(num_clients-100)_(clients_ratio-0.1)': 'Flame',
        'mesas_(num_clients-100)_(clients_ratio-0.1)': 'MESAS',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\dfh{}',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\df{}',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\dfo{}',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\dfh{}',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\df{}',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\dfo{}',
    }
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    clients_distributions = [
        {'simple_(poison-0.25)_(scale-2)': 0.45},
    ]
    
    server_types = [
        'simple_(num_clients-100)_(clients_ratio-0.1)',
        'krum_(num_clients-100)_(clients_ratio-0.1)',
        'flame_(num_clients-100)_(clients_ratio-0.1)',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(clients_distributions))
    
    table_string = ''
    for s, server_type in enumerate(server_types):
        table_string += '{}'.format(nicer_names[server_type])
        
        for d, dataset_name in enumerate(dataset_names):
                
            result_ = results_arr[d, 0, s]
            table_string += ' & {}({})'.format(
                '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
            )
        
        table_string += '\n\\\\\n'
    
    return table_string


def hyperparameter_analysis_healing_set_size(dataset_names, results_path_local: str, save_fig: bool=True):
    
    nicer_names = {
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
        'clean': 'Clean', 
        'simple_(poison-0.75)_(scale-2)': 'VTBA',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)': 'H-10',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'H-50',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': 'H-100',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)': 'H-500',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)': 'H-1000',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-1)': 'H-10-1',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)': 'H-50-1',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-1)': 'H-100-1',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-1)': 'H-500-1',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-1)': 'H-1000-1',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)_(healing_epochs-1)': 'H-5000-1',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)': 'O-10',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'O-50',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': 'O-100',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)': 'O-500',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)': 'O-1000',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-1)': 'O-10-1',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)': 'O-50-1',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-1)': 'O-100-1',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-1)': 'O-500-1',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-1)': 'O-1000-1',
        
    }
    
    # start generating figure of hyperparameter analysis
    clients_distributions = [ {'simple_(poison-0.25)': 0.45} ]
    keys = ['test_acc', 'poisoned_acc']
    healing_set_sizes = [10, 50, 100, 500, 1000]
    server_natures = ['heldout', 'ood']
    
    figs = []
    for d, dataset_name in enumerate(dataset_names):
        for s, server_nature in enumerate(server_natures):
            
            server_types = []
            for healing_set_size in healing_set_sizes:
                server_types.append(f'hasnet_{server_nature}_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-{healing_set_size})')
            
            results_arr = load_results_from_settings(
                [dataset_name], 
                clients_distributions, 
                server_types, 
                keys=keys,
                results_path_local=results_path_local
            )
            
            for c, clients_distribution in enumerate(clients_distributions):
                fig = plt.figure(figsize=(2.5, 2))
                plt.plot(results_arr[d, c, :, 0], marker='s', label='CA', color='blue')
                plt.plot(results_arr[d, c, :, 1], marker='^', label='ASR', color='red')
                plt.xticks(range(len(healing_set_sizes)), healing_set_sizes)
                plt.yticks(np.arange(0., 1.0, 0.15))
                plt.ylabel('Percentage')
                plt.xlabel('Healing Set Size: $|D_h|$')
                plt.legend()
                plt.tight_layout()
                
                figs.append(fig)
    
    if save_fig:
        with PdfPages('p1_hasnets/__paper__/figures/hyperparameter_healing_set_size.pdf') as p:
            for fig in figs:
                fig.savefig(p, format='pdf')
    
    return


def hyperparameter_backdoor_scale(dataset_names, results_path_local: str, save_fig: bool=True):
    
    nicer_names = {
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
        'clean': 'Clean', 
        'simple_(poison-0.75)_(scale-2)': 'VTBA',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)': 'H-10',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'H-50',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': 'H-100',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)': 'H-500',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)': 'H-1000',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-1)': 'H-10-1',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)': 'H-50-1',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-1)': 'H-100-1',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-1)': 'H-500-1',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-1)': 'H-1000-1',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-5000)_(healing_epochs-1)': 'H-5000-1',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)': 'O-10',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'O-50',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': 'O-100',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)': 'O-500',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)': 'O-1000',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-10)_(healing_epochs-1)': 'O-10-1',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(healing_epochs-1)': 'O-50-1',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)_(healing_epochs-1)': 'O-100-1',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-500)_(healing_epochs-1)': 'O-500-1',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-1000)_(healing_epochs-1)': 'O-1000-1',
        
    }
    
    # start generating figure of hyperparameter analysis
    clients_distributions = [
        {'simple_(poison-0.25)': 0.45},
        {'simple_(poison-0.25)_(scale-2)': 0.45},
        {'simple_(poison-0.25)_(scale-3)': 0.45},
        {'simple_(poison-0.25)_(scale-5)': 0.45},
    ]
    keys = ['test_acc', 'poisoned_acc']
    healing_set_sizes = [10, 50, 100, 500, 1000]
    server_natures = ['heldout', 'ood']
    
    figs = []
    for d, dataset_name in enumerate(dataset_names):
        for s, server_nature in enumerate(server_natures):
            
            server_types = [f'hasnet_{server_nature}_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)']
            
            results_arr = load_results_from_settings(
                [dataset_name], 
                clients_distributions, 
                server_types, 
                keys=keys,
                results_path_local=results_path_local
            )
            
            fig = plt.figure(figsize=(2.5, 2))
            plt.plot(results_arr[d, :, 0, 0], marker='s', label='CA', color='blue')
            plt.plot(results_arr[d, :, 0, 1], marker='^', label='ASR', color='red')
            # plt.xticks(range(len(clients_distribution)))
            plt.yticks(np.arange(0., 1.0, 0.15))
            plt.ylabel('Percentage')
            plt.xlabel('Backdoor Scaling')
            plt.legend()
            plt.tight_layout()
            
            figs.append(fig)
    
    if save_fig:
        with PdfPages('p1_hasnets/__paper__/figures/hyperparameter_backdoor_scaling.pdf') as p:
            for fig in figs:
                fig.savefig(p, format='pdf')
    
    return


def hyperparameter_analysis_clients_ratio(dataset_names, results_path_local: str):
    
    nicer_names = {
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
        'clean': 'Clean', 
        'simple_(poison-0.75)_(scale-2)': 'VTBA',
    }
    
    # start generating figure of hyperparameter analysis
    clients_distributions = [ {'simple_(poison-0.25)_(scale-2)': 0.45} ]
    keys = ['test_acc', 'poisoned_acc']
    clients_ratios = [0.1, 0.2, 0.3, 0.4]
    server_natures = ['heldout']
    
    figs = []
    for d, dataset_name in enumerate(dataset_names):
        for s, server_nature in enumerate(server_natures):
            
            server_types = []
            for clients_ratio in clients_ratios:
                server_types.append(f'hasnet_heldout_(num_clients-100)_(clients_ratio-{clients_ratio})')
                
            results_arr = load_results_from_settings(
                [dataset_name], 
                clients_distributions, 
                server_types, 
                keys=keys,
                results_path_local=results_path_local
            )
            
            for c, clients_distribution in enumerate(clients_distributions):
                fig = plt.figure(figsize=(2.5, 2))
                plt.plot(results_arr[d, c, :, 0], marker='s', label='CA', color='blue')
                plt.plot(results_arr[d, c, :, 1], marker='^', label='ASR', color='red')
                plt.xticks(range(len(clients_ratios)), clients_ratios)
                plt.yticks(np.arange(0., 1.0, 0.15))
                plt.ylabel('Percentage')
                plt.xlabel('Clients Ratio: $c/n$')
                plt.legend()
                plt.tight_layout()
                
                figs.append(fig)
    
    with PdfPages('p1_hasnets/__paper__/figures/hyperparameter_clients_ratio.pdf') as p:
        for fig in figs:
            fig.savefig(p, format='pdf')
    
    return


def ood_correctly_and_randomly_labeled(dataset_names, results_path_local: str):
    
    nicer_names = {
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
        'cifar10_non_sota': 'CIFAR-10', 'gtsrb_non_sota': 'GTSRB',
        'clean': 'Clean', 
        'simple_(poison-0.75)_(scale-2)': 'Simple Backdoor Attack',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\df{} (OOD)',
        'hasnet_ood_random_labelling_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\df{} (OOD Random)',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\df{} (Gen)'
    }
    
    clients_distributions = [
        {},
        {'simple_(poison-0.25)_(scale-2)': 0.45},
        {'invisible_(poison-0.25)_(scale-2)': 0.45},
    ]
    
    server_types = [
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'hasnet_ood_random_labelling_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        # 'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)'
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('c|' + 'c|'*len(clients_distributions))
    
    table_string = ''
    for d, dataset_name in enumerate(dataset_names):
        for s, server_type in enumerate(server_types):
            if s == 0:
                table_string += f'\\multirow{{{len(clients_distributions)}}}{{*}}{{{nicer_names[dataset_name]}}}'
            table_string += ' & {}'.format(nicer_names[server_type])
            for c, clients_distribution in enumerate(clients_distributions):
                result_ = results_arr[d, c, s]
                table_string += ' & {}({})'.format(
                    '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                    '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                )
        
            table_string += ' \\\\\n'
        table_string += '\\hline\n'
    
    return table_string


def _deprecated_generated_noise(dataset_names, results_path_local: str):
    
    nicer_names = {
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
        'clean': 'Clean', 
        'simple_(poison-0.75)_(scale-2)': 'Simple Backdoor Attack',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\df{} (Gen)',
    }
    
    clients_distributions = [
        {},
        {'simple_(poison-0.25)_(scale-2)': 0.45},
        {'invisible_(poison-0.25)_(scale-2)': 0.45},
    ]
    
    server_types = [
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('c|' + 'c|'*len(clients_distributions))
    
    table_string = ''
    for d, dataset_name in enumerate(dataset_names):
        for s, server_type in enumerate(server_types):
            table_string += '{}'.format(nicer_names[server_type])
            for c, clients_distribution in enumerate(clients_distributions):
                
                result_ = results_arr[d, c, s]
                table_string += ' & {}({})'.format(
                    '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                    '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                )
        
            table_string += '\n'
        table_string += '\\\\\n'
    
    return table_string


def special_table(dataset_names, results_path_local: str):
    
    nicer_names = {
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
        'clean': 'Clean', 
        'simple_(poison-0.75)_(scale-2)': 'Simple Backdoor Attack',
        'simple_(num_clients-100)_(clients_ratio-0.1)': 'FedAvg',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'HasNet (Heldout)',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'HasNet (Noise)',
        'dp_(num_clients-100)_(clients_ratio-0.1)': 'DP-SGD',
        'krum_(num_clients-100)_(clients_ratio-0.1)': 'Krum',
        'foolsgold_(num_clients-100)_(clients_ratio-0.1)': 'FoolsGold',
        'deepsight_(num_clients-100)_(clients_ratio-0.1)': 'DeepSight',
        'flame_(num_clients-100)_(clients_ratio-0.1)': 'Flame',
        'hasnet_another': 'Hasnet (Another)'
    }
    
    clients_distributions = [
        {},
        {'simple_(poison-0.25)_(scale-2)': 0.45},
        {'invisible_(poison-0.25)_(scale-2)': 0.45},
        {'neurotoxin_(poison-0.25)_(scale-2)': 0.45},
        {'iba_(poison-0.25)_(scale-2)': 0.45},
    ]
    
    server_types = [
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'hasnet_another'
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(clients_distributions))
    
    table_string = ''
    for s, server_type in enumerate(server_types):
        table_string += '{}'.format(nicer_names[server_type])
        
        for d, dataset_name in enumerate(dataset_names):
            for c, clients_distribution in enumerate(clients_distributions):
                
                result_ = results_arr[d, c, s]
                table_string += ' & {}({})'.format(
                    '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                    '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                )
        
            table_string += '\n'
        table_string += '\\\\\n'
    
    return table_string


def outnumbering_backdoored_clients_analysis(dataset_names, results_path_local: str, save_fig: bool=True):
    
    nicer_names = {
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
        'clean': 'Clean', 
        'simple_(poison-0.75)_(scale-2)': 'VTBA',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'AGSD (ID)',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': 'AGSD (ID-100)',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'AGSD (OOD)',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': 'HGSD (OOD-100)',
    }
    
    # start generating figure of hyperparameter analysis
    clients_distributions = [
        {'simple_(poison-0.25)_(scale-2)': 0.45},
        {'simple_(poison-0.25)_(scale-2)': 0.55},
        {'simple_(poison-0.25)_(scale-2)': 0.65},
        {'simple_(poison-0.25)_(scale-2)': 0.75},
        {'simple_(poison-0.25)_(scale-2)': 0.85},
    ]
    keys = ['test_acc', 'poisoned_acc']
    healing_set_sizes = [50]#, 100]
    server_natures = ['heldout', 'ood']
    server_markers = ['s', '^', 'o', 'x']
    
    figs = []
    for d, dataset_name in enumerate(dataset_names):
        
        fig = plt.figure(figsize=(5, 2.5))
        for s, server_nature in enumerate(server_natures):
            
            server_types = []
            for healing_set_size in healing_set_sizes:
                server_type = f'hasnet_{server_nature}_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-{healing_set_size})'
            
                results_arr = load_results_from_settings(
                    [dataset_name], 
                    clients_distributions, 
                    [server_type], 
                    keys=keys,
                    results_path_local=results_path_local
                )
                
                print(dataset_name, server_type, keys)
                plt.plot(results_arr[d, :, 0, 0], marker=server_markers[s], label=f'CA: {nicer_names[server_type]}')#, color='blue')
                plt.plot(results_arr[d, :, 0, 1], marker=server_markers[s], label=f'ASR: {nicer_names[server_type]}')#, color='red')
            
            plt.xticks(range(len(clients_distributions)), ['{:.2f}'.format(k) for k in np.arange(0.45, 0.45+0.1*len(clients_distributions), 0.1)])
            plt.yticks(np.arange(0., 1.0, 0.15))
            plt.ylabel('Percentage')
            plt.xlabel('$p/n$')
            plt.legend(ncols=2)
            plt.tight_layout()
            
            figs.append(fig)
    
    if save_fig:
        with PdfPages('p1_hasnets/__paper__/figures/outnumbering_backdoored_clients.pdf') as p:
            for fig in figs:
                fig.savefig(p, format='pdf')
    
    return


def non_iid_data_distribution_analysis(dataset_names, results_path_local: str):
    
    nicer_names = {
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'gtsrb': 'GTSRB',
        'clean': 'Clean', 
        'simple_(poison-0.75)_(scale-2)': 'Simple Backdoor Attack',
        'simple_(num_clients-100)_(clients_ratio-0.1)': 'FedAvg',
        'dp_(num_clients-100)_(clients_ratio-0.1)': 'DP-SGD',
        'krum_(num_clients-100)_(clients_ratio-0.1)': 'Krum',
        'foolsgold_(num_clients-100)_(clients_ratio-0.1)': 'FoolsGold',
        'deepsight_(num_clients-100)_(clients_ratio-0.1)': 'DeepSight',
        'flame_(num_clients-100)_(clients_ratio-0.1)': 'Flame',
        'mesas_(num_clients-100)_(clients_ratio-0.1)': 'MESAS',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\dfh{}',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\df{}',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\dfo{}',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\dfh{}',
        'hasnet_noise_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\df{}',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\dfo{}',
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)': '\\dfh{}3',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)': '\\dfo{}3',
    }
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    clients_distributions = [
        {'simple_(poison-0.25)_(scale-2)': 0.45},
    ]
    
    server_types = [
        'hasnet_heldout_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'hasnet_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    ]
    
    keys = ['test_acc', 'poisoned_acc']
    
    results_arr = load_results_from_settings(
        dataset_names, 
        clients_distributions, 
        server_types, 
        keys=keys,
        results_path_local=results_path_local
    )
    # data x client x server x key
    print(results_arr.shape)
    print('|c|' + 'c|'*len(dataset_names)*len(clients_distributions))
    
    table_string = ''
    for s, server_type in enumerate(server_types):
        table_string += '{}'.format(nicer_names[server_type])
        
        for d, dataset_name in enumerate(dataset_names):
            for c, clients_distribution in enumerate(clients_distributions):
                
                result_ = results_arr[d, c, s]
                table_string += ' & {}({})'.format(
                    '{:.2f}'.format(result_[0]) if result_[0] >= 0 else '-',
                    '{:.2f}'.format(result_[1]) if result_[1] >= 0 else '-'
                )
        
            table_string += '\n'
        table_string += '\\\\\n'
    
    return table_string



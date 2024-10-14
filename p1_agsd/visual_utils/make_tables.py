import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from .loading import load_results_from_settings, load_training_information_of_a_setting

from utils_.general_utils import confirm_directory



nicer_names = {
    
    'mnist_toy': 'MNIST', 'cifar10_toy': 'CIFAR-10', 'gtsrb_toy': 'GTSRB',
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
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'AGSD (ID)',
    'not_used_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': '\\df{}',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)': 'AGSD (OOD)',
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': 'AGSD (ID)',
    'not_used_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': '\\df{}',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)': 'AGSD (OOD)',
    'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)': 'AGSD (ID-3)',
    'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)_(n_clusters-3)': 'AGSD (OOD-3)',
    
    'simple_(poison-0.75)_(scale-2)': 'Simple Backdoor Attack',
    
}


def save_figure_multiple_pages(figs: list, save_fig_path_and_name: str):
    
    confirm_directory( '/'.join(save_fig_path_and_name.split('/')[:-1]) )
    
    with PdfPages(save_fig_path_and_name) as p:
            for fig in figs:
                fig.savefig(p, format='pdf')
            print('figure saved.')
    
    return


def comparison_with_sota(dataset_names, results_path_local: str):
    
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
        
        'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        # 'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
        # 'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-100)',
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
    
    # different backdoor clients (one at a time) with 30% backdoor distribution
    clients_distributions = [
        {'simple_(poison-0.25)_(scale-2)': 0.45},
    ]
    
    server_types = [
        'simple_(num_clients-100)_(clients_ratio-0.1)',
        'krum_(num_clients-100)_(clients_ratio-0.1)',
        'flame_(num_clients-100)_(clients_ratio-0.1)',
        'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
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


def hyperparameter_analysis_heldout_set_size(dataset_names, results_path_local: str, figure_name: str='hyperparameter_heldout_set_size', save_fig: bool=True):
    
    # start generating figure of hyperparameter analysis
    clients_distributions = [ {'simple_(poison-0.25)': 0.45} ]
    keys = ['test_acc', 'poisoned_acc']
    healing_set_sizes = [10, 50, 100, 500, 1000]
    server_natures = ['id', 'ood']
    server_markers = ['o', '^']
    
    figs = []
    for d, dataset_name in enumerate(dataset_names):
        
        fig = plt.figure(figsize=(5, 2.5))
        for s, server_nature in enumerate(server_natures):
            
            server_label = f'agsd_{server_nature}_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)'
            server_types = []
            for healing_set_size in healing_set_sizes:
                server_types.append(f'agsd_{server_nature}_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-{healing_set_size})')
            
            results_arr = load_results_from_settings(
                [dataset_name], 
                clients_distributions, 
                server_types, 
                keys=keys,
                results_path_local=results_path_local
            )
            
            for c, clients_distribution in enumerate(clients_distributions):
                plt.plot(results_arr[0, c, :, 0], marker=server_markers[s], label=f'CA: {nicer_names[server_label]}')#, color='blue')
                plt.plot(results_arr[0, c, :, 1], marker=server_markers[s], label=f'ASR: {nicer_names[server_label]}')#, color='red')
            
        plt.xticks(range(len(healing_set_sizes)), healing_set_sizes)
        plt.yticks(np.arange(0., 1.0, 0.15))
        plt.ylabel('Percentage')
        plt.xlabel('Heldout Set Size: $|D_h|$')
        plt.legend(ncols=2)
        plt.tight_layout()
        
        figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{figure_name}.pdf')
    
    return


def hyperparameter_backdoor_scale(dataset_names, results_path_local: str, figure_name: str='hyperparameter_backdoor_scaling', save_fig: bool=True):
    
    # start generating figure of hyperparameter analysis
    clients_distributions = [
        {'simple_(poison-0.25)': 0.45},
        {'simple_(poison-0.25)_(scale-2)': 0.45},
        {'simple_(poison-0.25)_(scale-3)': 0.45},
        {'simple_(poison-0.25)_(scale-5)': 0.45},
    ]
    keys = ['test_acc', 'poisoned_acc']
    server_natures = ['id', 'ood']
    server_markers = ['o', '^']
    
    figs = []
    for d, dataset_name in enumerate(dataset_names):
        
        fig = plt.figure(figsize=(5, 2.5))
        for s, server_nature in enumerate(server_natures):
            
            server_types = [f'agsd_{server_nature}_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)']
            
            results_arr = load_results_from_settings(
                [dataset_name], 
                clients_distributions, 
                server_types, 
                keys=keys,
                results_path_local=results_path_local
            )
            
            plt.plot(results_arr[0, :, 0, 0], marker='s', label=f'CA: {nicer_names[server_types[0]]}')#, color='blue')
            plt.plot(results_arr[0, :, 0, 1], marker='^', label=f'ASR: {nicer_names[server_types[0]]}')#, color='red')
        
        plt.xticks(range(len(clients_distributions)), [1, 2, 3, 5])
        plt.yticks(np.arange(0., 1.0, 0.15))
        plt.ylabel('Percentage')
        plt.xlabel('Backdoor Scaling Constant: $s_b$')
        plt.legend(ncols=2)
        plt.tight_layout()
        
        figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{figure_name}.pdf')
    
    return


def hyperparameter_n_clusters(dataset_names, results_path_local: str, figure_name: str='hyperparameter_n_clusters', save_fig: bool=True):
    
    # start generating figure of hyperparameter analysis
    clients_distributions = [
        {'simple_(poison-0.25)_(scale-2)': 0.45},
    ]
    keys = ['test_acc', 'poisoned_acc']
    # keys = ['simple_(poison-0.25)_(scale-2)_acc_ratio', 'clean_acc_ratio']
    server_natures = ['id', 'ood']
    n_clusters = [2, 3, 4, 5, 6]
    server_markers = ['o', '^']
    
    figs = []
    for d, dataset_name in enumerate(dataset_names):
        
        fig = plt.figure(figsize=(5, 2.5))
        for s, server_nature in enumerate(server_natures):
            
            server_types = []
            for n_cluster in n_clusters:
                str_ = f'agsd_{server_nature}_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)'
                server_types.append(str_ if n_cluster==2 else f'{str_}_(n_clusters-{n_cluster})')
            
            results_arr = load_results_from_settings(
                [dataset_name], 
                clients_distributions, 
                server_types, 
                keys=keys,
                results_path_local=results_path_local
            )
            
            plt.plot(results_arr[0, 0, :, 0], marker='s', label=f'CA: {nicer_names[server_types[0]]}')#, color='blue')
            plt.plot(results_arr[0, 0, :, 1], marker='^', label=f'ASR: {nicer_names[server_types[0]]}')#, color='red')
        
        plt.xticks(range(len(n_clusters)), n_clusters)
        plt.yticks(np.arange(0., 1.0, 0.15))
        plt.ylabel('Percentage')
        plt.xlabel('Number of Clusters')
        plt.legend(ncols=2)
        plt.tight_layout()
        
        figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{figure_name}.pdf')
    
    return


def cost_of_time(dataset_names, results_path_local: str, figure_name: str='cost_of_time', save_fig: bool=True):
    
    # start generating figure of hyperparameter analysis
    clients_distributions = [
        {'simple_(poison-0.25)_(scale-2)': 0.45},
        # {},
    ]
    
    server_types = [
        # SOTA SERVERS
        'simple_(num_clients-100)_(clients_ratio-0.1)',
        # 'dp_(num_clients-100)_(clients_ratio-0.1)',
        'krum_(num_clients-100)_(clients_ratio-0.1)',
        'foolsgold_(num_clients-100)_(clients_ratio-0.1)',
        # 'deepsight_(num_clients-100)_(clients_ratio-0.1)',
        'flame_(num_clients-100)_(clients_ratio-0.1)',
        'mesas_(num_clients-100)_(clients_ratio-0.1)',
        
        # AGSD SERVER ANALYSIS - THIS WILL BE A VERY DETAILED ANALYSIS
        'agsd_id_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
        'agsd_ood_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)',
    ]
    
    server_markers = ['s', '^', 'o', 'x', '1', '2', '3', '+']
    
    values = []
    for d, dataset_name in enumerate(dataset_names):
        
        _values = []
        for s, server_type in enumerate(server_types):
            
            col_name, dict_loaded = load_training_information_of_a_setting(
                dataset_name, 
                clients_distributions[0], 
                server_type, 
                # keys=keys,
                results_path_local=results_path_local,
                verbose=False
            )
            _values.append(np.mean(dict_loaded[col_name+'_time'][-3:]))
        values.append(np.array(_values))
    values = np.array(values)
    
    fig = plt.figure(figsize=(5, 2.5))
    for s, server_type in enumerate(server_types):
        plt.plot(values[:, s], marker=server_markers[s], label=f'{nicer_names[server_type]}')#, color='blue')
        
    plt.xticks(range(len(dataset_names)), [f'{nicer_names[dataset_name]}' for dataset_name in dataset_names])
    # plt.yticks(np.arange(0., 1.0, 0.15))
    plt.ylabel('Average time/round (s)')
    plt.xlabel('Dataset Name')
    plt.legend(ncols=2)
    plt.tight_layout()
    
    if save_fig:
        save_figure_multiple_pages([fig], f'__paper__/figures/{figure_name}.pdf')
            
    return


def hyperparameter_analysis_clients_ratio(dataset_names, results_path_local: str, figure_name: str='hyperparameter_clients_ratio', save_fig: bool=True):
    
    # start generating figure of hyperparameter analysis
    clients_distributions = [ {'simple_(poison-0.25)_(scale-2)': 0.45} ]
    keys = ['test_acc', 'poisoned_acc']
    clients_ratios = [0.1, 0.2, 0.3, 0.4]
    server_natures = ['id']
    server_markers = ['o', '^']
    
    figs = []
    for d, dataset_name in enumerate(dataset_names):
        
        fig = plt.figure(figsize=(5, 2.5))
        for s, server_nature in enumerate(server_natures):
            
            server_label = f'agsd_{server_nature}_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-50)'
            server_types = []
            for clients_ratio in clients_ratios:
                server_types.append(f'agsd_{server_nature}_(num_clients-100)_(clients_ratio-{clients_ratio})')
                
            results_arr = load_results_from_settings(
                [dataset_name], 
                clients_distributions, 
                server_types, 
                keys=keys,
                results_path_local=results_path_local
            )
            
            # for c, clients_distribution in enumerate(clients_distributions):
            plt.plot(results_arr[0, 0, :, 0], marker=server_markers[s], label=f'CA: {nicer_names[server_label]}')#, color='blue')
            plt.plot(results_arr[0, 0, :, 1], marker=server_markers[s], label=f'ASR: {nicer_names[server_label]}')#, color='red')
        
        plt.xticks(range(len(clients_ratios)), clients_ratios)
        plt.yticks(np.arange(0., 1.0, 0.15))
        plt.ylabel('Percentage')
        plt.xlabel('Clients Sampling Ratio: $c/n$')
        plt.legend(ncols=2)
        plt.tight_layout()
        
        figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{figure_name}.pdf')
    
    return


def hyperparameter_backdoored_clients_ratio(dataset_names, results_path_local: str, figure_name: str='hyperparameter_backdoored_clients_ratio', save_fig: bool=True):
    
    # start generating figure of hyperparameter analysis
    clients_distributions = [
        # {'simple_(poison-0.25)_(scale-2)': 0.01},
        {'simple_(poison-0.25)_(scale-2)': 0.05},
        # {'simple_(poison-0.25)_(scale-2)': 0.10},
        {'simple_(poison-0.25)_(scale-2)': 0.15},
        {'simple_(poison-0.25)_(scale-2)': 0.25},
        {'simple_(poison-0.25)_(scale-2)': 0.35},
        {'simple_(poison-0.25)_(scale-2)': 0.45},
        {'simple_(poison-0.25)_(scale-2)': 0.55},
        {'simple_(poison-0.25)_(scale-2)': 0.65},
        {'simple_(poison-0.25)_(scale-2)': 0.75},
        {'simple_(poison-0.25)_(scale-2)': 0.85},
    ]
    keys = ['test_acc', 'poisoned_acc']
    healing_set_sizes = [50]#, 100]
    server_natures = ['id', 'ood']
    server_markers = ['s', '^', 'o', 'x']
    
    figs = []
    for d, dataset_name in enumerate(dataset_names):
        
        fig = plt.figure(figsize=(5, 2.5))
        for s, server_nature in enumerate(server_natures):
            
            server_types = []
            for healing_set_size in healing_set_sizes:
                server_type = f'agsd_{server_nature}_(num_clients-100)_(clients_ratio-0.1)_(healing_set_size-{healing_set_size})'
            
                results_arr = load_results_from_settings(
                    [dataset_name], 
                    clients_distributions, 
                    [server_type], 
                    keys=keys,
                    results_path_local=results_path_local
                )
                
                plt.plot(results_arr[0, :, 0, 0], marker=server_markers[s], label=f'CA: {nicer_names[server_type]}')#, color='blue')
                plt.plot(results_arr[0, :, 0, 1], marker=server_markers[s], label=f'ASR: {nicer_names[server_type]}')#, color='red')
            
            plt.xticks(range(len(clients_distributions)), ['{:.2f}'.format(k) for k in np.arange(0.05, 0.05+0.1*len(clients_distributions), 0.1)])
            plt.yticks(np.arange(0., 1.0, 0.15))
            plt.ylabel('Percentage')
            plt.xlabel('Backdoored clients ratio: $p/n$')
            plt.legend(ncols=2)
            plt.tight_layout()
            
        figs.append(fig)
    
    if save_fig:
        save_figure_multiple_pages(figs, f'__paper__/figures/{figure_name}.pdf')
    
    return



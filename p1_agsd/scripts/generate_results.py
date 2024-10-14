from ..config import results_path

from ..visual_utils.make_tables import comparison_with_sota

from ..visual_utils.adaptive_attack_results import adaptive_attacks_evaluation_dba, adaptive_attacks_evaluation_mtba
from ..visual_utils.adaptive_attack_results import adaptive_attacks_evaluation_lba, adaptive_attacks_evaluation_specifically_desined

from ..visual_utils.make_tables import hyperparameter_backdoored_clients_ratio, hyperparameter_analysis_heldout_set_size, cost_of_time
from ..visual_utils.make_tables import hyperparameter_analysis_clients_ratio, hyperparameter_backdoor_scale, hyperparameter_n_clusters

from ..visual_utils.make_tables import non_iid_analysis



def generate_sota_analysis_tables():
    
    print('\n\nTable 1 of the paper here:')
    print(comparison_with_sota(['mnist'], results_path))
    
    print('\n\nTable 2 of the paper here:')
    print(comparison_with_sota(['cifar10'], results_path))
    
    print('\n\nTable 3 of the paper here:')
    print(comparison_with_sota(['gtsrb'], results_path))
    
    return


def generate_adaptive_analysis_tables():
    print('\n\nTable 5 of the paper here (adaptive backdoor attack results):')
    # print(adaptive_attacks_evaluation_dyba(['gtsrb_non_sota'], results_path))
    print(adaptive_attacks_evaluation_mtba(['gtsrb_non_sota'], results_path))
    print(adaptive_attacks_evaluation_lba(['gtsrb_non_sota'], results_path))
    print(adaptive_attacks_evaluation_dba(['gtsrb_non_sota'], results_path))
    print(adaptive_attacks_evaluation_specifically_desined(['gtsrb_non_sota'], results_path))
    return


def generate_hyperparameter_analysis_figures():
    hyperparameter_analysis_clients_ratio( ['gtsrb_non_sota'], results_path, figure_name='Figure_10_hyperparameter_clients_sampling_ratio', save_fig=True)
    hyperparameter_analysis_heldout_set_size( ['gtsrb_non_sota', 'cifar10_non_sota'], results_path, figure_name='Figure_11_hyperparameter_heldout_set_size', save_fig=True)
    hyperparameter_backdoor_scale( ['gtsrb_non_sota', 'cifar10_non_sota'], results_path, figure_name='Figure_12_hyperparameter_backdoor_scaling', save_fig=True)
    hyperparameter_backdoored_clients_ratio( ['gtsrb_non_sota', 'cifar10_non_sota'], results_path, figure_name='Figure_13_hyperparameter_backdoored_clients_ratio', save_fig=True)
    hyperparameter_n_clusters( ['gtsrb_non_sota', 'cifar10_non_sota'], results_path, figure_name='Figure_14_hyperparameter_n_clusters', save_fig=True)
    cost_of_time( ['mnist_toy', 'cifar10_toy', 'gtsrb_toy'], results_path, figure_name='Figure_15_time_cost', save_fig=True)
    return


def generate_non_iid_results_table():
    
    dataset_types = [
        'gtsrb_non_sota_standard_non_iid1',
        'gtsrb_non_sota_standard_non_iid3',
        'gtsrb_non_sota_standard_non_iid5',
        'gtsrb_non_sota_standard_non_iid7',
        'gtsrb_non_sota_standard_non_iid9',
    ]
    print('\n\nTable 4 of the paper here (non-IID data distribution results):')
    print(non_iid_analysis(dataset_types, results_path))
    
    return


def generate_all_results_and_tables():
    generate_sota_analysis_tables()
    generate_adaptive_analysis_tables()
    try: generate_non_iid_results_table()
    except: pass
    generate_hyperparameter_analysis_figures()
    return


from ..config import results_path

from ..visual_utils.make_tables import comparison_with_sota

from ..visual_utils.adaptive_attack_results import adaptive_attacks_evaluation_dyba, adaptive_attacks_evaluation_mtba
from ..visual_utils.adaptive_attack_results import adaptive_attacks_evaluation_dba
from ..visual_utils.adaptive_attack_results import adaptive_attacks_evaluation_lba, adaptive_attacks_evaluation_specifically_desined

from ..visual_utils.make_tables import hyperparameter_backdoored_clients_ratio, hyperparameter_analysis_heldout_set_size
from ..visual_utils.make_tables import hyperparameter_analysis_clients_ratio, hyperparameter_backdoor_scale

from ..visual_utils.make_tables import non_iid_analysis



def generate_sota_analysis_tables():
    print(comparison_with_sota(['mnist'], results_path))
    print(comparison_with_sota(['cifar10'], results_path))
    print(comparison_with_sota(['gtsrb'], results_path))
    return


def generate_adaptive_analysis_tables():
    # print(adaptive_attacks_evaluation_dyba(['gtsrb_non_sota'], results_path))
    print(adaptive_attacks_evaluation_mtba(['gtsrb_non_sota'], results_path))
    print(adaptive_attacks_evaluation_lba(['gtsrb_non_sota'], results_path))
    print(adaptive_attacks_evaluation_dba(['gtsrb_non_sota'], results_path))
    print(adaptive_attacks_evaluation_specifically_desined(['gtsrb_non_sota'], results_path))
    return


def generate_hyperparameter_analysis_figures():
    hyperparameter_analysis_clients_ratio( ['gtsrb_non_sota'], results_path, save_fig=True)
    hyperparameter_analysis_heldout_set_size( ['gtsrb_non_sota', 'cifar10_non_sota'], results_path, save_fig=True)
    hyperparameter_backdoor_scale( ['gtsrb_non_sota', 'cifar10_non_sota'], results_path, save_fig=True)
    hyperparameter_backdoored_clients_ratio( ['gtsrb_non_sota', 'cifar10_non_sota'], results_path, save_fig=True)
    return


def generate_non_iid_results_table():
    dataset_types = [
        'gtsrb_non_sota_standard_non_iid1',
        'gtsrb_non_sota_standard_non_iid3',
        'gtsrb_non_sota_standard_non_iid5',
        'gtsrb_non_sota_standard_non_iid7',
        'gtsrb_non_sota_standard_non_iid9',
    ]
    print(non_iid_analysis(dataset_types, results_path))
    return


def generate_all_results_and_tables():
    generate_sota_analysis_tables()
    generate_adaptive_analysis_tables()
    generate_non_iid_results_table()
    # generate_hyperparameter_analysis_figures()
    return


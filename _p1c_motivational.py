import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from p1_agsd.motivational_analysis.motivational_script import main
from p1_agsd.motivational_analysis.clustering_observation import main as clustering_observation
from p1_agsd.motivational_analysis.healing_direction import perform_analysis_and_save_figure
from p1_agsd.motivational_analysis.agsd_observations import perform_std_and_conf_analysis



if __name__ == '__main__':
    
    # main()
    
    # clustering_observation()
    
    # perform_analysis_and_save_figure(analysis_type='id')
    # perform_analysis_and_save_figure(analysis_type='ood')
    # perform_analysis_and_save_figure(analysis_type='ood_random')
    # perform_analysis_and_save_figure(analysis_type='noise_random')
    
    dataframe_name='p1_agsd/motivational_analysis/csv_file/std_conf_analysis.csv'
    # perform_std_and_conf_analysis(analysis_type='id', attack_type='fgsm', attack_epsilon=0.2, attack_iterations=1, dataframe_name=dataframe_name)
    # perform_std_and_conf_analysis(analysis_type='ood', attack_type='fgsm', attack_epsilon=0.2, attack_iterations=1, dataframe_name=dataframe_name)
    perform_std_and_conf_analysis(analysis_type='id', attack_type='pgd', attack_epsilon=0.2, attack_iterations=500, dataframe_name=dataframe_name)
    perform_std_and_conf_analysis(analysis_type='ood', attack_type='pgd', attack_epsilon=0.2, attack_iterations=500, dataframe_name=dataframe_name)
    
    
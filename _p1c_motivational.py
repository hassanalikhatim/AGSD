import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from p1_hasnets.motivational_analysis.motivational_script import main
from p1_hasnets.motivational_analysis.healing_direction import perform_analysis_and_save_figure



if __name__ == '__main__':
    
    # main()
    
    perform_analysis_and_save_figure(analysis_type='id')
    perform_analysis_and_save_figure(analysis_type='ood')
    perform_analysis_and_save_figure(analysis_type='ood_random')
    perform_analysis_and_save_figure(analysis_type='noise_random')
    
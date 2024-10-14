# Adversarially Guided Stateful Defense Against Backdoor Attacks in Deep Federated Learning

Recent works have shown that Federated Learning (FL) is vulnerable to backdoor attacks. Existing defenses cluster submitted updates from clients and select the best cluster for aggregation. However, they often rely on unrealistic assumptions regarding client submissions and sampled clients population while choosing the best cluster. We show that in realistic FL settings, state-of-the-art (SOTA) defenses struggle to perform well against backdoor attacks in FL. To address this, we highlight that backdoored submissions are adversarially biased and overconfident compared to clean submissions. We, therefore, propose an Adversarially Guided Stateful Defense (AGSD) against backdoor attacks on Deep Neural Networks (DNNs) in FL scenarios. AGSD employs adversarial perturbations to a small held-out dataset to compute a novel metric, called the trust index, that guides the cluster selection without relying on any unrealistic assumptions regarding client submissions. Moreover, AGSD maintains a trust state history of each client that adaptively penalizes backdoored clients and rewards clean clients. In realistic FL settings, where SOTA defenses mostly fail to resist attacks, AGSD mostly outperforms all SOTA defenses with minimal drop in clean accuracy (5% in the worst-case compared to best accuracy) even when (a) given a very small held-out dataset—typically AGSD assumes 50 samples (≤ 0.1% of the training data) and (b) no held-out dataset is available, and out-of-distribution data is used instead.

## Necessary Instructions

Please note that this artifact can take a months to train all required models to reproduce results on non-GPU machines. Even when several GPUs are available, it can take several weeks as it has to train a new model (e.g. Resnet-18 on GTSRB dataset) for each hyperparameter configuration and backdoor attack. The code is able to execute in multiprocessing mode, where the total number of processes can be controlled by changing the ```shots_at_a_time``` variable in the ```p1_agsd/config.py``` file.


## Setting up your enviornment
1. Download and install miniconda
```
source install_conda.sh
```

2. After restaring the shell, create a conda environment using requirements_agsd.yml and install all dependencies:
```
source install_packages.sh
```

## Downloading pretrained models
To download pretrained models use the following script:
```
source download_pretrained_models.sh
```
All models are not available online due to github limitations. Note that the pretrained models will only be available for a few months.

## Running the code
1. Use p1_agsd/config.py to set your experiment configurations. All standard configurations are already there.

2. If you would like to train your own models, use:
```
python _p1_agsd_main.py --train_models
```

4. Once the models are trained, compile results:
```
python _p1_agsd_main.py --compile_results
```

5. Once the results are compiled, get results tables (latex format) and figures (.pdf) using:
```
python _p1_agsd_main.py --generate_results
```

6. Step 5 will print latex tables in the terminal, and will also save figures in the folder ```__paper__/figures/``` for the hyperparameter analysis in the appendix of the paper.

## Cite as
```
@inproceedings{anonymous2024adversarially,
  title={Adversarially Guided Stateful Defense Against Backdoor Attacks in Deep Federated Learning},
  author={Anonymous et al.},
  booktitle={Proceedings of the 40th annual computer security applications conference},
  year={2024}
}
```


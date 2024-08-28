# Adversarially Guided Stateful Defense Against Backdoor Attacks in Deep Federated Learning

Recent works have shown that Federated Learning (FL) is vulnerable to backdoor attacks. Existing defenses cluster submitted updates from clients and select the best cluster for aggregation. However, they often rely on unrealistic assumptions regarding client submissions and sampled clients population while choosing the best cluster. We show that in realistic FL settings, state-of-the-art (SOTA) defenses struggle to perform well against backdoor attacks in FL. To address this, we highlight that backdoored submissions are adversarially biased and overconfident compared to clean submissions. We, therefore, propose an Adversarially Guided Stateful Defense (AGSD) against backdoor attacks on Deep Neural Networks (DNNs) in FL scenarios. AGSD employs adversarial perturbations to a small held-out dataset to compute a novel metric, called the trust index, that guides the cluster selection without relying on any unrealistic assumptions regarding client submissions. Moreover, AGSD maintains a trust state history of each client that adaptively penalizes backdoored clients and rewards clean clients. In realistic FL settings, where SOTA defenses mostly fail to resist attacks, AGSD mostly outperforms all SOTA defenses with minimal drop in clean accuracy (5% in the worst-case compared to best accuracy) even when (a) given a very small held-out dataset—typically AGSD assumes 50 samples (≤ 0.1% of the training data) and (b) no held-out dataset is available, and out-of-distribution data is used instead.


## Setting up your enviornment
1. Build a conda environment using requirements.txt:
```
conda create --name <env> --file requirements.txt
```

2. Once all the dependencies are installed, activate the environment using:
```
conda activate <env>
```

## Running the code
1. Use p1_agsd/config.py to set your experiment configurations. All standard configurations are already there.

2. If you would like to download the pretrained models (~13GB), use:
```
python _p1_agsd_main.py --download_pretrained_models --train_models
```

3. If you would like to train your own models, use:
```
python _p1_agsd_main.py --train_models
```

4. Once the models are trained, get results tables (latex format) and figures (.pdf) using:
```
python _p1_agsd_main.py --compile_results True --generate_results
```

## Cite as
```
@inproceedings{anonymous2024adversarially,
  title={Adversarially Guided Stateful Defense Against Backdoor Attacks in Deep Federated Learning},
  author={Anonymous et al.},
  booktitle={Proceedings of the 40th annual computer security applications conference},
  year={2024}
}
```
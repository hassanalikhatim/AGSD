mnist_toy_model_configuration = {
    'model_architecture': 'mnist_cnn',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 50,
    'batch_size': 512,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 100,
    'split_type': 'iid'
}
mnist_model_configuration = {
    'model_architecture': 'mnist_cnn',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 500,
    'batch_size': 512,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50,
    'split_type': 'iid'
}

cifar10_model_configuration = {
    'model_architecture': 'cifar10_resnet18',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 1000,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 150,
    'split_type': 'iid'
}
cifar10_model_configuration_non_sota = {
    'model_architecture': 'cifar10_resnet18',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 500,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 70,
    'split_type': 'iid'
}
cifar10_model_configuration_non_sota_standard_non_iid = {
    'model_architecture': 'cifar10_resnet18',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 500,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 70,
    'split_type': 'standard_non_iid'
}
cifar10_model_configuration_non_sota_mesas_non_iid = {
    'model_architecture': 'cifar10_resnet18',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 500,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 70,
    'split_type': 'mesas'
}


gtsrb_model_configuration = {
    'model_architecture': 'resnet18_gtsrb',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 1000,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 100,
    'split_type': 'iid'
}
gtsrb_model_configuration_non_sota = {
    'model_architecture': 'resnet18_gtsrb',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 200,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50,
    'split_type': 'iid'
}
gtsrb_non_sota_standard_non_iid_01 = {
    'model_architecture': 'resnet18_gtsrb',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 200,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50,
    'split_type': 'standard_non_iid',
    'alpha': 0.1
}
gtsrb_non_sota_standard_non_iid_03 = {
    'model_architecture': 'resnet18_gtsrb',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 200,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50,
    'split_type': 'standard_non_iid',
    'alpha': 0.3
}
gtsrb_non_sota_standard_non_iid_05 = {
    'model_architecture': 'resnet18_gtsrb',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 200,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50,
    'split_type': 'standard_non_iid',
    'alpha': 0.5
}
gtsrb_non_sota_standard_non_iid_07 = {
    'model_architecture': 'resnet18_gtsrb',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 200,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50,
    'split_type': 'standard_non_iid',
    'alpha': 0.7
}
gtsrb_non_sota_standard_non_iid_09 = {
    'model_architecture': 'resnet18_gtsrb',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 200,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50,
    'split_type': 'standard_non_iid',
    'alpha': 0.9
}
gtsrb_non_sota_mesas_non_iid = {
    'model_architecture': 'resnet18_gtsrb',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 200,
    'batch_size': 256,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50,
    'split_type': 'mesas_non_iid'
}


model_configurations = {
    'mnist_toy': mnist_toy_model_configuration,
    'mnist': mnist_model_configuration,
    'gtsrb': gtsrb_model_configuration,
    'cifar10': cifar10_model_configuration,
    # non standard settings
    'cifar10_non_sota': cifar10_model_configuration_non_sota,
    'gtsrb_non_sota': gtsrb_model_configuration_non_sota,
    # non standard non iid settings
    'cifar10_non_sota_standard_non_iid': cifar10_model_configuration_non_sota_standard_non_iid,
    'cifar10_non_sota_mesas_non_iid': cifar10_model_configuration_non_sota_mesas_non_iid,
    'gtsrb_non_sota_standard_non_iid1': gtsrb_non_sota_standard_non_iid_01,
    'gtsrb_non_sota_standard_non_iid3': gtsrb_non_sota_standard_non_iid_03,
    'gtsrb_non_sota_standard_non_iid5': gtsrb_non_sota_standard_non_iid_05,
    'gtsrb_non_sota_standard_non_iid7': gtsrb_non_sota_standard_non_iid_07,
    'gtsrb_non_sota_standard_non_iid9': gtsrb_non_sota_standard_non_iid_09,
    'gtsrb_non_sota_mesas_non_iid': gtsrb_non_sota_mesas_non_iid
}

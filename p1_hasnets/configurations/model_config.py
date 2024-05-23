mnist_toy_model_configuration = {
    'model_architecture': 'mnist_cnn',
    'learning_rate': 0.1,
    'loss_fn': 'crossentropy',
    'epochs': 50,
    'batch_size': 512,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'patience': 50
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
    'patience': 50
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
    'patience': 70
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
    'patience': 70
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
    'patience': 70
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
    'patience': 50
}


model_configurations = {
    'mnist_toy': mnist_toy_model_configuration,
    'mnist': mnist_model_configuration,
    'gtsrb': gtsrb_model_configuration,
    'cifar10': cifar10_model_configuration,
    # non standard settings
    'cifar10_non_sota': cifar10_model_configuration_non_sota,
    'gtsrb_non_sota': gtsrb_model_configuration_non_sota
}

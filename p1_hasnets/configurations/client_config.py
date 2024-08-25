# client configurations
local_epochs = 2

clean_client_configuration = {
    'local_epochs': local_epochs
}
simple_backdoor_client_configuration = {
    'local_epochs': local_epochs,
    'trigger': None,
    'target': 0,
    'poison_ratio': 0.8,
    'scale': 1
}
neurotoxin_backdoor_client_configuration = {
    'local_epochs': local_epochs,
    'trigger': None,
    'target': 0,
    'poison_ratio': 0.8,
    'scale': 1,
    'mask_ratio': 0.02
}
iba_backdoor_client_configuration = {
    'local_epochs': local_epochs,
    'trigger': None,
    'target': 0,
    'poison_ratio': 0.8,
    'scale': 1,
    'trigger_inversion_iterations': 500
}
multiple_target_backdoor_client_configuration = {
    'local_epochs': local_epochs,
    'trigger': None,
    'target': 0,
    'poison_ratio': 0.8,
    'scale': 1,
    'num_targets': 4
}
multitrigger_multitarget_backdoor_client_configuration = {
    'local_epochs': local_epochs,
    'trigger': None,
    'target': 0,
    'poison_ratio': 0.8,
    'scale': 1,
    'num_targets': 4
}
class_specific_backdoor_client_configuration = {
    'local_epochs': local_epochs,
    'trigger': None,
    'target': 2,
    'poison_ratio': 0.8,
    'scale': 1,
    'victim_class': [1]
}
low_confidence_backdoor_client_configuration = {
    'local_epochs': local_epochs,
    'trigger': None,
    'target': 0,
    'poison_ratio': 0.8,
    'scale': 1,
    'confidence': 0.4
}
adversarial_training_backdoor_client_configuration = {
    'local_epochs': local_epochs,
    'trigger': None,
    'target': 0,
    'poison_ratio': 0.8,
    'scale': 1,
    'adv_epsilon': 0.2,
    'adv_iterations': 100
}
adversarial_optimization_backdoor_client_configuration = {
    'local_epochs': local_epochs,
    'trigger': None,
    'target': 0,
    'poison_ratio': 0.8,
    'scale': 1,
    'threshold_scaler': 0.1
}
flip_defense_client_configuration = {
    'trigger_inversion_iterations': 200
}

client_configurations = {
    'clean': clean_client_configuration,
    'simple_backdoor': simple_backdoor_client_configuration,
    'invisible_backdoor': simple_backdoor_client_configuration,
    'neurotoxin_backdoor': neurotoxin_backdoor_client_configuration,
    'iba_backdoor': iba_backdoor_client_configuration,
    'multiple_target_backdoor': multiple_target_backdoor_client_configuration,
    'multitrigger_multitarget_backdoor': multitrigger_multitarget_backdoor_client_configuration,
    'flip_defense': flip_defense_client_configuration,
    'visible_backdoor_initially_good': simple_backdoor_client_configuration,
    'class_specific_backdoor': class_specific_backdoor_client_configuration,
    'low_confidence_backdoor': low_confidence_backdoor_client_configuration,
    'distributed_backdoor': simple_backdoor_client_configuration,
    'adv_training_backdoor': adversarial_training_backdoor_client_configuration,
    'adv_optimization_backdoor': adversarial_optimization_backdoor_client_configuration
}


different_clients_configured = {
    'clean': {'type': 'clean'},
    'clean_1': {'type': 'clean', 'local_epochs': 1},
    'clean_5': {'type': 'clean', 'local_epochs': 5},
    
    'simple_(poison-0.25)': {'type': 'simple_backdoor', 'poison_ratio': 0.25, 'scale': 1},
    'simple_(poison-0.50)': {'type': 'simple_backdoor', 'poison_ratio': 0.50, 'scale': 1},
    'simple_(poison-0.75)': {'type': 'simple_backdoor', 'poison_ratio': 0.75, 'scale': 1},
    'simple_(poison-1.00)': {'type': 'simple_backdoor', 'poison_ratio': 1.00, 'scale': 1},
    
    'simple_(poison-0.25)_(scale-2)': {'type': 'simple_backdoor', 'poison_ratio': 0.25, 'scale': 2},
    'simple_(poison-0.50)_(scale-2)': {'type': 'simple_backdoor', 'poison_ratio': 0.50, 'scale': 2},
    'simple_(poison-0.75)_(scale-2)': {'type': 'simple_backdoor', 'poison_ratio': 0.75, 'scale': 2},
    'simple_(poison-1.00)_(scale-2)': {'type': 'simple_backdoor', 'poison_ratio': 1.00, 'scale': 2},
    
    'simple_(poison-0.25)_(scale-3)': {'type': 'simple_backdoor', 'poison_ratio': 0.25, 'scale': 3},
    'simple_(poison-0.50)_(scale-3)': {'type': 'simple_backdoor', 'poison_ratio': 0.50, 'scale': 3},
    'simple_(poison-0.75)_(scale-3)': {'type': 'simple_backdoor', 'poison_ratio': 0.75, 'scale': 3},
    'simple_(poison-1.00)_(scale-3)': {'type': 'simple_backdoor', 'poison_ratio': 1.00, 'scale': 3},
    
    'simple_(poison-0.25)_(scale-5)': {'type': 'simple_backdoor', 'poison_ratio': 0.25, 'scale': 5},
    'simple_(poison-0.50)_(scale-5)': {'type': 'simple_backdoor', 'poison_ratio': 0.50, 'scale': 5},
    'simple_(poison-0.75)_(scale-5)': {'type': 'simple_backdoor', 'poison_ratio': 0.75, 'scale': 5},
    'simple_(poison-1.00)_(scale-5)': {'type': 'simple_backdoor', 'poison_ratio': 1.00, 'scale': 5},
    
    'invisible_(poison-0.25)': {'type': 'invisible_backdoor', 'poison_ratio': 0.25, 'scale': 1},
    'invisible_(poison-0.50)': {'type': 'invisible_backdoor', 'poison_ratio': 0.50, 'scale': 1},
    'invisible_(poison-0.75)': {'type': 'invisible_backdoor', 'poison_ratio': 0.75, 'scale': 1},
    'invisible_(poison-1.00)': {'type': 'invisible_backdoor', 'poison_ratio': 1.00, 'scale': 1},
    
    'neurotoxin_(poison-0.25)': {'type': 'neurotoxin_backdoor', 'poison_ratio': 0.25, 'scale': 1},
    'neurotoxin_(poison-0.50)': {'type': 'neurotoxin_backdoor', 'poison_ratio': 0.50, 'scale': 1},
    'neurotoxin_(poison-0.75)': {'type': 'neurotoxin_backdoor', 'poison_ratio': 0.75, 'scale': 1},
    'neurotoxin_(poison-1.00)': {'type': 'neurotoxin_backdoor', 'poison_ratio': 1.00, 'scale': 1},
    
    'iba_(poison-0.25)': {'type': 'iba_backdoor', 'poison_ratio': 0.25, 'scale': 1},
    'iba_(poison-0.50)': {'type': 'iba_backdoor', 'poison_ratio': 0.50, 'scale': 1},
    'iba_(poison-0.75)': {'type': 'iba_backdoor', 'poison_ratio': 0.75, 'scale': 1},
    'iba_(poison-1.00)': {'type': 'iba_backdoor', 'poison_ratio': 1.00, 'scale': 1},
    
    'class_specific_(poison-0.25)': {'type': 'class_specific_backdoor', 'poison_ratio': 0.25, 'scale': 1},
    'class_specific_(poison-0.50)': {'type': 'class_specific_backdoor', 'poison_ratio': 0.50, 'scale': 1},
    'class_specific_(poison-0.75)': {'type': 'class_specific_backdoor', 'poison_ratio': 0.75, 'scale': 1},
    'class_specific_(poison-1.00)': {'type': 'class_specific_backdoor', 'poison_ratio': 1.00, 'scale': 1},
    
    'invisible_(poison-0.25)_(scale-2)': {'type': 'invisible_backdoor', 'poison_ratio': 0.25, 'scale': 2},
    'invisible_(poison-0.50)_(scale-2)': {'type': 'invisible_backdoor', 'poison_ratio': 0.50, 'scale': 2},
    'invisible_(poison-0.75)_(scale-2)': {'type': 'invisible_backdoor', 'poison_ratio': 0.75, 'scale': 2},
    'invisible_(poison-1.00)_(scale-2)': {'type': 'invisible_backdoor', 'poison_ratio': 1.00, 'scale': 2},
    
    'neurotoxin_(poison-0.25)_(scale-2)': {'type': 'neurotoxin_backdoor', 'poison_ratio': 0.25, 'scale': 2},
    'neurotoxin_(poison-0.50)_(scale-2)': {'type': 'neurotoxin_backdoor', 'poison_ratio': 0.50, 'scale': 2},
    'neurotoxin_(poison-0.75)_(scale-2)': {'type': 'neurotoxin_backdoor', 'poison_ratio': 0.75, 'scale': 2},
    'neurotoxin_(poison-1.00)_(scale-2)': {'type': 'neurotoxin_backdoor', 'poison_ratio': 1.00, 'scale': 2},
    
    'iba_(poison-0.25)_(scale-2)': {'type': 'iba_backdoor', 'poison_ratio': 0.25, 'scale': 2},
    'iba_(poison-0.50)_(scale-2)': {'type': 'iba_backdoor', 'poison_ratio': 0.50, 'scale': 2},
    'iba_(poison-0.75)_(scale-2)': {'type': 'iba_backdoor', 'poison_ratio': 0.75, 'scale': 2},
    'iba_(poison-1.00)_(scale-2)': {'type': 'iba_backdoor', 'poison_ratio': 1.00, 'scale': 2},
    
    'class_specific_(poison-0.25)_(scale-1)': {'type': 'class_specific_backdoor', 'poison_ratio': 0.25, 'scale': 1},
    'class_specific_(poison-0.25)_(scale-2)': {'type': 'class_specific_backdoor', 'poison_ratio': 0.25, 'scale': 2},
    'class_specific_(poison-0.50)_(scale-2)': {'type': 'class_specific_backdoor', 'poison_ratio': 0.50, 'scale': 2},
    'class_specific_(poison-0.75)_(scale-2)': {'type': 'class_specific_backdoor', 'poison_ratio': 0.75, 'scale': 2},
    'class_specific_(poison-1.00)_(scale-2)': {'type': 'class_specific_backdoor', 'poison_ratio': 1.00, 'scale': 2},
    
    'visible_backdoor_initially_good_(poison-0.25)_(scale-1)': {'type': 'visible_backdoor_initially_good', 'poison_ratio': 0.25, 'scale': 1},
    'invisible_(poison-0.25)_(scale-2)_(target-1)': {'type': 'invisible_backdoor', 'poison_ratio': 0.25, 'scale': 2, 'target': 1},
    
    # adaptive backdoor clients
    'low_confidence_(poison-0.25)_(scale-1)': {'type': 'low_confidence_backdoor', 'poison_ratio': 0.25, 'scale': 1},
    'low_confidence_(poison-0.35)_(scale-1)': {'type': 'low_confidence_backdoor', 'poison_ratio': 0.35, 'scale': 1},
    'low_confidence_(poison-0.45)_(scale-1)': {'type': 'low_confidence_backdoor', 'poison_ratio': 0.45, 'scale': 1},
    'low_confidence_(poison-0.55)_(scale-1)': {'type': 'low_confidence_backdoor', 'poison_ratio': 0.55, 'scale': 1},
    'low_confidence_(poison-0.65)_(scale-1)': {'type': 'low_confidence_backdoor', 'poison_ratio': 0.65, 'scale': 1},
    
    'low_confidence_(poison-0.25)_(scale-2)': {'type': 'low_confidence_backdoor', 'poison_ratio': 0.25, 'scale': 2},
    'low_confidence_(poison-0.35)_(scale-2)': {'type': 'low_confidence_backdoor', 'poison_ratio': 0.35, 'scale': 2},
    'low_confidence_(poison-0.45)_(scale-2)': {'type': 'low_confidence_backdoor', 'poison_ratio': 0.45, 'scale': 2},
    'low_confidence_(poison-0.55)_(scale-2)': {'type': 'low_confidence_backdoor', 'poison_ratio': 0.55, 'scale': 2},
    'low_confidence_(poison-0.65)_(scale-2)': {'type': 'low_confidence_backdoor', 'poison_ratio': 0.65, 'scale': 2},
    
    'distributed_(poison-0.25)_(scale-2)': {'type': 'distributed_backdoor', 'poison_ratio': 0.25, 'scale': 2},
    'distributed_(poison-0.35)_(scale-2)': {'type': 'distributed_backdoor', 'poison_ratio': 0.35, 'scale': 2},
    'distributed_(poison-0.45)_(scale-2)': {'type': 'distributed_backdoor', 'poison_ratio': 0.45, 'scale': 2},
    'distributed_(poison-0.55)_(scale-2)': {'type': 'distributed_backdoor', 'poison_ratio': 0.55, 'scale': 2},
    'distributed_(poison-0.65)_(scale-2)': {'type': 'distributed_backdoor', 'poison_ratio': 0.65, 'scale': 2},
    
    
    'adv_training_(poison-0.25)_(scale-1)': {'type': 'adv_training_backdoor', 'poison_ratio': 0.25, 'scale': 1},
    'adv_training_(poison-0.25)_(scale-2)': {'type': 'adv_training_backdoor', 'poison_ratio': 0.25, 'scale': 2},
    
    
    'adv_optimization_(poison-0.25)_(scale-1)': {'type': 'adv_optimization_backdoor', 'poison_ratio': 0.25, 'scale': 1},
    'adv_optimization_(poison-0.25)_(scale-2)': {'type': 'adv_optimization_backdoor', 'poison_ratio': 0.25, 'scale': 2},
    
    
    'multiple_target_(poison-0.25)_(scale-1)': {'type': 'multiple_target_backdoor', 'poison_ratio': 0.25, 'scale': 1},
    'multiple_target_(poison-0.35)_(scale-1)': {'type': 'multiple_target_backdoor', 'poison_ratio': 0.35, 'scale': 1},
    'multiple_target_(poison-0.45)_(scale-1)': {'type': 'multiple_target_backdoor', 'poison_ratio': 0.45, 'scale': 1},
    'multiple_target_(poison-0.55)_(scale-1)': {'type': 'multiple_target_backdoor', 'poison_ratio': 0.55, 'scale': 1},
    'multiple_target_(poison-0.65)_(scale-1)': {'type': 'multiple_target_backdoor', 'poison_ratio': 0.65, 'scale': 1},
    
    'multiple_target_(poison-0.25)_(scale-2)': {'type': 'multiple_target_backdoor', 'poison_ratio': 0.25, 'scale': 2},
    'multiple_target_(poison-0.35)_(scale-2)': {'type': 'multiple_target_backdoor', 'poison_ratio': 0.35, 'scale': 2},
    'multiple_target_(poison-0.45)_(scale-2)': {'type': 'multiple_target_backdoor', 'poison_ratio': 0.45, 'scale': 2},
    'multiple_target_(poison-0.55)_(scale-2)': {'type': 'multiple_target_backdoor', 'poison_ratio': 0.55, 'scale': 2},
    'multiple_target_(poison-0.65)_(scale-2)': {'type': 'multiple_target_backdoor', 'poison_ratio': 0.65, 'scale': 2},
    
    'multitrigger_multitarget_(poison-0.25)_(scale-1)': {'type': 'multitrigger_multitarget_backdoor', 'poison_ratio': 0.25, 'scale': 1},
    'multitrigger_multitarget_(poison-0.35)_(scale-1)': {'type': 'multitrigger_multitarget_backdoor', 'poison_ratio': 0.35, 'scale': 1},
    'multitrigger_multitarget_(poison-0.45)_(scale-1)': {'type': 'multitrigger_multitarget_backdoor', 'poison_ratio': 0.45, 'scale': 1},
    'multitrigger_multitarget_(poison-0.55)_(scale-1)': {'type': 'multitrigger_multitarget_backdoor', 'poison_ratio': 0.55, 'scale': 1},
    'multitrigger_multitarget_(poison-0.65)_(scale-1)': {'type': 'multitrigger_multitarget_backdoor', 'poison_ratio': 0.65, 'scale': 1},
    
    'multitrigger_multitarget_(poison-0.25)_(scale-2)': {'type': 'multitrigger_multitarget_backdoor', 'poison_ratio': 0.25, 'scale': 2},
    'multitrigger_multitarget_(poison-0.35)_(scale-2)': {'type': 'multitrigger_multitarget_backdoor', 'poison_ratio': 0.35, 'scale': 2},
    'multitrigger_multitarget_(poison-0.45)_(scale-2)': {'type': 'multitrigger_multitarget_backdoor', 'poison_ratio': 0.45, 'scale': 2},
    'multitrigger_multitarget_(poison-0.55)_(scale-2)': {'type': 'multitrigger_multitarget_backdoor', 'poison_ratio': 0.55, 'scale': 2},
    'multitrigger_multitarget_(poison-0.65)_(scale-2)': {'type': 'multitrigger_multitarget_backdoor', 'poison_ratio': 0.65, 'scale': 2},
    
}
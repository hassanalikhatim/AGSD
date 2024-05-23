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
flip_defense_client_configuration = {
    'trigger_inversion_iterations': 200
}

client_configurations = {
    'clean': clean_client_configuration,
    'simple_backdoor': simple_backdoor_client_configuration,
    'invisible_backdoor': simple_backdoor_client_configuration,
    'neurotoxin_backdoor': neurotoxin_backdoor_client_configuration,
    'iba_backdoor': iba_backdoor_client_configuration,
    'flip_defense': flip_defense_client_configuration,
    'visible_backdoor_initially_good': simple_backdoor_client_configuration,
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
    
    'visible_backdoor_initially_good_(poison-0.25)_(scale-1)': {'type': 'visible_backdoor_initially_good', 'poison_ratio': 0.25, 'scale': 1}
       
}
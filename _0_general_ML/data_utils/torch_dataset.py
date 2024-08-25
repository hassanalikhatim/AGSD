import torch
import numpy as np



class Torch_Dataset:
    
    def __init__(
        self, 
        data_name: str, 
        preferred_size: int=0
    ):
        
        self.data_name = data_name
        self.preferred_size = preferred_size
        
        self.train = None
        self.test = None
        self.num_classes = None
        
        return
    
    
    def renew_data(self): self.not_implemented()
    def not_implemented(self):
        print_str = 'This is the parent class. Please call the corresponding function '
        print_str += 'of the specific dataset to get the functionality.'
        assert False, print_str
        return
    def get_output_shape(self): return tuple([len(self.get_class_names())])
    def get_input_shape(self): return self.train.__getitem__(0)[0].shape
    def get_class_names(self):  return np.arange(len(np.unique( [self.train[i][1] for i in range(self.train.__len__())] )))
    def get_num_classes(self): return 1+np.max([self.train[i][1] for i in range(self.train.__len__())])
    
    
    def prepare_data_loaders(self, batch_size=64):
        
        self.batch_size = batch_size
        
        train_loader = torch.utils.data.DataLoader(self.train, shuffle=True, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(self.test, shuffle=True, batch_size=batch_size)
        
        return train_loader, test_loader
    
    
    def sample_data(self, data_type, batch_size=64):
        
        sample_size = data_type.__len__()
        if batch_size:
            sample_size = min(sample_size, batch_size)
        
        data_indices = np.random.choice(data_type.__len__(), sample_size, replace=False)
        data_x, data_y = [], []
        for i, ind in enumerate(data_indices):
            print(f'\rSampling data: {i+1}/{len(data_indices)}', end='')
            x, y = data_type.__getitem__(ind)
            data_x.append(x)
            data_y.append(y)
        data_x, data_y = torch.stack(data_x, dim=0), torch.tensor(data_y)
        
        return (data_x, data_y)
    
    
    def view_random_samples(self, n_rows, n_cols):
        
        import matplotlib.pyplot as plt
        
        random_indices = np.random.choice(self.train.__len__(), size=n_rows*n_cols).reshape(n_rows, n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_rows, n_cols))
        for n_r in range(n_rows):
            for n_c in range(n_cols):
                ax = axs[n_r][n_c]
                
                x, y = self.train.__getitem__(random_indices[n_r, n_c])
                
                ax.imshow( np.transpose(x, (1,2,0)) )
                ax.set_title( y )
                ax.set_xticks([])
                ax.set_yticks([])
                
        plt.tight_layout()
        
        return
    
    
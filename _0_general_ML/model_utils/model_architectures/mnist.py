import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    
    def __init__(
        self
    ):
        
        super(Conv2D, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
        return
    
    
    def forward(
        self, x
    ):
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x)
    
    
    
class MNIST_CNN(nn.Module):

    def __init__(
        self, 
        layer_sizes=[(1,20), (20, 30), 3000, 500, 10], 
        activation=None, regularization=None
    ):
        
        super().__init__()
        
        self.layer_sizes = layer_sizes
        
        # Convolution Layer 1                             # 28 x 28 x 1  (input)
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)      # 24 x 24 x 20  (after 1st convolution)
        # Convolution Layer 2
        self.conv2 = nn.Conv2d(20, 30, kernel_size=5)     # 20 x 20 x 30  (after 2nd convolution)
        self.conv2_drop = nn.Dropout2d(p=0.5)             # Same as above
        self.maxpool2 = nn.MaxPool2d(2)                   # 10 x 10 x 30  (after pooling)
        # Fully connected layers
        self.fc1 = nn.Linear(3000, 500)
        self.fc2 = nn.Linear(500, 10)
        # Activation
        self.relu = nn.ReLU()                             # Same as above
        
        return
    

    def forward(self, x):
        
        # Convolution Layer 1                    
        x = self.conv1(x)                        
        x = self.relu(x)                        
        # Convolution Layer 2
        x = self.conv2(x)               
        x = self.conv2_drop(x)
        x = self.maxpool2(x)
        x = self.relu(x)
        # Switch from activation maps to vectors
        x = x.view(-1, 3000)
        # Fully connected layer 1
        x = self.fc1(x)
        x = self.relu(x)
        # Fully connected layer 2
        x = self.fc2(x)
        
        return x


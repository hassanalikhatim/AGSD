from torchvision.models import resnet50, ResNet50_Weights, resnet18
from torch import nn
import torch.nn.functional as F



#adjust resnet50 to my dataset
class Resnet50_GTSRB(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # downloading resent50 pretrained on ImageNet 
        self.rn50 = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1, 
            progress=True
        )
        # self.rn50 = resnet50()
        
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256, 43)
        
        return
    
        
    def forward(self, X):
        
        X = self.rn50(X)
        X = X.view(len(X), -1)
        X = F.relu(self.fl1(X))
        X = F.dropout(X, p=0.25)
        X = self.fl2(X)
        
        return F.log_softmax(X, dim=1)
    

class CNN_GTSRB(nn.Module):

    def __init__(
        self, 
        layer_sizes=[(1,20), (20, 30), 3000, 500, 10], 
        activation=None, regularization=None
    ):
        
        super().__init__()
        
        self.layer_sizes = layer_sizes
        
        # Convolution Layer 1                             # 40 x 40 x 3  (input)
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5)      # 36 x 36 x 20  (after 1st convolution)
        # Convolution Layer 2
        self.conv2 = nn.Conv2d(20, 30, kernel_size=5)     # 32 x 32 x 30  (after 2nd convolution)
        self.conv2_drop = nn.Dropout2d(p=0.5)             # Same as above
        self.maxpool2 = nn.MaxPool2d(2)                   # 16 x 16 x 30  (after pooling)
        # Fully connected layers
        self.fc1 = nn.Linear(7680, 500)
        self.fc2 = nn.Linear(500, 43)
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
        x = x.view(-1, 7680)
        # Fully connected layer 1
        x = self.fc1(x)
        x = self.relu(x)
        # Fully connected layer 2
        x = self.fc2(x)
        
        return x
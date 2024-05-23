from torchvision.models import resnet50, ResNet50_Weights, resnet18
from torch import nn
import torch.nn.functional as F
import torch



#adjust resnet50 to my dataset
class Resnet18_CIFAR10(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # # downloading resent50 pretrained on ImageNet 
        # self.rn50 = resnet18(
        #     weights=ResNet50_Weights.IMAGENET1K_V1, 
        #     progress=True
        # )
        self.rn50 = resnet18()
        
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256, 10)
        
        return
    
        
    def forward(self, X):
        
        X = self.rn50(X)
        X = X.view(len(X), -1)
        X = F.relu(self.fl1(X))
        X = F.dropout(X, p=0.5)
        X = self.fl2(X)
        
        X = F.log_softmax(X, dim=1)
        
        return X
    
    
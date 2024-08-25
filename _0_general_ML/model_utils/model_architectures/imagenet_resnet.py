from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import torch.nn.functional as F
import torch



#adjust resnet50 to my dataset
class Resnet50_Imagenet(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # # downloading resent50 pretrained on ImageNet 
        self.rn50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True)
        
        return
    
        
    def forward(self, X):
        
        X = self.rn50(X)
        
        X = F.log_softmax(X, dim=1)
        
        return X
    
    
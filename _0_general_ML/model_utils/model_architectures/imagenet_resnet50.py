from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

from torch import nn
import torch.nn.functional as F
import torch



class Resnet50_Imagenet(nn.Module):
    
    def __init__(self):
        super().__init__()
        # self.rn50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True)
        self.rn50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=True)
        return
    
        
    def forward(self, X):
        X = self.rn50(X)
        X = F.log_softmax(X, dim=1)
        return X
        


class Resnet18_Imagenet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.rn18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
        return
    
        
    def forward(self, X):
        X = self.rn18(X)
        X = F.log_softmax(X, dim=1)
        return X
    

    
class ViT_B_Imagenet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.rn18 = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1, progress=True)
        return
    
        
    def forward(self, X):
        X = self.rn18(X)
        X = F.log_softmax(X, dim=1)
        return X
    
    
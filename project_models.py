import torch
import torch.nn as nn
from torchvision.models import resnet152

class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96*96*3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)
        )
            
    def forward(self, x): 
        return self.layers(x)
    

class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layers(x)
    
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            BasicConvBlock(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConvBlock(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 6),
        )
            
    def forward(self, x): 
        return self.layers(x)
    

class LayeredConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layers(x)
    
class NaiveDeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            LayeredConvBlock(3, 64),
            LayeredConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            LayeredConvBlock(64, 128),
            LayeredConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            LayeredConvBlock(128, 256),
            LayeredConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            LayeredConvBlock(256, 512),
            LayeredConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            LayeredConvBlock(512, 1024),
            LayeredConvBlock(1024, 1024),
            nn.MaxPool2d(kernel_size=2, stride=2),
            LayeredConvBlock(1024, 2048),
            LayeredConvBlock(2048, 2048),
            nn.MaxPool2d(kernel_size=2, stride=2),
            LayeredConvBlock(2048, 6),
            LayeredConvBlock(6, 6),
            nn.Flatten(),
        )
            
    def forward(self, x): 
        return self.layers(x)        

    
class SkipConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        if in_channels == out_channels:
            self.identity = nn.Identity()
        else:
            self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            
        self.conv = LayeredConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        return self.identity(x) + self.conv(x)

class SkipCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            SkipConvBlock(3, 64),
            SkipConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(64, 128),
            SkipConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(128, 256),
            SkipConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(256, 512),
            SkipConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(512, 1024),
            SkipConvBlock(1024, 1024),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(1024, 2048),
            SkipConvBlock(2048, 2048),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(2048, 6),
            SkipConvBlock(6, 6),
            nn.Flatten(),
        )
            
    def forward(self, x): 
        return self.layers(x)

    
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.f = nn.Conv2d(in_channels, in_channels//8, kernel_size=1, bias=False)
        self.g = nn.Conv2d(in_channels, in_channels//8, kernel_size=1, bias=False)
        self.h = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        input_shape = x.shape
        f = self.f(x).view(input_shape[0], input_shape[1]//8, -1)
        g = self.g(x).view(input_shape[0], input_shape[1]//8, -1)
        h = self.h(x).view(input_shape[0], input_shape[1], -1)
        
        beta = torch.bmm(f.transpose(1,2), g)
        beta = torch.softmax(beta, dim=-1)
   
        res = torch.bmm(h, beta.transpose(1,2))
        res = res.view(input_shape)
        res = self.gamma*res + x
        
        return res
    
class SelfAttentionSkipCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            SkipConvBlock(3, 64),
            SkipConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(64, 128),
            SelfAttentionBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(128, 256),
            SelfAttentionBlock(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(256, 512),
            SelfAttentionBlock(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(512, 1024),
            SelfAttentionBlock(1024),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(1024, 2048),
            SelfAttentionBlock(2048),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SkipConvBlock(2048, 6),
            nn.Flatten(),
        )
            
    def forward(self, x): 
        return self.layers(x)


    
class ResnetModelPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        resnet_backbone = resnet152(weights="DEFAULT")
        for param in resnet_backbone.parameters():
            param.requires_grad = False
        self.layers = nn.Sequential(
            *(list(resnet_backbone.children())[:-1]),
            nn.Flatten(),
            nn.Linear(2048, 6),
        )
    
    def forward(self, x):
        return self.layers(x)
    
class ResnetModelRetrained(nn.Module):
    def __init__(self):
        super().__init__()
        resnet_backbone = resnet152(weights="DEFAULT")
        self.layers = nn.Sequential(
            *(list(resnet_backbone.children())[:-1]),
            nn.Flatten(),
            nn.Linear(2048, 6),
        )
    
    def forward(self, x):
        return self.layers(x)
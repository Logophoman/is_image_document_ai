# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    """
    A very simple CNN for 224x224 input images, 
    outputting 'num_classes' logits.
    """
    def __init__(self, num_classes=2):
        super(TinyCNN, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # A pool layer to reduce spatial size
        self.pool = nn.MaxPool2d(2, 2)
        
        # After two pool ops, 224 -> 112 -> 56
        # So final feature map is [16, 56, 56]
        # Flatten -> 16*56*56 = 50176
        self.fc1 = nn.Linear(16 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))  # shape: [batch, 8, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # shape: [batch, 16, 56, 56]
        
        # Flatten
        x = x.view(x.size(0), -1)            # shape: [batch, 16*56*56=50176]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))              # shape: [batch, 64]
        x = self.fc2(x)                      # shape: [batch, num_classes]
        return x

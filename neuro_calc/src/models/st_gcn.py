import torch
import torch.nn as nn
from src.models.components.agcn_layer import AdaptiveGraphConv
from src.models.components.tcn_layer import TemporalConv

class STGCNBlock(nn.Module):
    """
    The Atomic Unit of Action Recognition.
    Flow: Spatial GCN -> Temporal TCN -> Dropout
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # 1. Spatial Modeling (Adaptive Topology)
        self.gcn = AdaptiveGraphConv(in_channels, out_channels)
        
        # 2. Temporal Modeling (Sequence Analysis)
        self.tcn = TemporalConv(out_channels, out_channels, stride=stride)
        
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (N, C, T, V)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.act(x)

class HandSignRecognizer(nn.Module):
    """
    The Full Network Architecture.
    """
    def __init__(self, num_classes=15): # 0-9 digits + 5 operators
        super().__init__()
        
        # Input: 3 Channels (x, y, z) normalized
        self.data_bn = nn.BatchNorm1d(21 * 3) 

        # Deep Stack
        self.layers = nn.ModuleList([
            STGCNBlock(3, 64),
            STGCNBlock(64, 64),
            STGCNBlock(64, 128, stride=2), # Temporal Pooling
            STGCNBlock(128, 128),
            STGCNBlock(128, 256, stride=2),
            STGCNBlock(256, 256)
        ])
        
        # Classification Head
        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Input x: (N, 3, T, 21)
        N, C, T, V = x.size()
        
        # Normalization (Crucial for variance scaling)
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        # Feature Extraction
        for layer in self.layers:
            x = layer(x)
            
        # Global Pooling (Average over Time and Vertices)
        # We want one prediction per video clip
        # x shape: (N, 256, T_downsampled, V)
        x = nn.functional.avg_pool2d(x, x.shape[2:]) # (N, 256, 1, 1)
        
        # Prediction
        x = self.fcn(x) # (N, Classes, 1, 1)
        return x.view(N, -1)
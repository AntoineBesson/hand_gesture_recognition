import torch
import torch.nn as nn
import numpy as np
from src.graphs.adjacency import HandGraph

class AdaptiveGraphConv(nn.Module):
    """
    Spatial Graph Convolution with Learnable Adjacency Residuals.
    Equation: X_out = Sum_k (A_k + B_k) * X_in * W_k
    """
    def __init__(self, in_channels, out_channels, num_vertices=42, kernel_size=3):
        super().__init__()
        
        self.num_vertices = num_vertices
        
        # 1. Load Physical Graph
        graph = HandGraph(strategy='spatial')
        # Register A as a buffer (not a parameter, so no gradient, but saved in state_dict)
        self.register_buffer('A', torch.tensor(graph.A, dtype=torch.float32, requires_grad=False))
        
        # Kernel size corresponds to the 3 partitions (Self, Inward, Outward)
        self.num_subsets = 3 
        
        # 2. Learnable Topology (The "Adaptive" part)
        # We initialize it to zero so training starts with pure physics
        self.PA = nn.Parameter(torch.zeros(self.num_subsets, num_vertices, num_vertices))
        
        # 3. Feature Transformation (Standard Conv2d effectively acts as FC here)
        # We use Conv2d with kernel (1,1) to implement the Weight matrix W
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * self.num_subsets, 
            kernel_size=(1, 1)
        )
        
        # Batch Norm / ReLU / Dropout standard block
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)  # Changed to inplace=False

    def forward(self, x):
        """
        Input: (Batch, Channels, Time, Nodes/Vertices) -> (N, C, T, V)
        """
        N, C, T, V = x.size()
        
        # 1. Feature Transformation: X * W
        # Result: (N, C_out * 3, T, V)
        x = self.conv(x)
        
        # Reshape to separate the K subsets: (N, K, C_out, T, V)
        x = x.view(N, self.num_subsets, -1, T, V)
        
        # 2. Graph Convolution: A * X
        # We combine Physical A + Learnable PA
        A_total = self.A + self.PA 
        
        # Einstein Summation for graph propagation
        x = torch.einsum('kvw, nkctw->nkctv', A_total, x)
        
        # Sum over the subsets (K dimension)
        x = x.sum(dim=1) # Result: (N, C_out, T, V)
        
        return self.act(self.bn(x))
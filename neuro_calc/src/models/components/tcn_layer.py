import torch
import torch.nn as nn

class TemporalConv(nn.Module):
    """
    Temporal Convolution Module with Residual Connection.
    Architecture: BatchNorm -> ReLU -> Conv(9x1) -> BatchNorm -> Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        
        # Padding calculation to maintain temporal dimension (T)
        # We want "Same" padding behavior
        pad = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1), # (Time, Vertices)
            padding=(pad, 0),             # Pad time only
            stride=(stride, 1)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1, inplace=True)
        
        # Residual Path
        # If input/output channels differ or stride != 1, we need a 1x1 projection
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        """
        Input: (N, C, T, V)
        Output: (N, C, T, V)
        """
        residual = self.downsample(x)
        
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.dropout(x)
        
        return x + residual
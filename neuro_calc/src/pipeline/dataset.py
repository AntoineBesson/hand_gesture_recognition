import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset
from src.core.geometry import vectorize_sequence_canonicalization

class HandGestureDataset(Dataset):
    """
    High-Performance Data Loader for ST-GCN.
    
    Expected Data Structure on Disk:
    /data/processed/
        /class_0_jump/
            seq001.npy (Shape: T x 21 x 3)
            seq002.npy
        /class_1_run/
            ...
    """
    def __init__(self, data_root, window_size=64, mode='train', transform=None):
        """
        Args:
            data_root (str): Path to data directory.
            window_size (int): Fixed temporal dimension T for the network.
            mode (str): 'train' (random sampling) or 'eval' (center sampling).
        """
        self.data_root = data_root
        self.window_size = window_size
        self.mode = mode
        self.transform = transform
        
        # 1. Index the filesystem
        self.samples = [] # List of (path_to_npy, label_int)
        self.classes = sorted(os.listdir(data_root))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_root, cls_name)
            if not os.path.isdir(cls_dir): continue
            
            # Find all .npy files
            files = glob.glob(os.path.join(cls_dir, "*.npy"))
            idx = self.class_to_idx[cls_name]
            for f in files:
                self.samples.append((f, idx))
                
        print(f"[{mode.upper()}] Loaded {len(self.samples)} samples from {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def _temporal_resample(self, sequence):
        """
        Interpolates variable length sequence T_in to fixed length T_out.
        Method: Linear Interpolation via Zoom (scipy) or Uniform Sampling.
        We use Uniform Sampling for speed.
        """
        T_in = sequence.shape[0]
        T_out = self.window_size
        
        if T_in == T_out:
            return sequence
            
        if self.mode == 'train':
            # Random Offset Sampling (Data Augmentation)
            # If video is long, pick a random valid start point? 
            # OR Uniformly sample indices with some jitter?
            # We stick to Uniform Resizing for geometric consistency.
            indices = np.linspace(0, T_in - 1, T_out).astype(int)
        else:
            # Deterministic Sampling (Center)
            indices = np.linspace(0, T_in - 1, T_out).astype(int)
            
        return sequence[indices] # (T_out, 21, 3)

    def __getitem__(self, index):
        path, label = self.samples[index]
        
        # 1. Load Raw Skeleton (T_var, 21, 3)
        # mmap_mode='r' is faster for large datasets, prevents loading full file if we slice immediately
        raw_seq = np.load(path) 
        
        # 2. Temporal Normalization -> (T_fixed, 21, 3)
        resampled_seq = self._temporal_resample(raw_seq)
        
        # 3. Geometric Canonicalization (The "Invariant" step)
        # Projects global coordinates to hand-relative coordinates
        projected_seq = vectorize_sequence_canonicalization(resampled_seq)
        
        # 4. Feature Engineering (Optional)
        # We could add velocity or bone angles here. 
        # For now, we stick to raw XYZ coordinates.
        
        # 5. Tensor Formatting
        # PyTorch expects (Channels, Time, Vertices)
        # Current: (T, V, C)
        tensor_seq = torch.from_numpy(projected_seq).float()
        tensor_seq = tensor_seq.permute(2, 0, 1) # (3, T, 21)
        
        return tensor_seq, label
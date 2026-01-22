import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset
from src.core.geometry import process_dual_hand_frame

class HandGestureDataset(Dataset):
    """
    Dual-Hand Dataset Loader for ST-GCN.
    
    Input Data: .npy files of shape (T_variable, 2, 21, 3)
    Output Tensor: (3, T_fixed, 42) -> (Channels, Time, Vertices)
    """
    def __init__(self, data_root, window_size=64, mode='train', transform=None):
        """
        Args:
            data_root (str): Path to 'data/raw'
            window_size (int): Fixed temporal dimension (e.g., 64)
            mode (str): 'train' (random sampling) or 'eval' (deterministic)
        """
        self.data_root = data_root
        self.window_size = window_size
        self.mode = mode
        self.transform = transform
        
        # 1. Index the filesystem
        self.samples = [] # List of (path_to_npy, label_int)
        
        # Ensure classes are sorted to maintain index consistency
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")
            
        self.classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        if not self.classes:
            raise ValueError(f"No class folders found in {data_root}")
            
        for cls_name in self.classes:
            cls_dir = os.path.join(data_root, cls_name)
            
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
        Input: (T_in, 2, 21, 3)
        Output: (T_out, 2, 21, 3)
        Uniformly resamples the sequence to fit the model's window size.
        """
        T_in = sequence.shape[0]
        T_out = self.window_size
        
        if T_in == T_out:
            return sequence
            
        # Create indices for uniform sampling
        indices = np.linspace(0, T_in - 1, T_out).astype(int)
        
        # Use advanced indexing to resample the time axis (axis 0)
        return sequence[indices]

    def _vectorize_dual_sequence(self, raw_seq):
        """
        Applies geometry processing frame-by-frame.
        Input: (T, 2, 21, 3)
        Output: (T, 42, 3)
        """
        T = raw_seq.shape[0]
        processed_frames = []
        
        for t in range(T):
            # raw_seq[t, 0] is Left Hand (21, 3)
            # raw_seq[t, 1] is Right Hand (21, 3)
            # Both might be zero-padded if missing, handled by geometry.py
            
            frame_unified = process_dual_hand_frame(
                raw_seq[t, 0], 
                raw_seq[t, 1]
            )
            processed_frames.append(frame_unified)
            
        return np.array(processed_frames) # (T, 42, 3)

    def __getitem__(self, index):
        path, label = self.samples[index]
        
        # 1. Load Raw Data (T, 2, 21, 3)
        try:
            raw_seq = np.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a zero tensor in case of corruption (fail gracefully)
            return torch.zeros(3, self.window_size, 42), label
        
        # 2. Temporal Normalization -> (Window_Size, 2, 21, 3)
        resampled_seq = self._temporal_resample(raw_seq)
        
        # 3. Geometric Canonicalization (The "Invariant" step)
        # Result: (Window_Size, 42, 3)
        # This aligns the dual-hand scene to the wrist-midpoint basis
        projected_seq = self._vectorize_dual_sequence(resampled_seq)
        
        # 4. Tensor Formatting
        # PyTorch Conv1d/2d expects (Channels, Time, Vertices)
        # Current: (T, V, C) -> Need (C, T, V)
        tensor_seq = torch.from_numpy(projected_seq).float()
        tensor_seq = tensor_seq.permute(2, 0, 1) # (3, T, 42)
        
        return tensor_seq, label
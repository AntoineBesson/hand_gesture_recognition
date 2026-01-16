from torch.utils.data import DataLoader, random_split
from src.pipeline.dataset import HandGestureDataset

def create_dataloaders(cfg):
    """
    Factory for production-grade DataLoaders.
    Input cfg should come from Hydra: cfg.data.path, cfg.data.batch_size
    """
    full_dataset = HandGestureDataset(
        data_root=cfg.data.path,
        window_size=cfg.model.window_size,
        mode='train'
    )
    
    # 80/20 Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    # Validation set should be deterministic
    # (In a real scenario, you'd want a separate Dataset instance with mode='eval')
    val_set.dataset.mode = 'eval' 

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=4,            # Adjust based on CPU cores
        pin_memory=True,          # Speed up Host-to-Device transfer
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader
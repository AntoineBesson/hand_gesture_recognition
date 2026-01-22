import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast  # Updated import
from omegaconf import DictConfig
from tqdm import tqdm
import os
import logging

# Import our custom architecture
from src.pipeline.loader import create_dataloaders
from src.models.st_gcn import HandSignRecognizer

# Setup Logger
log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg: DictConfig, model, train_loader, val_loader):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"  # Only use AMP on CUDA
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 1. Optimization Strategy
        # AdamW is superior to Adam for deep geometric models (better weight decay handling)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=cfg.training.lr, 
            weight_decay=1e-4
        )
        
        # 2. Learning Rate Scheduler
        # Cosine Annealing without warm restarts allows settling into wide minima
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=cfg.training.epochs
        )
        
        # 3. Loss Function
        # Label Smoothing prevents the model from being "over-confident" on ambiguous frames
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 4. Mixed Precision Scaler (FP16) - Only enabled for CUDA
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.best_acc = 0.0

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}/{self.cfg.training.epochs} [Train]")
        
        for batch_idx, (data, targets) in enumerate(loop):
            # data: (N, 3, T, V)
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Zero Gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward Pass (with Auto-Casting for speed)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(data) # (N, num_classes)
                loss = self.criterion(outputs, targets)
            
            # Backward Pass (Scaled)
            self.scaler.scale(loss).backward()
            
            # Gradient Clipping (Crucial for GCN stability)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        return total_loss / len(self.train_loader), 100.*correct/total

    def validate(self, epoch_idx):
        self.model.eval()
        correct = 0
        total = 0
        
        if self.val_loader is None or len(self.val_loader) == 0:
            log.warning("Validation loader is empty, skipping validation.")
            return 0.0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(data)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        log.info(f"Epoch {epoch_idx} [Val] Accuracy: {acc:.2f}%")
        
        # Checkpointing
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_checkpoint("best_model.pth")
            
        return acc

    def save_checkpoint(self, filename):
        save_path = os.path.join(os.getcwd(), filename)
        torch.save({
            'state_dict': self.model.state_dict(),
            'config': self.cfg,
            'best_acc': self.best_acc
        }, save_path)
        log.info(f"Model saved to {save_path}")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Reproducibility
    torch.manual_seed(cfg.project.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.project.seed)
    
    # 2. Data Pipeline
    log.info("Initializing Data Pipeline...")
    train_loader, val_loader = create_dataloaders(cfg)
    
    if train_loader is None or len(train_loader) == 0:
        log.error("Training loader is empty! Check your data directory.")
        return
    
    # 3. Model Initialization
    log.info("Compiling ST-GCN Architecture...")
    # Calculate num_classes dynamically from config list
    num_classes = len(cfg.classes)
    model = HandSignRecognizer(num_classes=num_classes)
    
    # Log device info
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    log.info(f"Training on device: {device}")
    
    # 4. Training Loop
    trainer = Trainer(cfg, model, train_loader, val_loader)
    
    log.info("Starting Optimization...")
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, train_acc = trainer.train_epoch(epoch)
        val_acc = trainer.validate(epoch)
        
        # Step the scheduler
        trainer.scheduler.step()
        
    log.info(f"Training Complete. Best Accuracy: {trainer.best_acc:.2f}%")

if __name__ == "__main__":
    main()
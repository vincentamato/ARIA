import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
from pathlib import Path
import argparse
import json
from datetime import datetime
import numpy as np

from src.modules.image_encoder import ImageEncoder
from src.data.image_dataset import ImageDataset

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change in monitored value to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop

class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient Loss."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate CCC loss between predictions and targets.
        
        Args:
            pred: Predicted values
            target: Target values
            
        Returns:
            CCC loss (1 - CCC to make it a minimization problem)
        """
        pred = pred.squeeze()
        target = target.squeeze()
        
        mean_pred = torch.mean(pred)
        mean_target = torch.mean(target)
        
        var_pred = torch.var(pred, unbiased=False)
        var_target = torch.var(target, unbiased=False)
        
        covar = torch.mean((pred - mean_pred) * (target - mean_target))
        
        ccc = (2 * covar) / (var_pred + var_target + (mean_pred - mean_target) ** 2)
        
        return 1 - ccc

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function (CCC)
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Dict of training metrics
    """
    model.train()
    total_loss = 0
    total_valence_loss = 0
    total_arousal_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Get batch
        images = batch["image"].to(device)
        valence = batch["valence"].to(device)
        arousal = batch["arousal"].to(device)
        
        # Forward pass
        pred_valence, pred_arousal = model(images)
        
        # Calculate losses
        valence_loss = criterion(pred_valence, valence)
        arousal_loss = criterion(pred_arousal, arousal)
        loss = 0.5 * (valence_loss + arousal_loss)  # Equal weighting
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_valence_loss += valence_loss.item()
        total_arousal_loss += arousal_loss.item()
        
        # Log batch metrics
        wandb.log({
            "batch/train_loss": loss.item(),
            "batch/train_valence_loss": valence_loss.item(),
            "batch/train_arousal_loss": arousal_loss.item(),
            "batch/train_step": epoch * len(train_loader) + batch_idx
        })
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "v_loss": f"{valence_loss.item():.4f}",
            "a_loss": f"{arousal_loss.item():.4f}"
        })
    
    # Calculate average losses
    avg_loss = total_loss / len(train_loader)
    avg_valence_loss = total_valence_loss / len(train_loader)
    avg_arousal_loss = total_arousal_loss / len(train_loader)
    
    return {
        "train/loss": avg_loss,
        "train/valence_loss": avg_valence_loss,
        "train/arousal_loss": avg_arousal_loss
    }

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> dict:
    """Validate the model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function (CCC)
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Dict of validation metrics
    """
    model.eval()
    total_loss = 0
    total_valence_loss = 0
    total_arousal_loss = 0
    
    val_valence_preds = []
    val_valence_targets = []
    val_arousal_preds = []
    val_arousal_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            # Get batch
            images = batch["image"].to(device)
            valence = batch["valence"].to(device)
            arousal = batch["arousal"].to(device)
            
            # Forward pass
            pred_valence, pred_arousal = model(images)
            
            # Calculate losses
            valence_loss = criterion(pred_valence, valence)
            arousal_loss = criterion(pred_arousal, arousal)
            loss = 0.5 * (valence_loss + arousal_loss)  # Equal weighting
            
            # Update metrics
            total_loss += loss.item()
            total_valence_loss += valence_loss.item()
            total_arousal_loss += arousal_loss.item()
            
            # Store predictions and targets for correlation
            val_valence_preds.extend(pred_valence.squeeze().cpu().numpy())
            val_valence_targets.extend(valence.cpu().numpy())
            val_arousal_preds.extend(pred_arousal.squeeze().cpu().numpy())
            val_arousal_targets.extend(arousal.cpu().numpy())
            
            # Log batch metrics
            wandb.log({
                "batch/val_loss": loss.item(),
                "batch/val_valence_loss": valence_loss.item(),
                "batch/val_arousal_loss": arousal_loss.item(),
                "batch/val_step": epoch * len(val_loader) + batch_idx
            })
    
    # Calculate average losses
    avg_loss = total_loss / len(val_loader)
    avg_valence_loss = total_valence_loss / len(val_loader)
    avg_arousal_loss = total_arousal_loss / len(val_loader)
    
    # Calculate correlations (still useful as additional metric)
    valence_corr = np.corrcoef(val_valence_preds, val_valence_targets)[0,1]
    arousal_corr = np.corrcoef(val_arousal_preds, val_arousal_targets)[0,1]
    
    return {
        "val/loss": avg_loss,
        "val/valence_loss": avg_valence_loss,
        "val/arousal_loss": avg_arousal_loss,
        "val/valence_corr": valence_corr,
        "val/arousal_corr": arousal_corr
    }

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_loss: float,
    checkpoint_dir: Path,
    is_best: bool = False
) -> None:
    """Save a checkpoint of the model.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        epoch: Current epoch number
        best_val_loss: Best validation loss so far
        checkpoint_dir: Directory to save checkpoint in
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss
    }
    
    # Save latest checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save numbered checkpoint every checkpoint_freq epochs
    epoch_path = checkpoint_dir / f"epoch_{epoch}.pt"
    torch.save(checkpoint, epoch_path)
    
    # Save best model if this is the best so far
    if is_best:
        best_path = checkpoint_dir / "best.pt"
        torch.save(checkpoint, best_path)

def main(args):
    # Initialize wandb
    wandb.init(
        project="artwork-emotion",
        config=vars(args),
        save_code=True,
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints") / wandb.run.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets and dataloaders
    train_dataset, val_dataset = ImageDataset.get_splits(
        json_path=args.annotations_path,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = ImageEncoder(clip_model_name=args.clip_model)
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # Create loss function (changed from MSE to CCC)
    criterion = CCCLoss()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    # Training loop
    best_val_loss = float("inf")
    start_epoch = 0
    
    # Load checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resuming from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        metrics = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            **train_metrics,
            **val_metrics
        }
        wandb.log(metrics)
        
        # Save checkpoint
        is_best = val_metrics["val/loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val/loss"]
            
        if epoch % args.checkpoint_freq == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                checkpoint_dir=checkpoint_dir,
                is_best=is_best
            )
            
        # Early stopping check
        if early_stopping(val_metrics["val/loss"]):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data args
    parser.add_argument("--annotations_path", type=str, required=True,
                        help="Path to annotations JSON file")
    parser.add_argument("--image_size", type=int, default=336,
                        help="Size to resize images to")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    
    # Model args
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14-336",
                        help="HuggingFace CLIP model name to use")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                        help="How often to save checkpoints")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str,
                        help="Path to checkpoint to resume from")
    
    # Early stopping args
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="Minimum change in validation loss to qualify as an improvement")
    
    args = parser.parse_args()
    main(args)

"""
Training Script for Phase 2: Style-Aware Contrastive Learning.

This script trains the encoder using weighted supervised contrastive learning
to create style-aware representations before cross-encoder fine-tuning.

Usage:
    python scripts/train_contrastive.py --config configs/config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data import TeluguStyleDataset, StyleGraph, create_dataloaders
from scripts.models import ContrastiveModel, ContrastiveModelWithClassifier

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 9
) -> Dict[str, float]:
    """
    Compute representation quality metrics.
    
    Metrics:
    - Alignment: Mean cosine similarity within same class
    - Uniformity: How uniformly distributed embeddings are
    """
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute alignment (same-class similarity)
    alignments = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 1:
            class_emb = embeddings[mask]
            sim_matrix = np.dot(class_emb, class_emb.T)
            # Exclude diagonal
            np.fill_diagonal(sim_matrix, 0)
            n = len(class_emb)
            alignments.append(sim_matrix.sum() / (n * (n - 1) + 1e-8))
    
    alignment = np.mean(alignments) if alignments else 0.0
    
    # Compute uniformity (pairwise Gaussian potential)
    sq_dist = np.sum(embeddings ** 2, axis=1, keepdims=True) + \
              np.sum(embeddings ** 2, axis=1) - 2 * np.dot(embeddings, embeddings.T)
    uniformity = np.log(np.exp(-2 * sq_dist).mean() + 1e-8)
    
    return {
        "alignment": alignment,
        "uniformity": uniformity
    }


class ContrastiveTrainer:
    """Trainer for contrastive learning phase."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training config
        train_cfg = config.get("contrastive", {})
        self.epochs = train_cfg.get("epochs", 15)
        self.lr = train_cfg.get("learning_rate", 2e-5)
        self.warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
        self.weight_decay = train_cfg.get("weight_decay", 0.01)
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 2)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.logging_steps = train_cfg.get("logging_steps", 50)
        self.eval_steps = train_cfg.get("eval_steps", 250)
        self.save_steps = train_cfg.get("save_steps", 500)
        
        # Output directory
        self.output_dir = Path(config["project"]["output_dir"]) / "contrastive"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision
        self.use_amp = config.get("hardware", {}).get("fp16", True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.global_step = 0
        self.best_score = -float('inf')  # Combined metric: alignment - |uniformity|
        self.train_losses = []
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting contrastive training for {self.epochs} epochs")
        logger.info(f"Output directory: {self.output_dir}")
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                leave=True
            )
            
            for step, batch in enumerate(progress):
                loss = self.training_step(batch)
                epoch_loss += loss
                epoch_steps += 1
                
                # Update progress bar
                progress.set_postfix({
                    "loss": f"{loss:.4f}",
                    "avg_loss": f"{epoch_loss / epoch_steps:.4f}"
                })
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    self.train_losses.append(loss)
                
                # Evaluation
                if self.global_step % self.eval_steps == 0 and self.global_step > 0:
                    metrics = self.evaluate()
                    logger.info(
                        f"Step {self.global_step}: "
                        f"alignment={metrics['alignment']:.4f}, "
                        f"uniformity={metrics['uniformity']:.4f}"
                    )
                    
                    # Save best model based on combined score
                    # Good uniformity should be < -0.2 (more negative = better spread)
                    # Penalize if uniformity too close to 0 (collapse)
                    score = metrics['alignment'] + metrics['uniformity']  # uniformity is negative
                    if score > self.best_score and metrics['uniformity'] < -0.05:
                        self.best_score = score
                        self.save_checkpoint("best")
                        logger.info(f"New best score: {score:.4f} (alignment={metrics['alignment']:.4f}, uniformity={metrics['uniformity']:.4f})")
                
                # Periodic saving
                if self.global_step % self.save_steps == 0 and self.global_step > 0:
                    self.save_checkpoint(f"step_{self.global_step}")
                    # Delete old step checkpoints to save disk space (keep only last 2)
                    self._cleanup_old_checkpoints(keep_last=2)
            
            # End of epoch
            avg_loss = epoch_loss / epoch_steps
            logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")
            
            # Evaluate at end of epoch
            metrics = self.evaluate()
            logger.info(
                f"Epoch {epoch + 1} metrics: "
                f"alignment={metrics['alignment']:.4f}, "
                f"uniformity={metrics['uniformity']:.4f}"
            )
            
            # Save epoch checkpoint (skip to save disk space, only keep step checkpoints)
            # self.save_checkpoint(f"epoch_{epoch + 1}")
        
        logger.info("Contrastive training complete!")
        logger.info(f"Best score achieved: {self.best_score:.4f}")
        
        return self.best_score
    
    def training_step(self, batch: dict) -> float:
        """Single training step with gradient accumulation."""
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        similarity_weights = batch.get("similarity_weights")
        if similarity_weights is not None:
            similarity_weights = similarity_weights.to(self.device)
        
        # Forward pass with AMP
        if self.use_amp:
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    similarity_weights=similarity_weights
                )
                loss = outputs["loss"] / self.grad_accum_steps
            
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                similarity_weights=similarity_weights
            )
            loss = outputs["loss"] / self.grad_accum_steps
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.grad_accum_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return loss.item() * self.grad_accum_steps
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate representation quality on validation set."""
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            all_embeddings.append(outputs["projections"].cpu())
            all_labels.append(labels)
        
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        metrics = compute_metrics(embeddings, labels, self.config["model"]["num_labels"])
        
        self.model.train()
        return metrics
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_path = self.output_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(save_path))
        
        # Save training state
        torch.save({
            "global_step": self.global_step,
            "best_score": self.best_score,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "train_losses": self.train_losses
        }, save_path / "training_state.pt")
        
        logger.info(f"Saved checkpoint to {save_path}")
    
    def _cleanup_old_checkpoints(self, keep_last: int = 2):
        """Delete old step checkpoints, keeping only the last N and 'best'."""
        try:
            checkpoint_dirs = [d for d in self.output_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('step_')]
            
            # Sort by step number
            checkpoint_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
            
            # Delete all except the last N
            for old_ckpt in checkpoint_dirs[:-keep_last]:
                import shutil
                shutil.rmtree(old_ckpt)
                logger.info(f"Deleted old checkpoint: {old_ckpt.name}")
        except Exception as e:
            logger.warning(f"Error cleaning up checkpoints: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train contrastive model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config["project"]["seed"])
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Resolve backbone path
    backbone_path = config['model']['backbone']
    if backbone_path.startswith('./'):
        backbone_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', backbone_path[2:]))
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {backbone_path}")
    tokenizer = AutoTokenizer.from_pretrained(backbone_path)
    
    # Create style graph
    style_graph = StyleGraph(
        weak_positive_weight=config["style_graph"]["weak_positive_weight"]
    )
    
    # Create dataloaders
    logger.info(f"Loading data from: {config['data']['train_path']}")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config["data"]["train_path"],
        tokenizer=tokenizer,
        style_graph=style_graph,
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
        max_length=config["data"]["max_length"],
        batch_size=config["contrastive"]["batch_size"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        mode="contrastive",
        num_workers=config["hardware"]["dataloader_num_workers"],
        random_state=config["project"]["seed"]
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating contrastive model")
    model = ContrastiveModelWithClassifier(
        model_name=backbone_path,
        num_labels=config["model"]["num_labels"],
        pooling=config["model"]["pooling"],
        dropout=config["model"]["dropout"],
        temperature=config["contrastive"]["temperature"],
        use_weighted_loss=config["contrastive"].get("use_weighted_loss", True)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = ContrastiveModelWithClassifier.from_pretrained(args.resume)
    
    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    best_score = trainer.train()
    
    logger.info(f"Training complete! Best score: {best_score:.4f}")
    logger.info(f"Best model saved to: {trainer.output_dir / 'best'}")


if __name__ == "__main__":
    main()

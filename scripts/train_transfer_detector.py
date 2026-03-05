"""
Style Transfer Detection Training Script.

Purpose: Train a classifier to predict style based on (Original, Changed) text pairs.
Input: [CLS] Original Text [SEP] Changed Text [SEP]
Task: Multi-class classification (9 styles)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data import TeluguStyleDataset, StyleGraph
from scripts.data.dataset import load_paired_data, split_data

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class StyleTransferDetector(nn.Module):
    """
    BERT-based classifier for Style Transfer Detection.
    Takes tokenized pairs: [CLS] Original [SEP] Changed [SEP]
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 9,
        dropout: float = 0.1,
        pooling: str = "cls",
    ):
        super().__init__()

        self.num_labels = num_labels
        self.pooling = pooling

        # Load backbone
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

        # Check if classifier head exists in the model path (for fine-tuning stage 2)
        head_path = Path(model_name) / "classifier_head.pt"
        if head_path.exists():
            logger.info(f"Loading classifier head from {head_path}")
            state_dict = torch.load(head_path, map_location="cpu")
            self.classifier.load_state_dict(state_dict)

        logger.info(f"Initialized StyleTransferDetector with {num_labels} labels")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Pool
        if self.pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            # Mean pooling
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # Classify
        logits = self.classifier(pooled)
        predictions = logits.argmax(dim=-1)

        result = {"logits": logits, "predictions": predictions}

        return result


class Trainer:
    """Trainer for Style Transfer Detector."""

    def __init__(
        self,
        model: StyleTransferDetector,
        train_dataset: TeluguStyleDataset,
        val_dataset: TeluguStyleDataset,
        config: dict,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device

        # Training config (reuse simple_classifier config structure or create new)
        # We'll use "style_transfer" key if available, else "simple_classifier"
        train_cfg = config.get("style_transfer", config.get("simple_classifier", {}))

        self.epochs = train_cfg.get("epochs", 15)
        self.batch_size = train_cfg.get("batch_size", 8)
        self.lr = float(train_cfg.get("learning_rate", 2e-5))
        self.warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
        self.weight_decay = train_cfg.get("weight_decay", 0.01)
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.eval_steps = train_cfg.get("eval_steps", 200)
        self.patience = train_cfg.get("early_stopping_patience", 5)

        # Create dataloaders
        num_workers = config.get("hardware", {}).get("dataloader_num_workers", 4)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Output directory
        suffix = ""
        if "contrastive" in config["model"]["backbone"]:
            suffix = "_contrastive"
        elif "cross_encoder" in config["model"]["backbone"]:
            suffix = "_cross_encoder"
        elif "dapt" in config["model"]["backbone"]:
            suffix = "_dapt"

        self.output_dir = (
            Path(config["project"]["output_dir"]) / f"style_transfer_detector{suffix}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        label_smoothing = train_cfg.get("label_smoothing", 0.1)

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device), label_smoothing=label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Scheduler
        total_steps = len(self.train_loader) * self.epochs // self.grad_accum_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision
        self.use_amp = (
            config.get("hardware", {}).get("fp16", True) and torch.cuda.is_available()
        )
        self.scaler = GradScaler() if self.use_amp else None

        # State
        self.patience_counter = 0
        self.best_f1 = 0.0
        self.global_step = 0

        self.style_graph = StyleGraph()

    def train(self) -> float:
        """Main training loop."""
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Output directory: {self.output_dir}")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for step, batch in enumerate(progress):
                loss = self.training_step(batch)
                epoch_loss += loss

                progress.set_postfix(
                    {"loss": f"{loss:.4f}", "best_f1": f"{self.best_f1:.4f}"}
                )

                # Periodic evaluation
                if self.global_step % self.eval_steps == 0 and self.global_step > 0:
                    self.evaluate_and_save()

            logger.info(
                f"Epoch {epoch + 1} Avg Loss: {epoch_loss / len(self.train_loader):.4f}"
            )

            # End of epoch evaluation
            self.evaluate_and_save(end_of_epoch=True)

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return self.best_f1

    def training_step(self, batch: dict) -> float:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Check for soft labels
        soft_labels = batch.get("soft_labels")
        if soft_labels is not None:
            soft_labels = soft_labels.to(self.device)

        # BERT takes token_type_ids for pairs usually, checking if tokenizer returned them
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        if self.use_amp:
            with autocast():
                outputs = self.model(
                    input_ids, attention_mask, token_type_ids=token_type_ids
                )

                # Use soft labels if available
                if soft_labels is not None:
                    loss = self.criterion(outputs["logits"], soft_labels)
                else:
                    loss = self.criterion(outputs["logits"], labels)

                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(
                input_ids, attention_mask, token_type_ids=token_type_ids
            )

            # Use soft labels if available
            if soft_labels is not None:
                loss = self.criterion(outputs["logits"], soft_labels)
            else:
                loss = self.criterion(outputs["logits"], labels)

            loss = loss / self.grad_accum_steps
            loss.backward()

        if (self.global_step + 1) % self.grad_accum_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1
        return loss.item() * self.grad_accum_steps

    def evaluate_and_save(self, end_of_epoch=False):
        metrics = self.evaluate(detailed=end_of_epoch)

        if metrics["macro_f1"] > self.best_f1:
            self.best_f1 = metrics["macro_f1"]
            self.save_checkpoint("best")
            logger.info(f"New best Macro-F1: {self.best_f1:.4f}")
            self.patience_counter = 0
        elif end_of_epoch:
            self.patience_counter += 1

    @torch.no_grad()
    def evaluate(self, detailed: bool = False) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            outputs = self.model(
                input_ids, attention_mask, token_type_ids=token_type_ids
            )
            all_preds.extend(outputs["predictions"].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        metrics = {
            "macro_f1": f1_score(all_labels, all_preds, average="macro"),
            "accuracy": accuracy_score(all_labels, all_preds),
        }

        if detailed:
            print("\nEVALUATION RESULTS")
            print(
                classification_report(
                    all_labels,
                    all_preds,
                    target_names=self.style_graph.labels,
                    zero_division=0,
                )
            )

        self.model.train()
        return metrics

    def save_checkpoint(self, name: str):
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.backbone.save_pretrained(save_dir)
        torch.save(self.model.classifier.state_dict(), save_dir / "classifier_head.pt")

        with open(save_dir / "training_config.yaml", "w") as f:
            yaml.dump(
                {
                    "best_f1": self.best_f1,
                    "model_config": {
                        "num_labels": self.model.num_labels,
                        "pooling": self.model.pooling,
                    },
                },
                f,
            )


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load Data (Paired)
    data_path = config["data"]["train_path"]
    if data_path.startswith("./"):
        data_path = str(Path(args.config).parent.parent / data_path[2:])

    # Use config column names needed
    orig_col = config["data"].get("original_column", "ORIGINAL TRANSCRIPTS")
    changed_col = config["data"].get(
        "text_column", "CHANGE STYLE"
    )  # Use this as changed
    label_col = config["data"].get("label_column", "STYLE")

    pairs, labels = load_paired_data(
        data_path,
        original_column=orig_col,
        changed_column=changed_col,
        label_column=label_col,
    )

    # Split
    splits = split_data(
        pairs,
        labels,
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        stratify=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
    style_graph = StyleGraph()

    # Create Datasets
    train_cfg = config.get("style_transfer", config.get("simple_classifier", {}))
    num_crops = train_cfg.get("num_crops_per_sample", 1)

    train_ds = TeluguStyleDataset(
        splits.train_texts,
        splits.train_labels,
        tokenizer,
        style_graph,
        max_length=512,
        mode="style_transfer",
        split="train",
        num_crops_per_sample=num_crops,
    )
    val_ds = TeluguStyleDataset(
        splits.val_texts,
        splits.val_labels,
        tokenizer,
        style_graph,
        max_length=512,
        mode="style_transfer",
        split="val",
    )

    labels_list = splits.train_labels
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels_list), y=labels_list
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    model = StyleTransferDetector(
        model_name=config["model"]["backbone"],
        num_labels=len(style_graph.STYLES),
        dropout=config["model"]["dropout"],
    )

    trainer = Trainer(
        model, train_ds, val_ds, config, class_weights=class_weights, device=device
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)

"""
Training Script for Phase 4: Hard Confusion Refinement.

This script implements the refinement step described in Section 6 of the paper.
1. Loads the trained model from Phase 3.
2. Identifies "hard confusions" (high confidence errors) on the training set.
3. Retrains on these samples with a pairwise penalty loss.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data import TeluguStyleDataset, StyleGraph
from scripts.data.dataset import load_paired_data, split_data
from scripts.train_transfer_detector import StyleTransferDetector
from scripts.models.losses import RefinementLoss, OverlapAwareCrossEntropyLoss

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def identify_hard_confusions(
    model: nn.Module, dataloader: DataLoader, device: str, threshold: float = 0.7
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Identify samples that are misclassified with high confidence.
    Also returns the most frequent confused pairs.
    """
    model.eval()
    hard_confusion_indices = []
    confused_pairs_count = {}  # (true, pred) -> count

    logger.info(f"Identifying hard confusions (threshold={threshold})...")

    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scanning"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(input_ids, attention_mask, token_type_ids=token_type_ids)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = outputs["predictions"]

            # Check each sample in batch
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                prob_pred = probs[i, pred_label].item()

                # If misclassified AND high confidence
                if true_label != pred_label and prob_pred > threshold:
                    hard_confusion_indices.append(global_idx)

                    # Track pair
                    pair = tuple(sorted((true_label, pred_label)))
                    confused_pairs_count[pair] = confused_pairs_count.get(pair, 0) + 1

                global_idx += 1

    logger.info(f"Found {len(hard_confusion_indices)} hard confusion samples.")

    # Select top confused pairs (e.g., top 3)
    sorted_pairs = sorted(
        confused_pairs_count.items(), key=lambda x: x[1], reverse=True
    )
    top_pairs = [p[0] for p in sorted_pairs[:3]]
    logger.info(f"Top confused pairs: {top_pairs}")

    return hard_confusion_indices, top_pairs


def train_refinement(
    model: nn.Module,
    train_loader: DataLoader,
    confusion_pairs: List[Tuple[int, int]],
    config: dict,
    device: str,
):
    """Fine-tune loop for refinement."""
    # Config
    refine_cfg = config.get("refinement", {})
    epochs = refine_cfg.get("epochs", 3)
    lr = float(refine_cfg.get("learning_rate", 1e-5))
    beta = refine_cfg.get("beta", 0.1)

    # Loss
    base_criterion = OverlapAwareCrossEntropyLoss()
    criterion = RefinementLoss(
        base_criterion, beta=beta, confusion_pairs=confusion_pairs
    )

    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    logger.info(f"Starting refinement training for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0

        for batch in tqdm(train_loader, desc=f"Refinement Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]

            # Loss
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            steps += 1

        logger.info(f"Epoch {epoch + 1} Loss: {epoch_loss / steps:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--model_path", type=str, help="Path to Phase 3 model checkpoint"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1. Load Model (Phase 3)
    # If not provided, try to guess
    model_path = args.model_path
    if not model_path:
        model_path = (
            Path(config["project"]["output_dir"]) / "style_transfer_detector/best"
        )

    logger.info(f"Loading model from {model_path}")
    # Initialize model structure
    style_graph = StyleGraph()
    model = StyleTransferDetector(
        model_name=config["model"]["backbone"],
        num_labels=len(style_graph.STYLES),
        dropout=config["model"]["dropout"],
    )

    # Load weights
    # Assuming standard huggingface save or custom save
    # Try loading backbone
    try:
        model.backbone.from_pretrained(model_path)
        # Try loading classifier head
        head_path = Path(model_path) / "classifier_head.pt"
        if head_path.exists():
            model.classifier.load_state_dict(torch.load(head_path, map_location=device))
    except Exception as e:
        logger.warning(
            f"Could not load full model from path: {e}. Ensure path is correct."
        )

    model.to(device)

    # 2. Load Training Data
    # We need to scan the training set to find hard confusions
    data_path = config["data"]["train_path"]
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    # Load pairs
    orig_col = config["data"].get("original_column", "ORIGINAL TRANSCRIPTS")
    changed_col = config["data"].get("text_column", "CHANGE STYLE")
    label_col = config["data"].get("label_column", "STYLE")

    pairs, labels = load_paired_data(data_path, orig_col, changed_col, label_col)
    splits = split_data(pairs, labels, stratify=True)

    train_dataset = TeluguStyleDataset(
        splits.train_texts,
        splits.train_labels,
        tokenizer,
        style_graph,
        max_length=512,
        mode="style_transfer",
        split="train",
        num_crops_per_sample=1,  # No cropping for scanning
    )

    scan_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # 3. Identify Hard Confusions
    indices, confusion_pairs = identify_hard_confusions(model, scan_loader, device)

    if not indices:
        logger.info("No hard confusions found. Skipping refinement.")
        return

    # 4. Create Refinement Dataset
    refinement_dataset = Subset(train_dataset, indices)
    refine_loader = DataLoader(refinement_dataset, batch_size=8, shuffle=True)

    # 5. Retrain
    train_refinement(model, refine_loader, confusion_pairs, config, device)

    # 6. Save Refined Model
    output_dir = Path(config["project"]["output_dir"]) / "refined_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.backbone.save_pretrained(str(output_dir))
    torch.save(model.classifier.state_dict(), output_dir / "classifier_head.pt")
    logger.info(f"Refined model saved to {output_dir}")


if __name__ == "__main__":
    main()

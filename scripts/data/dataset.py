"""
Telugu Style Dataset Module.
Handles data loading, preprocessing, and DataLoader creation.

Supports three modes:
1. classification: Standard text -> label
2. contrastive: Text with similarity weights for SupCon
3. cross_encoder: Text paired with all 9 style hypotheses
"""

import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, PreTrainedTokenizer
from sklearn.model_selection import train_test_split

from .preprocessing import TeluguPreprocessor
from .style_graph import StyleGraph

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Container for train/val/test splits."""

    train_texts: List[Union[str, Tuple[str, str]]]
    train_labels: List[str]
    val_texts: List[Union[str, Tuple[str, str]]]
    val_labels: List[str]
    test_texts: List[Union[str, Tuple[str, str]]]
    test_labels: List[str]


class TeluguStyleDataset(Dataset):
    """
    Dataset for Telugu style classification.

    Supports multiple modes for different training phases:
    - classification: Standard single-label classification
    - contrastive: Returns similarity weights for SupCon loss
    - cross_encoder: Pairs text with all style hypotheses
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer: PreTrainedTokenizer,
        style_graph: StyleGraph,
        max_length: int = 256,
        mode: str = "classification",
        split: str = "train",
        hypothesis_language: str = "english",
        preprocessor: Optional[TeluguPreprocessor] = None,
        num_crops_per_sample: int = 1,
    ):
        """
        Initialize dataset.

        Args:
            texts: List of Telugu texts
            labels: List of style labels
            tokenizer: HuggingFace tokenizer
            style_graph: StyleGraph instance for label encoding
            max_length: Maximum sequence length for tokenization
            mode: One of "classification", "contrastive", "cross_encoder"
            split: "train", "val", or "test" (determines cropping strategy)
            hypothesis_language: "english", "telugu", or "both"
            preprocessor: Optional TeluguPreprocessor instance
        """
        assert len(texts) == len(labels), "Texts and labels must have same length"
        assert mode in [
            "classification",
            "contrastive",
            "cross_encoder",
            "style_transfer",
        ], f"Invalid mode: {mode}"

        self.tokenizer = tokenizer
        self.style_graph = style_graph
        self.max_length = max_length
        self.mode = mode
        self.split = split
        self.hypothesis_language = hypothesis_language
        self.preprocessor = preprocessor or TeluguPreprocessor()

        # Multi-crop training: augment training data by extracting multiple crops
        # Only apply to training split; val/test use single crop
        self.num_crops_per_sample = num_crops_per_sample if split == "train" else 1

        # Preprocess texts
        # If texts are tuples (style_transfer mode), we defer preprocessing to __getitem__
        # to avoid complexity here or unpacking the whole list.
        # If texts are strings (classification/contrastive), we can preprocess here.
        if texts and isinstance(texts[0], tuple):
            self.texts = texts  # Store as is
        else:
            self.texts = [self.preprocessor(t) for t in texts]
        self.labels = labels

        # Convert labels to indices
        self.label_indices = [style_graph.label_to_idx(l) for l in labels]

        # Validate labels
        invalid_count = sum(1 for idx in self.label_indices if idx < 0)
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} samples with invalid labels")

        # Cache hypotheses for cross-encoder mode
        if mode == "cross_encoder":
            self.hypotheses = style_graph.get_hypotheses_list(hypothesis_language)

        logger.info(
            f"Created TeluguStyleDataset: {len(self)} samples, "
            f"mode={mode}, max_length={max_length}"
        )

    def __len__(self) -> int:
        # Multi-crop: expand dataset size by num_crops for training
        return len(self.texts) * self.num_crops_per_sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        # Multi-crop: map global idx to (sample_idx, crop_idx)
        sample_idx = idx // self.num_crops_per_sample
        crop_idx = idx % self.num_crops_per_sample

        text = self.texts[sample_idx]
        label = self.labels[sample_idx]  # Original line: label = self.labels[idx]
        label_idx = self.label_indices[
            sample_idx
        ]  # Original line: label_idx = self.label_indices[idx]

        if self.mode == "classification":
            return self._get_classification_item(text, label_idx)
        elif self.mode == "contrastive":
            return self._get_contrastive_item(text, label, label_idx)
        elif self.mode == "cross_encoder":
            return self._get_cross_encoder_item(text, label_idx)
        elif self.mode == "style_transfer":
            # In this mode, text is expected to be a tuple (original, changed)
            return self._get_style_transfer_item(text, label_idx)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_classification_item(
        self, text: str, label_idx: int
    ) -> Dict[str, torch.Tensor]:
        # Standard classification format with RANDOM/CENTER CROPPING
        # Dataset analysis showed intros are duplicates and otros are generic noise.
        # We need the MIDDLE content.

        # Tokenize without truncation first
        inputs = self.tokenizer(
            text,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        input_ids = inputs["input_ids"]
        total_len = len(input_ids)
        target_len = self.max_length - 2  # Reserve space for [CLS] and [SEP]

        start_idx = 0
        end_idx = total_len

        if total_len > target_len:
            if self.split == "train":
                # Random Crop for training (Data Augmentation + Avoid Edges)
                # Avoid the very first 10% (Intro) and very last 10% (Outro)
                start_margin = int(total_len * 0.1)
                end_margin = int(total_len * 0.9)

                # Make sure margins leave enough space
                if end_margin - start_margin < target_len:
                    # Fallback to full range random
                    max_start = total_len - target_len
                    start_idx = torch.randint(0, max_start + 1, (1,)).item()
                else:
                    max_start = end_margin - target_len
                    # Ensure range is valid
                    if max_start <= start_margin:
                        start_idx = start_margin
                    else:
                        start_idx = torch.randint(
                            start_margin, max_start + 1, (1,)
                        ).item()
            else:
                # Center Crop for validation/test (Deterministic)
                start_idx = (total_len - target_len) // 2

            end_idx = start_idx + target_len

        # Slice
        input_ids = input_ids[start_idx:end_idx]

        # Add special tokens manually ([CLS] ... [SEP])
        special_ids = []
        if self.tokenizer.cls_token_id is not None:
            special_ids.append(self.tokenizer.cls_token_id)

        input_ids = special_ids + input_ids

        if self.tokenizer.sep_token_id is not None:
            input_ids = input_ids + [self.tokenizer.sep_token_id]

        # Create attention mask (1s for real tokens)
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_idx, dtype=torch.long),
        }

    def _get_contrastive_item(
        self, text: str, label: str, label_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Contrastive learning format with similarity weights."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Get similarity weights for this label to all other labels
        similarity_weights = self.style_graph.get_similarity_vector(label)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_idx, dtype=torch.long),
            "similarity_weights": torch.tensor(similarity_weights, dtype=torch.float32),
        }

    def _get_cross_encoder_item(
        self, text: str, label_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Cross-encoder format: text paired with ALL 9 style hypotheses.

        Returns:
            Dict with:
            - input_ids: (9, seq_len)
            - attention_mask: (9, seq_len)
            - token_type_ids: (9, seq_len) if available
            - labels: scalar (gold label index)
            - soft_labels: (9,) soft target distribution
        """
        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []

        # Create text-hypothesis pairs for all 9 styles
        for hypothesis in self.hypotheses:
            encoding = self.tokenizer(
                text,
                hypothesis,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            all_input_ids.append(encoding["input_ids"])
            all_attention_masks.append(encoding["attention_mask"])

            if "token_type_ids" in encoding:
                all_token_type_ids.append(encoding["token_type_ids"])

        # Stack into (9, seq_len) tensors
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)

        # Get soft target distribution
        gold_label = self.style_graph.idx_to_label(label_idx)
        soft_labels = self.style_graph.get_soft_target(gold_label, temperature=1.0)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_idx, dtype=torch.long),
            "soft_labels": torch.tensor(soft_labels, dtype=torch.float32),
        }

        if all_token_type_ids:
            result["token_type_ids"] = torch.cat(all_token_type_ids, dim=0)

        return result

    def _get_style_transfer_item(
        self, text_pair: Union[Tuple[str, str], str], label_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Style Transfer Detection format.
        Input: [CLS] Original Text [SEP] Changed Text [SEP]
        """
        if isinstance(text_pair, str):
            # Fallback if somehow a string is passed (e.g. at inference if single input)
            original = ""
            changed = text_pair
        else:
            original, changed = text_pair

        # Preprocess both
        if self.preprocessor:
            original = self.preprocessor(original)
            changed = self.preprocessor(changed)

        # ---------------------------------------------------------
        # 1. Multi-Crop Augmentation (Random Windowing)
        # ---------------------------------------------------------
        if self.split == "train" and self.num_crops_per_sample > 1:
            # Check lengths (approximate via spaces)
            orig_words = original.split()
            changed_words = changed.split()

            # Only crop if combined length is significant (> 60% of max_len)
            if len(orig_words) + len(changed_words) > self.max_length * 0.6:
                # Randomly drop words from the beginning (0-30%)
                drop_pct = torch.rand(1).item() * 0.3

                start_orig = int(len(orig_words) * drop_pct)
                start_changed = int(len(changed_words) * drop_pct)

                original = " ".join(orig_words[start_orig:])
                changed = " ".join(changed_words[start_changed:])

        # Tokenize pair
        encoding = self.tokenizer(
            original,
            changed,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # ---------------------------------------------------------
        # 2. Soft-Target Generation (Style Neighborhoods)
        # ---------------------------------------------------------
        gold_label = self.style_graph.idx_to_label(label_idx)
        # Use a moderate temperature to soften the distribution
        soft_labels = self.style_graph.get_soft_target(gold_label, temperature=1.0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_idx, dtype=torch.long),
            "soft_labels": torch.tensor(soft_labels, dtype=torch.float32),
        }

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in dataset."""
        from collections import Counter

        return dict(Counter(self.labels))

    def set_mode(self, mode: str):
        """Change dataset mode."""
        assert mode in ["classification", "contrastive", "cross_encoder"]
        self.mode = mode
        if mode == "cross_encoder" and not hasattr(self, "hypotheses"):
            self.hypotheses = self.style_graph.get_hypotheses_list(
                self.hypothesis_language
            )


def load_data(
    data_path: str, text_column: str = "CHANGE STYLE", label_column: str = "STYLE"
) -> Tuple[List[str], List[str]]:
    """
    Load data from CSV file.

    Args:
        data_path: Path to CSV or Excel file (.csv, .xlsx, .xls)
        text_column: Name of text column
        label_column: Name of label column

    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Loading data from {data_path}")

    # Support both CSV and Excel files
    file_ext = os.path.splitext(data_path)[1].lower()
    if file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    # Handle potential column name variations
    text_col = None
    label_col = None

    for col in df.columns:
        if col.lower().strip() == text_column.lower().strip():
            text_col = col
        if col.lower().strip() == label_column.lower().strip():
            label_col = col

    if text_col is None:
        raise ValueError(
            f"Text column '{text_column}' not found. Available: {list(df.columns)}"
        )
    if label_col is None:
        raise ValueError(
            f"Label column '{label_column}' not found. Available: {list(df.columns)}"
        )

    # Drop rows with missing values
    df = df.dropna(subset=[text_col, label_col])

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    logger.info(f"Loaded {len(texts)} samples")

    return texts, labels


def load_paired_data(
    data_path: str,
    original_column: str = "ORIGINAL TRANSCRIPTS",
    changed_column: str = "CHANGE STYLE",
    label_column: str = "STYLE",
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Load paired data (Original, Changed) from CSV or Excel file.

    Args:
        data_path: Path to file
        original_column: Name of original text column
        changed_column: Name of changed text column
        label_column: Name of label column

    Returns:
        Tuple of (list_of_pairs, labels) where pairs are (original, changed)
    """
    logger.info(f"Loading paired data from {data_path}")

    # Support both CSV and Excel files
    file_ext = os.path.splitext(data_path)[1].lower()
    if file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    # Handle potential column name variations
    orig_col = None
    changed_col = None
    label_col = None

    for col in df.columns:
        c = col.lower().strip()
        if c == original_column.lower().strip():
            orig_col = col
        if c == changed_column.lower().strip():
            changed_col = col
        if c == label_column.lower().strip():
            label_col = col

    if not all([orig_col, changed_col, label_col]):
        missing = []
        if not orig_col:
            missing.append(original_column)
        if not changed_col:
            missing.append(changed_column)
        if not label_col:
            missing.append(label_column)
        raise ValueError(f"Columns {missing} not found. Available: {list(df.columns)}")

    # Drop rows with missing values
    df = df.dropna(subset=[orig_col, changed_col, label_col])

    # Sort to ensure deterministic order (crucial for random split stability)
    df = df.sort_index()

    original_texts = df[orig_col].astype(str).tolist()
    changed_texts = df[changed_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    # Zip into pairs
    pairs = list(zip(original_texts, changed_texts))

    logger.info(f"Loaded {len(pairs)} pairs")
    return pairs, labels


def split_data(
    texts: List[Union[str, Tuple[str, str]]],
    labels: List[str],
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> DataSplit:
    """
    Split data into train/val/test sets.

    Args:
        texts: List of texts
        labels: List of labels
        val_split: Fraction for validation
        test_split: Fraction for test
        random_state: Random seed
        stratify: Whether to stratify by label

    Returns:
        DataSplit with train/val/test texts and labels
    """
    stratify_labels = labels if stratify else None

    # First split: train+val vs test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_split,
        random_state=random_state,
        stratify=stratify_labels,
    )

    # Second split: train vs val
    val_ratio = val_split / (1 - test_split)
    stratify_labels = train_val_labels if stratify else None

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_labels,
    )

    logger.info(
        f"Data split: train={len(train_texts)}, "
        f"val={len(val_texts)}, test={len(test_texts)}"
    )

    return DataSplit(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
    )


def create_dataloaders(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    style_graph: StyleGraph,
    text_column: str = "CHANGE STYLE",
    label_column: str = "STYLE",
    max_length: int = 256,
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    mode: str = "classification",
    hypothesis_language: str = "english",
    num_workers: int = 4,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        data_path: Path to CSV data file
        tokenizer: HuggingFace tokenizer
        style_graph: StyleGraph instance
        text_column: Name of text column in CSV
        label_column: Name of label column in CSV
        max_length: Max sequence length
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        mode: Dataset mode (classification/contrastive/cross_encoder)
        hypothesis_language: Language for cross-encoder hypotheses
        num_workers: DataLoader workers
        random_state: Random seed

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load data
    texts, labels = load_data(data_path, text_column, label_column)

    # Split data
    split = split_data(
        texts,
        labels,
        val_split=val_split,
        test_split=test_split,
        random_state=random_state,
    )

    # Create preprocessor
    preprocessor = TeluguPreprocessor()

    # Create datasets
    train_dataset = TeluguStyleDataset(
        texts=split.train_texts,
        labels=split.train_labels,
        tokenizer=tokenizer,
        style_graph=style_graph,
        max_length=max_length,
        mode=mode,
        split="train",
        hypothesis_language=hypothesis_language,
        preprocessor=preprocessor,
    )

    val_dataset = TeluguStyleDataset(
        texts=split.val_texts,
        labels=split.val_labels,
        tokenizer=tokenizer,
        style_graph=style_graph,
        max_length=max_length,
        mode=mode,
        split="val",
        hypothesis_language=hypothesis_language,
        preprocessor=preprocessor,
    )

    test_dataset = TeluguStyleDataset(
        texts=split.test_texts,
        labels=split.test_labels,
        tokenizer=tokenizer,
        style_graph=style_graph,
        max_length=max_length,
        mode=mode,
        split="test",
        hypothesis_language=hypothesis_language,
        preprocessor=preprocessor,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def create_datasets(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    style_graph: StyleGraph,
    text_column: str = "CHANGE STYLE",
    label_column: str = "STYLE",
    max_length: int = 256,
    val_split: float = 0.1,
    test_split: float = 0.1,
    mode: str = "classification",
    hypothesis_language: str = "english",
    random_state: int = 42,
    num_crops_per_sample: int = 1,
) -> Tuple[TeluguStyleDataset, TeluguStyleDataset, TeluguStyleDataset]:
    """
    Create train/val/test datasets (without DataLoader wrapping).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load and split data
    texts, labels = load_data(data_path, text_column, label_column)
    split = split_data(texts, labels, val_split, test_split, random_state)

    preprocessor = TeluguPreprocessor()

    train_dataset = TeluguStyleDataset(
        texts=split.train_texts,
        labels=split.train_labels,
        tokenizer=tokenizer,
        style_graph=style_graph,
        max_length=max_length,
        mode=mode,
        split="train",
        hypothesis_language=hypothesis_language,
        preprocessor=preprocessor,
        num_crops_per_sample=num_crops_per_sample,
    )

    val_dataset = TeluguStyleDataset(
        texts=split.val_texts,
        labels=split.val_labels,
        tokenizer=tokenizer,
        style_graph=style_graph,
        max_length=max_length,
        mode=mode,
        split="val",
        hypothesis_language=hypothesis_language,
        preprocessor=preprocessor,
    )

    test_dataset = TeluguStyleDataset(
        texts=split.test_texts,
        labels=split.test_labels,
        tokenizer=tokenizer,
        style_graph=style_graph,
        max_length=max_length,
        mode=mode,
        split="test",
        hypothesis_language=hypothesis_language,
        preprocessor=preprocessor,
    )

    return train_dataset, val_dataset, test_dataset

"""
Domain Adaptive Pre-Training (DAPT) for Telugu Style Classification.

Continue pre-training IndicBERT on Telugu domain corpus using Masked Language Modeling (MLM).
This adapts the model to Telugu text patterns before fine-tuning on style classification.
"""

import torch
import pandas as pd
import logging
from pathlib import Path
from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_telugu_corpus(data_path: str, text_column: str, include_style_data: bool = True) -> List[str]:
    """Load Telugu corpus from CSV and optionally add style classification data."""
    logger.info(f"Loading Telugu corpus from {data_path}")
    
    # Load main DAPT corpus
    df = pd.read_csv(data_path)
    texts = df[text_column].dropna().tolist()
    logger.info(f"Loaded {len(texts)} texts from DAPT corpus")
    
    # Optionally add style classification training data for domain adaptation
    if include_style_data:
        try:
            from scripts.data.dataset import load_data
            style_texts, _ = load_data(
                "database/PR_train_cleaned_v2.xlsx",
                "CHANGE STYLE",
                "STYLE"
            )
            texts.extend(style_texts)
            logger.info(f"Added {len(style_texts)} texts from style classification dataset")
        except Exception as e:
            logger.warning(f"Could not load style data: {e}")
    
    logger.info(f"Total corpus size: {len(texts)} texts")
    return texts


def prepare_mlm_dataset(texts: List[str], tokenizer, max_length: int = 96):
    """Prepare dataset for MLM training."""
    logger.info(f"Tokenizing {len(texts)} texts with max_length={max_length}")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        # Tokenize without padding (DataCollator will handle that)
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    return tokenized


def train_dapt(config: dict):
    """Run DAPT training."""
    dapt_cfg = config["dapt"]
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    output_dir = Path(dapt_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and model
    backbone = config["model"]["backbone"]
    logger.info(f"Loading model and tokenizer: {backbone}")
    
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForMaskedLM.from_pretrained(backbone)
    
    logger.info(f"Model has {model.num_parameters():,} parameters")
    
    # Load corpus
    texts = load_telugu_corpus(
        dapt_cfg["data_path"],
        dapt_cfg["text_column"],
        include_style_data=dapt_cfg.get("include_style_data", True)
    )
    
    # Prepare MLM dataset
    dataset = prepare_mlm_dataset(
        texts,
        tokenizer,
        max_length=dapt_cfg["max_length"]
    )
    
    # Split train/val (95/5)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=dapt_cfg.get("mlm_probability", 0.15)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=dapt_cfg["epochs"],
        per_device_train_batch_size=dapt_cfg["batch_size"],
        per_device_eval_batch_size=dapt_cfg["batch_size"],
        gradient_accumulation_steps=dapt_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=dapt_cfg["learning_rate"],
        warmup_ratio=dapt_cfg.get("warmup_ratio", 0.1),
        weight_decay=dapt_cfg.get("weight_decay", 0.01),
        fp16=dapt_cfg.get("fp16", True),
        gradient_checkpointing=dapt_cfg.get("gradient_checkpointing", True),
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0  # Avoid multiprocessing issues
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train - resume from checkpoint if available
    logger.info("Starting DAPT training...")
    
    # Check for existing checkpoints
    checkpoints = list(output_dir.glob("checkpoint-*"))
    resume_from_checkpoint = None
    if checkpoints:
        # Sort by step number and get the latest
        checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
        resume_from_checkpoint = str(checkpoints[-1])
        logger.info(f"Resuming training from {resume_from_checkpoint}")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    logger.info(f"DAPT model saved to {final_dir}")
    logger.info("DAPT training complete!")
    
    return str(final_dir)


def main():
    # Load config
    config_path = "configs/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if not config.get("dapt", {}).get("enabled", False):
        logger.error("DAPT is not enabled in config.yaml")
        return
    
    # Run DAPT
    dapt_model_path = train_dapt(config)
    
    print("\n" + "="*80)
    print("DAPT COMPLETE!")
    print("="*80)
    print(f"Domain-adapted model saved to: {dapt_model_path}")
    print("\nNext steps:")
    print("1. Update config.yaml to use DAPT model:")
    print(f"   simple_classifier.backbone: \"{dapt_model_path}\"")
    print("2. Run style classification training:")
    print("   python scripts/train_simple_classifier.py --config configs/config.yaml")
    print("="*80)


if __name__ == "__main__":
    main()

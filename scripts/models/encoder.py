"""
Style Encoder Module.
Wrapper around transformer backbone for extracting style representations.

Supports:
- [CLS] token pooling
- Mean pooling
- Max pooling
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class StyleEncoder(nn.Module):
    """
    Encoder for extracting style-aware representations.
    
    Uses pretrained transformer (MuRIL, XLM-R, etc.) as backbone
    with configurable pooling strategy.
    """
    
    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        pooling: str = "cls",
        dropout: float = 0.1,
        hidden_size: Optional[int] = None,
        projection_size: Optional[int] = None,
        freeze_backbone: bool = False
    ):
        """
        Initialize encoder.
        
        Args:
            model_name: HuggingFace model name or path
            pooling: Pooling strategy ("cls", "mean", "max")
            dropout: Dropout rate
            hidden_size: Override hidden size (uses model config if None)
            projection_size: If set, add projection layer to this size
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.pooling = pooling
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from config
        self.hidden_size = hidden_size or self.config.hidden_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Optional projection layer
        self.projection = None
        self.output_size = self.hidden_size
        
        if projection_size is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, projection_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_size, projection_size)
            )
            self.output_size = projection_size
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Optional segment IDs (batch_size, seq_len)
            return_hidden_states: Whether to return all hidden states
            
        Returns:
            Dict with:
            - embeddings: Pooled representations (batch_size, output_size)
            - last_hidden_state: Full sequence output (batch_size, seq_len, hidden_size)
            - hidden_states: (optional) All layer outputs
        """
        # Prepare inputs
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": return_hidden_states
        }
        
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        
        # Forward through backbone
        outputs = self.backbone(**inputs)
        
        # Get last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Pool to get fixed-size representation
        embeddings = self._pool(last_hidden_state, attention_mask)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        # Apply projection if exists
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        result = {
            "embeddings": embeddings,
            "last_hidden_state": last_hidden_state
        }
        
        if return_hidden_states:
            result["hidden_states"] = outputs.hidden_states
        
        return result
    
    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequence representations.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            Pooled tensor (batch_size, hidden_size)
        """
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        
        elif self.pooling == "mean":
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling == "max":
            # Masked max pooling
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            return hidden_states.max(dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def get_embedding_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_size
    
    def save_pretrained(self, save_directory: str):
        """Save model and config."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save backbone
        self.backbone.save_pretrained(save_directory)
        
        # Save full model state
        torch.save({
            "projection": self.projection.state_dict() if self.projection else None,
            "pooling": self.pooling,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size
        }, os.path.join(save_directory, "encoder_config.pt"))
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load model from directory."""
        import os
        
        # Load config
        config_path = os.path.join(load_directory, "encoder_config.pt")
        if os.path.exists(config_path):
            saved_config = torch.load(config_path)
            kwargs.setdefault("pooling", saved_config.get("pooling", "cls"))
        
        # Create encoder
        encoder = cls(model_name=load_directory, **kwargs)
        
        # Load projection if exists
        if os.path.exists(config_path):
            saved_config = torch.load(config_path)
            if saved_config.get("projection") and encoder.projection:
                encoder.projection.load_state_dict(saved_config["projection"])
        
        return encoder


class StyleEncoderForClassification(nn.Module):
    """
    Style encoder with classification head.
    
    Used for standard classification training before cross-encoder.
    """
    
    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        num_labels: int = 9,
        pooling: str = "cls",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = StyleEncoder(
            model_name=model_name,
            pooling=pooling,
            dropout=dropout
        )
        
        self.classifier = nn.Linear(
            self.encoder.get_embedding_dim(),
            num_labels
        )
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.
        
        Returns:
            Dict with logits and optionally loss
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        embeddings = encoder_output["embeddings"]
        logits = self.classifier(embeddings)
        
        result = {
            "logits": logits,
            "embeddings": embeddings
        }
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)
        
        return result
    
    def get_encoder(self) -> StyleEncoder:
        """Get the underlying encoder."""
        return self.encoder

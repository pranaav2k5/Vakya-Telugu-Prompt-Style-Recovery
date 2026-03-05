"""
Contrastive Learning Model for Telugu Style Classification.

Implements Phase 2: Style-Aware Supervised Contrastive Learning.

Purpose:
- Learn representation space where similar styles cluster together
- Stylistically similar texts are close, dissimilar are far
- Does NOT force mutual exclusivity between compatible styles
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import StyleEncoder
from .losses import WeightedSupConLoss, SupConLoss


class ContrastiveModel(nn.Module):
    """
    Contrastive learning model for style representation.
    
    Uses StyleEncoder backbone + projection head for contrastive learning.
    The projection head maps to a normalized embedding space.
    """
    
    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        pooling: str = "cls",
        dropout: float = 0.1,
        projection_dim: int = 256,
        temperature: float = 0.1,
        use_weighted_loss: bool = True
    ):
        """
        Initialize contrastive model.
        
        Args:
            model_name: Pretrained model name/path
            pooling: Pooling strategy
            dropout: Dropout rate
            projection_dim: Dimension of projection space
            temperature: Contrastive loss temperature
            use_weighted_loss: Use style-weighted SupCon loss
        """
        super().__init__()
        
        self.encoder = StyleEncoder(
            model_name=model_name,
            pooling=pooling,
            dropout=dropout
        )
        
        hidden_dim = self.encoder.get_embedding_dim()
        
        # Projection head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.use_weighted_loss = use_weighted_loss
        
        # Loss functions
        if use_weighted_loss:
            self.criterion = WeightedSupConLoss(temperature=temperature)
        else:
            self.criterion = SupConLoss(temperature=temperature)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        similarity_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Optional segment IDs
            labels: Ground truth labels for loss computation
            similarity_weights: Style similarity weights for weighted loss
            
        Returns:
            Dict with:
            - embeddings: Raw encoder embeddings
            - projections: Projected embeddings (for contrastive loss)
            - loss: (optional) Contrastive loss
        """
        # Get encoder embeddings
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        embeddings = encoder_output["embeddings"]
        
        # Project to contrastive space
        projections = self.projection_head(embeddings)
        
        # L2 normalize projections
        projections = F.normalize(projections, p=2, dim=1)
        
        result = {
            "embeddings": embeddings,
            "projections": projections
        }
        
        # Compute loss if labels provided
        if labels is not None:
            if self.use_weighted_loss and similarity_weights is not None:
                loss = self.criterion(
                    features=projections,
                    labels=labels,
                    similarity_weights=similarity_weights
                )
            else:
                loss = self.criterion(
                    features=projections,
                    labels=labels
                )
            result["loss"] = loss
        
        return result
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Get normalized embeddings for inference.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Optional segment IDs
            normalize: L2 normalize embeddings
            
        Returns:
            Embeddings tensor
        """
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        
        embeddings = output["embeddings"]
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_encoder(self) -> StyleEncoder:
        """Return the encoder for transfer to cross-encoder."""
        return self.encoder
    
    def save_pretrained(self, save_directory: str):
        """Save model."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save encoder
        self.encoder.save_pretrained(save_directory)
        
        # Save projection head
        torch.save({
            "projection_head": self.projection_head.state_dict(),
            "projection_dim": self.projection_dim,
            "temperature": self.temperature,
            "use_weighted_loss": self.use_weighted_loss
        }, os.path.join(save_directory, "contrastive_config.pt"))
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load model."""
        import os
        
        config_path = os.path.join(load_directory, "contrastive_config.pt")
        if os.path.exists(config_path):
            saved_config = torch.load(config_path)
            kwargs.setdefault("projection_dim", saved_config.get("projection_dim", 256))
            kwargs.setdefault("temperature", saved_config.get("temperature", 0.1))
            kwargs.setdefault("use_weighted_loss", saved_config.get("use_weighted_loss", True))
        
        model = cls(model_name=load_directory, **kwargs)
        
        if os.path.exists(config_path):
            saved_config = torch.load(config_path)
            model.projection_head.load_state_dict(saved_config["projection_head"])
        
        return model


class ContrastiveModelWithClassifier(nn.Module):
    """
    Contrastive model with optional classification head.
    
    Useful for joint contrastive + classification training.
    """
    
    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        num_labels: int = 9,
        pooling: str = "cls",
        dropout: float = 0.1,
        projection_dim: int = 256,
        temperature: float = 0.1,
        contrastive_weight: float = 0.5,
        use_weighted_loss: bool = True
    ):
        super().__init__()
        
        self.contrastive_model = ContrastiveModel(
            model_name=model_name,
            pooling=pooling,
            dropout=dropout,
            projection_dim=projection_dim,
            temperature=temperature,
            use_weighted_loss=use_weighted_loss
        )
        
        hidden_dim = self.contrastive_model.encoder.get_embedding_dim()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )
        
        self.num_labels = num_labels
        self.contrastive_weight = contrastive_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        similarity_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward with joint loss."""
        # Get contrastive outputs
        contrastive_output = self.contrastive_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            similarity_weights=similarity_weights
        )
        
        embeddings = contrastive_output["embeddings"]
        
        # Classification
        logits = self.classifier(embeddings)
        
        result = {
            "embeddings": embeddings,
            "projections": contrastive_output["projections"],
            "logits": logits
        }
        
        if labels is not None:
            contrastive_loss = contrastive_output.get("loss", torch.tensor(0.0))
            classification_loss = self.ce_loss(logits, labels)
            
            total_loss = (
                self.contrastive_weight * contrastive_loss +
                (1 - self.contrastive_weight) * classification_loss
            )
            
            result["loss"] = total_loss
            result["contrastive_loss"] = contrastive_loss
            result["classification_loss"] = classification_loss
        
        return result
    
    def get_encoder(self) -> StyleEncoder:
        """Return encoder for transfer."""
        return self.contrastive_model.get_encoder()
    
    def save_pretrained(self, save_directory: str):
        """Save model components."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save contrastive model
        self.contrastive_model.save_pretrained(save_directory)
        
        # Save classifier state
        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save({
            "classifier": self.classifier.state_dict(),
            "num_labels": self.num_labels,
            "contrastive_weight": self.contrastive_weight
        }, classifier_path)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load model."""
        import os
        
        # Load contrastive config
        config_path = os.path.join(load_directory, "contrastive_config.pt")
        if os.path.exists(config_path):
            saved_config = torch.load(config_path)
            kwargs.setdefault("projection_dim", saved_config.get("projection_dim", 256))
            kwargs.setdefault("temperature", saved_config.get("temperature", 0.1))
        
        # Load classifier config
        classifier_path = os.path.join(load_directory, "classifier.pt")
        if os.path.exists(classifier_path):
            classifier_config = torch.load(classifier_path)
            kwargs.setdefault("num_labels", classifier_config.get("num_labels", 9))
            kwargs.setdefault("contrastive_weight", classifier_config.get("contrastive_weight", 0.5))
        
        # Create model
        model = cls(model_name=load_directory, **kwargs)
        
        # Load classifier weights
        if os.path.exists(classifier_path):
            classifier_config = torch.load(classifier_path)
            model.classifier.load_state_dict(classifier_config["classifier"])
        
        return model

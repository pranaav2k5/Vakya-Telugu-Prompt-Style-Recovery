"""
Loss Functions for Telugu Style Classification.

Implements:
1. WeightedSupConLoss: Weighted supervised contrastive loss (Phase 2)
2. OverlapAwareCrossEntropyLoss: Soft cross-entropy with style overlap (Phase 3)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSupConLoss(nn.Module):
    """
    Weighted Supervised Contrastive Loss.
    
    From Pipeline Phase 2.2:
    - Strong positives: Same style label (weight = 1.0)
    - Weak positives: Stylistically close labels (weight = 0.3-0.5)
    - Negatives: Distant styles (weight = 0.0, contribute to denominator)
    
    Loss:
        L_i = -Σ_p w_ip * log(exp(sim(z_i, z_p)/τ) / Σ_a exp(sim(z_i, z_a)/τ))
    
    Benefits:
    - Models multi-style overlap naturally
    - Prevents over-separation of valid styles
    - Produces smooth style geometry
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        base_temperature: float = 0.1,
        contrast_mode: str = "all"
    ):
        """
        Initialize loss.
        
        Args:
            temperature: Scaling temperature τ
            base_temperature: Base temperature for scaling
            contrast_mode: "one" (use one view) or "all" (use all views)
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        similarity_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted supervised contrastive loss.
        
        Args:
            features: Hidden vectors (batch_size, hidden_dim), L2 normalized
            labels: Ground truth labels (batch_size,)
            similarity_weights: Pairwise weights matrix (batch_size, num_classes)
                               If None, uses hard positive/negative
            mask: Optional contrastive mask (batch_size, batch_size)
        
        Returns:
            Scalar loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # L2 normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive mask from labels
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(device)
        
        # If similarity weights provided, use them for weighted positives
        if similarity_weights is not None:
            # similarity_weights: (batch_size, num_classes)
            # We need to convert to pairwise weights
            weight_mask = self._compute_pairwise_weights(
                labels.squeeze(), similarity_weights
            )
        else:
            # Use binary mask (1 for same label, 0 otherwise)
            weight_mask = label_mask
        
        # Mask out self-contrast
        logits_mask = torch.ones_like(similarity_matrix) - torch.eye(batch_size).to(device)
        
        # Apply mask
        mask_positives = weight_mask * logits_mask
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix * logits_mask, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log-softmax over all negatives + positives
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        # Compute weighted mean of log-likelihood over positive pairs
        # Weight by similarity_weights
        mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1) / (
            mask_positives.sum(dim=1) + 1e-12
        )
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss
    
    def _compute_pairwise_weights(
        self,
        labels: torch.Tensor,
        similarity_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert per-sample similarity weights to pairwise weight matrix.
        
        Args:
            labels: (batch_size,) label indices
            similarity_weights: (batch_size, num_classes) similarity to each class
            
        Returns:
            (batch_size, batch_size) pairwise weight matrix
        """
        batch_size = labels.shape[0]
        device = labels.device
        
        # Get weight for each pair based on target sample's label
        # weight[i,j] = similarity_weights[i, labels[j]]
        pairwise_weights = torch.zeros(batch_size, batch_size, device=device)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    # Weight from i's perspective to j's label
                    pairwise_weights[i, j] = similarity_weights[i, labels[j]]
        
        return pairwise_weights


class SupConLoss(nn.Module):
    """
    Standard Supervised Contrastive Loss (without weighting).
    
    Fallback for when similarity weights are not available.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: (batch_size, hidden_dim) L2-normalized embeddings
            labels: (batch_size,) ground truth labels
            
        Returns:
            Scalar loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # L2 normalize
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarities
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Mask for same-label pairs (positives)
        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-comparisons
        identity = torch.eye(batch_size, device=device)
        pos_mask = pos_mask - identity
        
        # LogSumExp for numerical stability
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        
        # Compute log-softmax
        exp_sim = torch.exp(sim_matrix) * (1 - identity)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        
        # Mean over positives
        pos_count = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-12)
        
        # Only compute loss for samples with at least one positive
        valid_mask = pos_count > 0
        loss = -mean_log_prob[valid_mask].mean()
        
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=device)


class OverlapAwareCrossEntropyLoss(nn.Module):
    """
    Overlap-Aware Cross-Entropy Loss for Cross-Encoder.
    
    From Pipeline Phase 3.3:
    - Gold style should have highest probability
    - Confusable styles allowed non-zero mass
    - Uses soft cross-entropy / KL-divergence
    
    Training targets are soft distributions based on style similarity.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        use_kl: bool = True
    ):
        """
        Initialize loss.
        
        Args:
            temperature: Temperature for softmax
            label_smoothing: Optional label smoothing (0.0 = disabled)
            use_kl: If True, use KL divergence. If False, use soft CE.
        """
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.use_kl = use_kl
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute overlap-aware loss.
        
        Args:
            logits: Raw model outputs (batch_size, num_classes)
            labels: Hard labels (batch_size,) - used if soft_labels is None
            soft_labels: Soft target distribution (batch_size, num_classes)
            
        Returns:
            Scalar loss
        """
        num_classes = logits.shape[-1]
        
        # Apply temperature to logits
        scaled_logits = logits / self.temperature
        
        # Get log probabilities
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        
        if soft_labels is not None:
            # Use soft labels
            targets = soft_labels
            
            # Optional label smoothing on top of soft labels
            if self.label_smoothing > 0:
                smooth = self.label_smoothing / num_classes
                targets = (1 - self.label_smoothing) * targets + smooth
        else:
            # Convert hard labels to soft with optional smoothing
            if self.label_smoothing > 0:
                targets = torch.zeros_like(logits)
                targets.fill_(self.label_smoothing / num_classes)
                targets.scatter_(1, labels.unsqueeze(1), 1 - self.label_smoothing)
            else:
                targets = F.one_hot(labels, num_classes).float()
        
        if self.use_kl:
            # KL divergence: KL(targets || predictions)
            loss = F.kl_div(log_probs, targets, reduction='batchmean')
        else:
            # Soft cross-entropy: -sum(target * log(pred))
            loss = -(targets * log_probs).sum(dim=-1).mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance (optional).
    
    Useful if the dataset becomes imbalanced after preprocessing.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights (num_classes,)
            gamma: Focusing parameter
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[labels]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for joint contrastive + classification training.
    """
    
    def __init__(
        self,
        contrastive_weight: float = 0.5,
        classification_weight: float = 0.5,
        temperature: float = 0.1
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        embeddings: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        Compute combined loss.
        
        Returns:
            Dict with total_loss, contrastive_loss, classification_loss
        """
        contrastive = self.supcon_loss(embeddings, labels)
        classification = self.ce_loss(logits, labels)
        
        total = (
            self.contrastive_weight * contrastive +
            self.classification_weight * classification
        )
        
        return {
            "loss": total,
            "contrastive_loss": contrastive,
            "classification_loss": classification
        }
class RefinementLoss(nn.Module):
    """
    Hard Confusion Refinement Loss (Phase 4).

    From Paper Equation (7):
    L_ref = L_ce + beta * Sum (p_j + p_k - 1)^2
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        beta: float = 0.1,
        confusion_pairs: Optional[list] = None,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.beta = beta
        self.confusion_pairs = confusion_pairs or []

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if soft_labels is not None and isinstance(
            self.base_criterion, OverlapAwareCrossEntropyLoss
        ):
            ce_loss = self.base_criterion(logits, labels, soft_labels)
        else:
            ce_loss = self.base_criterion(logits, labels)

        penalty = torch.tensor(0.0, device=logits.device)
        if self.confusion_pairs:
            probs = F.softmax(logits, dim=-1)
            for j, k in self.confusion_pairs:
                p_j = probs[:, j]
                p_k = probs[:, k]
                term = (p_j + p_k - 1) ** 2
                penalty += term.mean()

        return ce_loss + self.beta * penalty

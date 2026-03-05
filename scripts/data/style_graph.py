"""
Style Similarity Graph Module.
Defines stylistic proximity relationships for weighted contrastive learning.

From Pipeline (Phase 2.1):
    Inspiring     ↔ Optimistic, Persuasive
    Optimistic    ↔ Inspiring
    Serious       ↔ Authoritative
    Authoritative ↔ Serious, Persuasive
    Informal      ↔ Humorous
    Humorous      ↔ Informal
    Pessimistic   ↔ Serious
"""

from typing import Dict, List, Optional
import numpy as np


class StyleGraph:
    """
    Style similarity graph for Telugu prompt styles.
    
    Provides:
    - Weighted similarity matrix for contrastive learning
    - Style hypothesis definitions for cross-encoder
    - Label encoding/decoding utilities
    
    The graph captures natural overlaps between styles:
    - Formal/Authoritative share professional tone
    - Informal/Humorous share casual register  
    - Inspiring/Optimistic share positive sentiment
    - Serious/Pessimistic share grave undertones
    """
    
    # Canonical 9 style labels (must match dataset)
    STYLES = [
        "Formal",
        "Informal",
        "Optimistic",
        "Pessimistic",
        "Humorous",
        "Serious",
        "Inspiring",
        "Authoritative",
        "Persuasive"
    ]
    
    # English style hypothesis definitions for cross-encoder
    STYLE_HYPOTHESES_EN = {
        "Formal": "Polite, structured Telugu with professional register, respectful tone, full sentences, and minimal slang.",
        "Informal": "Casual, conversational Telugu with friendly tone, slang, or contractions.",
        "Optimistic": "Positive outlook, hopeful expressions, encouragement, future success focus.",
        "Pessimistic": "Negative, doubtful, discouraging, or cautionary language.",
        "Humorous": "Playful Telugu using jokes, exaggeration, irony, or light metaphors.",
        "Serious": "Sober, factual, grave tone focused on important or sensitive matters.",
        "Inspiring": "Motivational and uplifting language with calls to action.",
        "Authoritative": "Commanding, directive tone with expert certainty and imperatives.",
        "Persuasive": "Language intended to convince or influence through appeal or urgency."
    }
    
    # Telugu style hypothesis definitions for cross-encoder
    STYLE_HYPOTHESES_TE = {
        "Formal": "మర్యాదపూర్వకమైన, వ్యవస్థీకృత తెలుగు భాష, వృత్తిపరమైన శైలి, గౌరవప్రదమైన స్వరం, పూర్తి వాక్యాలు, కనీస మాండలికం.",
        "Informal": "సాధారణ, సంభాషణాత్మక తెలుగు, స్నేహపూర్వక స్వరం, మాండలికం లేదా సంక్షిప్తాలు.",
        "Optimistic": "సానుకూల దృక్పథం, ఆశాజనక వ్యక్తీకరణలు, ప్రోత్సాహం, భవిష్యత్తు విజయంపై దృష్టి.",
        "Pessimistic": "ప్రతికూల, సందేహాస్పద, నిరుత్సాహపరిచే లేదా హెచ్చరిక భాష.",
        "Humorous": "చమత్కారమైన తెలుగు, జోకులు, అతిశయోక్తి, వ్యంగ్యం లేదా తేలికపాటి ఉపమానాలు.",
        "Serious": "నిదానమైన, వాస్తవిక, గంభీరమైన స్వరం, ముఖ్యమైన లేదా సున్నితమైన విషయాలపై దృష్టి.",
        "Inspiring": "ప్రేరణాత్మక మరియు ఉత్తేజపరిచే భాష, చర్యకు పిలుపులు.",
        "Authoritative": "ఆజ్ఞాపూర్వక, నిర్దేశక స్వరం, నిపుణుల నిశ్చయత మరియు ఆదేశాలు.",
        "Persuasive": "ఆకర్షణ లేదా ఆవశ్యకత ద్వారా ఒప్పించడానికి లేదా ప్రభావితం చేయడానికి ఉద్దేశించిన భాష."
    }
    
    # Combined bilingual hypotheses (computed as property to avoid class-level reference issues)
    STYLE_HYPOTHESES_BOTH = {
        "Formal": "Polite, structured Telugu with professional register, respectful tone, full sentences, and minimal slang. / మర్యాదపూర్వకమైన, వ్యవస్థీకృత తెలుగు భాష, వృత్తిపరమైన శైలి, గౌరవప్రదమైన స్వరం, పూర్తి వాక్యాలు, కనీస మాండలికం.",
        "Informal": "Casual, conversational Telugu with friendly tone, slang, or contractions. / సాధారణ, సంభాషణాత్మక తెలుగు, స్నేహపూర్వక స్వరం, మాండలికం లేదా సంక్షిప్తాలు.",
        "Optimistic": "Positive outlook, hopeful expressions, encouragement, future success focus. / సానుకూల దృక్పథం, ఆశాజనక వ్యక్తీకరణలు, ప్రోత్సాహం, భవిష్యత్తు విజయంపై దృష్టి.",
        "Pessimistic": "Negative, doubtful, discouraging, or cautionary language. / ప్రతికూల, సందేహాస్పద, నిరుత్సాహపరిచే లేదా హెచ్చరిక భాష.",
        "Humorous": "Playful Telugu using jokes, exaggeration, irony, or light metaphors. / చమత్కారమైన తెలుగు, జోకులు, అతిశయోక్తి, వ్యంగ్యం లేదా తేలికపాటి ఉపమానాలు.",
        "Serious": "Sober, factual, grave tone focused on important or sensitive matters. / నిదానమైన, వాస్తవిక, గంభీరమైన స్వరం, ముఖ్యమైన లేదా సున్నితమైన విషయాలపై దృష్టి.",
        "Inspiring": "Motivational and uplifting language with calls to action. / ప్రేరణాత్మక మరియు ఉత్తేజపరిచే భాష, చర్యకు పిలుపులు.",
        "Authoritative": "Commanding, directive tone with expert certainty and imperatives. / ఆజ్ఞాపూర్వక, నిర్దేశక స్వరం, నిపుణుల నిశ్చయత మరియు ఆదేశాలు.",
        "Persuasive": "Language intended to convince or influence through appeal or urgency. / ఆకర్షణ లేదా ఆవశ్యకత ద్వారా ఒప్పించడానికి లేదా ప్రభావితం చేయడానికి ఉద్దేశించిన భాష."
    }
    
    # Default adjacency relationships
    DEFAULT_ADJACENCY = {
        "Formal": ["Authoritative", "Serious"],
        "Informal": ["Humorous"],
        "Optimistic": ["Inspiring"],
        "Pessimistic": ["Serious"],
        "Humorous": ["Informal"],
        "Serious": ["Authoritative", "Pessimistic", "Formal"],
        "Inspiring": ["Optimistic", "Persuasive"],
        "Authoritative": ["Serious", "Persuasive", "Formal"],
        "Persuasive": ["Inspiring", "Authoritative"]
    }
    
    def __init__(
        self,
        weak_positive_weight: float = 0.4,
        adjacency: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize style graph.
        
        Args:
            weak_positive_weight: Weight for stylistically similar pairs (0.3-0.5)
            adjacency: Custom adjacency dict, or use default
        """
        self.weak_positive_weight = weak_positive_weight
        self.adjacency = adjacency or self.DEFAULT_ADJACENCY
        
        # Label encodings
        self.style_to_idx = {s: i for i, s in enumerate(self.STYLES)}
        self.idx_to_style = {i: s for i, s in enumerate(self.STYLES)}
        self.num_styles = len(self.STYLES)
        
        # Build similarity matrix
        self._similarity_matrix = self._build_similarity_matrix()
    
    def _build_similarity_matrix(self) -> np.ndarray:
        """
        Build NxN style similarity matrix.
        
        Returns:
            Matrix where:
            - diagonal = 1.0 (same style)
            - adjacent styles = weak_positive_weight
            - distant styles = 0.0
        """
        matrix = np.zeros((self.num_styles, self.num_styles), dtype=np.float32)
        
        # Diagonal: same style = 1.0
        np.fill_diagonal(matrix, 1.0)
        
        # Adjacent styles: weak positive weight (symmetric)
        for style, neighbors in self.adjacency.items():
            i = self.style_to_idx.get(style)
            if i is None:
                continue
            for neighbor in neighbors:
                j = self.style_to_idx.get(neighbor)
                if j is None:
                    continue
                matrix[i, j] = self.weak_positive_weight
                matrix[j, i] = self.weak_positive_weight
        
        return matrix
    
    def get_similarity(self, style1: str, style2: str) -> float:
        """Get similarity weight between two styles."""
        i = self.style_to_idx.get(style1)
        j = self.style_to_idx.get(style2)
        if i is None or j is None:
            return 0.0
        return float(self._similarity_matrix[i, j])
    
    def get_similarity_vector(self, style: str) -> np.ndarray:
        """
        Get similarity weights from one style to all others.
        
        Args:
            style: Style label
            
        Returns:
            Array of shape (num_styles,) with similarity weights
        """
        idx = self.style_to_idx.get(style)
        if idx is None:
            return np.zeros(self.num_styles, dtype=np.float32)
        return self._similarity_matrix[idx].copy()
    
    def get_neighbors(self, style: str) -> List[str]:
        """Get adjacent (similar) styles."""
        return self.adjacency.get(style, [])
    
    def is_adjacent(self, style1: str, style2: str) -> bool:
        """Check if two styles are adjacent (similar)."""
        return style2 in self.adjacency.get(style1, [])
    
    def get_hypothesis(
        self,
        style: str,
        language: str = "english"
    ) -> str:
        """
        Get style hypothesis definition.
        
        Args:
            style: Style label
            language: "english", "telugu", or "both"
            
        Returns:
            Hypothesis text
        """
        if language == "telugu":
            return self.STYLE_HYPOTHESES_TE.get(style, "")
        elif language == "both":
            return self.STYLE_HYPOTHESES_BOTH.get(style, "")
        else:
            return self.STYLE_HYPOTHESES_EN.get(style, "")
    
    def get_all_hypotheses(
        self,
        language: str = "english"
    ) -> Dict[str, str]:
        """Get all style hypotheses."""
        if language == "telugu":
            return self.STYLE_HYPOTHESES_TE.copy()
        elif language == "both":
            return self.STYLE_HYPOTHESES_BOTH.copy()
        else:
            return self.STYLE_HYPOTHESES_EN.copy()
    
    def get_hypotheses_list(
        self,
        language: str = "english"
    ) -> List[str]:
        """Get hypotheses as ordered list (matching STYLES order)."""
        hypotheses = self.get_all_hypotheses(language)
        return [hypotheses[s] for s in self.STYLES]
    
    def label_to_idx(self, label: str) -> int:
        """Convert style label to index."""
        return self.style_to_idx.get(label, -1)
    
    def idx_to_label(self, idx: int) -> str:
        """Convert index to style label."""
        return self.idx_to_style.get(idx, "Unknown")
    
    def batch_labels_to_idx(self, labels: List[str]) -> List[int]:
        """Convert batch of labels to indices."""
        return [self.label_to_idx(l) for l in labels]
    
    def batch_idx_to_labels(self, indices: List[int]) -> List[str]:
        """Convert batch of indices to labels."""
        return [self.idx_to_label(i) for i in indices]
    
    def get_soft_target(
        self,
        gold_label: str,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Get soft target distribution for overlap-aware training.
        
        Creates a soft probability distribution where:
        - Gold label gets highest mass
        - Similar styles get some mass (based on adjacency)
        - Distant styles get minimal mass
        
        Args:
            gold_label: Ground truth style
            temperature: Softmax temperature (lower = sharper)
            
        Returns:
            Soft probability distribution over styles
        """
        similarities = self.get_similarity_vector(gold_label)
        
        # Apply temperature-scaled softmax
        # Add small epsilon to prevent division by zero
        exp_sim = np.exp(similarities / (temperature + 1e-8))
        soft_target = exp_sim / (exp_sim.sum() + 1e-8)
        
        return soft_target.astype(np.float32)
    
    def get_contrastive_weights(
        self,
        anchor_label: str,
        batch_labels: List[str]
    ) -> np.ndarray:
        """
        Get contrastive learning weights for a batch.
        
        Args:
            anchor_label: Label of anchor sample
            batch_labels: Labels of all samples in batch
            
        Returns:
            Weight array where:
            - 1.0 for same label (strong positive)
            - weak_positive_weight for adjacent labels
            - 0.0 for distant labels (negative)
        """
        weights = np.zeros(len(batch_labels), dtype=np.float32)
        anchor_idx = self.style_to_idx.get(anchor_label)
        
        if anchor_idx is None:
            return weights
        
        for i, label in enumerate(batch_labels):
            label_idx = self.style_to_idx.get(label)
            if label_idx is not None:
                weights[i] = self._similarity_matrix[anchor_idx, label_idx]
        
        return weights
    
    @property
    def similarity_matrix(self) -> np.ndarray:
        """Return copy of the full similarity matrix."""
        return self._similarity_matrix.copy()
    
    @property
    def labels(self) -> List[str]:
        """Return list of style labels."""
        return self.STYLES.copy()
    
    def __repr__(self) -> str:
        return (
            f"StyleGraph(num_styles={self.num_styles}, "
            f"weak_weight={self.weak_positive_weight})"
        )
    
    def __len__(self) -> int:
        return self.num_styles


def create_style_graph(config: Optional[dict] = None) -> StyleGraph:
    """
    Factory function to create StyleGraph from config.
    
    Args:
        config: Config dict with 'weak_positive_weight' and 'adjacency'
        
    Returns:
        Configured StyleGraph instance
    """
    if config is None:
        config = {}
    
    return StyleGraph(
        weak_positive_weight=config.get('weak_positive_weight', 0.4),
        adjacency=config.get('adjacency', None)
    )

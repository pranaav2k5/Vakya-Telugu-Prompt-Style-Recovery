"""
Telugu Text Preprocessing Module.
Handles Unicode normalization while preserving stylistic markers.

Key Principles (from Pipeline):
- Unicode normalization (Telugu UTF-8)
- Preserve punctuation, emojis, discourse markers
- Do NOT remove slang or stylistic noise
- Minimal token normalization only
"""

import re
import unicodedata
from typing import Optional, List


class TeluguPreprocessor:
    """
    Preprocessor for Telugu text that preserves stylistic features.
    
    Designed for style classification where stylistic markers
    (punctuation, emojis, informal language) are important signals.
    """
    
    # Telugu Unicode range: U+0C00 to U+0C7F
    TELUGU_RANGE = (0x0C00, 0x0C7F)
    
    # Extended Telugu range including digits
    TELUGU_EXTENDED_RANGE = (0x0C00, 0x0C7F)
    
    # Common emoji pattern (preserve these for style detection)
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols Extended-A
        "\U00002600-\U000026FF"  # Misc symbols
        "\U00002700-\U000027BF"  # Dingbats
        "]+",
        flags=re.UNICODE
    )
    
    # Discourse markers and stylistic punctuation to preserve
    PRESERVE_PATTERNS = [
        r'\.{2,}',      # Ellipsis
        r'\!+',          # Multiple exclamation marks
        r'\?+',          # Multiple question marks
        r'\~+',          # Tilde (informal)
        r'\*+',          # Asterisks (emphasis)
    ]
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
        preserve_emojis: bool = True,
        preserve_punctuation: bool = True,
        preserve_case: bool = True,
        min_length: int = 1
    ):
        """
        Initialize preprocessor with configuration.
        
        Args:
            normalize_unicode: Apply NFC normalization
            normalize_whitespace: Collapse multiple spaces
            preserve_emojis: Keep emoji characters
            preserve_punctuation: Keep stylistic punctuation
            preserve_case: Keep original case
            min_length: Minimum text length after processing
        """
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.preserve_emojis = preserve_emojis
        self.preserve_punctuation = preserve_punctuation
        self.preserve_case = preserve_case
        self.min_length = min_length
    
    def __call__(self, text: str) -> str:
        """Process a single text string."""
        return self.preprocess(text)
    
    def preprocess(self, text: Optional[str]) -> str:
        """
        Main preprocessing pipeline.
        
        Args:
            text: Raw Telugu text
            
        Returns:
            Cleaned text preserving stylistic features
        """
        # Handle None or non-string input
        if text is None or not isinstance(text, str):
            return ""
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        if not text:
            return ""
        
        # Unicode normalization (NFC is standard for Telugu)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)
            
        # Remove common footer artifacts
        text = self.remove_footer(text)
        
        # Normalize whitespace (collapse multiple spaces)
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        # Case normalization (optional, usually preserve for Telugu)
        if not self.preserve_case:
            text = text.lower()
        
        # Remove very short texts
        if len(text.strip()) < self.min_length:
            return ""
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters into single space."""
        # Replace multiple spaces/tabs with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize multiple newlines to single newline
        text = re.sub(r'\n+', '\n', text)
        # Strip each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()
    
    def is_telugu_char(self, char: str) -> bool:
        """Check if a character is in Telugu Unicode range."""
        if len(char) != 1:
            return False
        code_point = ord(char)
        return self.TELUGU_RANGE[0] <= code_point <= self.TELUGU_RANGE[1]
    
    def get_telugu_ratio(self, text: str) -> float:
        """
        Calculate the ratio of Telugu characters in text.
        
        Useful for filtering non-Telugu or mixed-language texts.
        """
        if not text:
            return 0.0
        
        telugu_chars = sum(1 for c in text if self.is_telugu_char(c))
        alpha_chars = sum(1 for c in text if c.isalpha())
        
        if alpha_chars == 0:
            return 0.0
        
        return telugu_chars / alpha_chars
    
    def has_emojis(self, text: str) -> bool:
        """Check if text contains emojis."""
        return bool(self.EMOJI_PATTERN.search(text))
    
    def extract_emojis(self, text: str) -> List[str]:
        """Extract all emojis from text."""
        return self.EMOJI_PATTERN.findall(text)
    
    def count_stylistic_markers(self, text: str) -> dict:
        """
        Count stylistic markers in text.
        
        Returns:
            Dict with counts of different marker types
        """
        markers = {
            'exclamation': len(re.findall(r'\!', text)),
            'question': len(re.findall(r'\?', text)),
            'ellipsis': len(re.findall(r'\.{2,}', text)),
            'emojis': len(self.extract_emojis(text)),
            'capitals': sum(1 for c in text if c.isupper()),
            'quotes': len(re.findall(r'["\'\"\"]', text)),
        }
        return markers
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """Process a batch of texts."""
        return [self.preprocess(t) for t in texts]
    
    def remove_footer(self, text: str) -> str:
        """Remove common footer artifacts identified in dataset."""
        # Footer text parts (sometimes appearing together, sometimes alone)
        footers = [
            "మీకు ఈ ట్రాన్స్క్రిప్ట్ పూర్తిగా, అవసరమైన విధంగా, సహజంగా మరియు నిఖార్సైన తెలుగులో అందించబడింది. మరింత వివరాలు కావాలంటే తెలియజేయండి.",
            "మీకు ఈ ట్రాన్స్క్రిప్ట్ పూర్తిగా, అవసరమైన విధంగా, సహజంగా మరియు నిఖార్సైన తెలుగులో అందించబడింది.",
        ]
        
        for f in footers:
            text = text.replace(f, "")
            
        return text.strip()

    def get_stats(self, texts: List[str]) -> dict:
        """
        Get preprocessing statistics for a batch.
        
        Args:
            texts: List of texts
            
        Returns:
            Statistics dict
        """
        processed = self.batch_preprocess(texts)
        
        lengths = [len(t) for t in processed if t]
        telugu_ratios = [self.get_telugu_ratio(t) for t in processed if t]
        
        return {
            'total': len(texts),
            'non_empty': len([t for t in processed if t]),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'avg_telugu_ratio': sum(telugu_ratios) / len(telugu_ratios) if telugu_ratios else 0,
            'with_emojis': sum(1 for t in processed if self.has_emojis(t)),
        }


def create_preprocessor(config: Optional[dict] = None) -> TeluguPreprocessor:
    """
    Factory function to create preprocessor.
    
    Args:
        config: Optional configuration dict
        
    Returns:
        Configured TeluguPreprocessor instance
    """
    if config is None:
        config = {}
    
    return TeluguPreprocessor(
        normalize_unicode=config.get('normalize_unicode', True),
        normalize_whitespace=config.get('normalize_whitespace', True),
        preserve_emojis=config.get('preserve_emojis', True),
        preserve_punctuation=config.get('preserve_punctuation', True),
        preserve_case=config.get('preserve_case', True),
        min_length=config.get('min_length', 1)
    )

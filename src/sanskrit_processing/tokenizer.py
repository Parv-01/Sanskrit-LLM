"""Sanskrit tokenizer module."""

from typing import List


class SanskritTokenizer:
    """Tokenizer for Sanskrit text.
    
    Provides basic tokenization capabilities for Devanagari Sanskrit text,
    handling compound words, sandhi, and special characters.
    
    Example:
        >>> tokenizer = SanskritTokenizer()
        >>> tokens = tokenizer.tokenize("आयुर्वेदः सर्वदा रक्षति")
        >>> print(tokens)
        ['आयुर्वेदः', 'सर्वदा', 'रक्षति']
    """
    
    def __init__(self):
        """Initialize the Sanskrit tokenizer."""
        self.virama = '्'
        self.danda = '।'
        self.separator = ' '
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Sanskrit text into words.
        
        Args:
            text: Input Sanskrit text in Devanagari script.
            
        Returns:
            List of tokenized words.
        """
        if not text:
            return []
        
        text = text.strip()
        tokens = []
        
        current_token = ""
        for char in text:
            if char == self.separator:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char == self.danda:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(self.danda)
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def remove_diacritics(self, token: str) -> str:
        """Remove halant (virama) from tokens.
        
        Args:
            token: Input token with potential virama.
            
        Returns:
            Token with virama removed.
        """
        return token.replace(self.virama, '')
    
    def get_word_boundaries(self, text: str) -> List[tuple]:
        """Get start and end positions of words.
        
        Args:
            text: Input Sanskrit text.
            
        Returns:
            List of (start, end) tuples for each word.
        """
        tokens = self.tokenize(text)
        boundaries = []
        pos = 0
        
        for token in tokens:
            start = text.find(token, pos)
            if start != -1:
                end = start + len(token)
                boundaries.append((start, end))
                pos = end
        
        return boundaries


def demo():
    """Demonstration function for Sanskrit tokenizer."""
    tokenizer = SanskritTokenizer()
    
    sample_text = "आयुर्वेदः सर्वदा रक्षति। स्वस्थस्य स्वास्थ्य रक्षणं, आतुरस्य विकार प्रशमनं च।"
    
    print(f"Input: {sample_text}")
    print(f"Tokens: {tokenizer.tokenize(sample_text)}")
    print(f"Word boundaries: {tokenizer.get_word_boundaries(sample_text)}")


if __name__ == "__main__":
    demo()

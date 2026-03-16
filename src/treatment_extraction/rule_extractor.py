"""Rule-based treatment extraction from Sanskrit texts."""

from typing import List, Dict, Optional
import re


class RuleBasedExtractor:
    """Rule-based extractor for Ayurvedic treatments.
    
    Uses pattern matching and keyword rules to extract treatment
    information from Sanskrit Ayurvedic texts.
    
    Example:
        >>> extractor = RuleBasedExtractor()
        >>> treatments = extractor.extract("ज्वरस्य निदानं तिक्ताम्ललवणं परिहरेत्")
        >>> print(treatments)
    """
    
    TREATMENT_PATTERNS: List[re.Pattern] = [
        re.compile(r'(\w+)\s+निदानं', re.UNICODE),
        re.compile(r'(\w+)\s+चिकित्सा', re.UNICODE),
        re.compile(r'(\w+)\s+उपचारः', re.UNICODE),
        re.compile(r'परिहरेत्\s+(\w+)', re.UNICODE),
        re.compile(r'खादयेत्\s+(\w+)', re.UNICODE),
        re.compile(r'पिबेत्\s+(\w+)', re.UNICODE),
    ]
    
    TREATMENT_KEYWORDS: List[str] = [
        'निदानं', 'चिकित्सा', 'उपचारः', 'परिहरेत्', 'खादयेत्',
        'पिबेत्', 'लेहयेत्', 'अनुलोमयेत्', 'सेवयेत्',
        'रसायनं', 'औषधं', 'क्वाथः', 'चूर्णं', 'गुड़िका',
    ]
    
    def __init__(self):
        """Initialize the rule-based treatment extractor."""
        self.patterns = self.TREATMENT_PATTERNS
        self.keywords = self.TREATMENT_KEYWORDS
    
    def extract(self, text: str) -> List[Dict[str, str]]:
        """Extract treatments from Sanskrit text.
        
        Args:
            text: Input Sanskrit text.
            
        Returns:
            List of extracted treatments with metadata.
        """
        results = []
        
        for pattern in self.patterns:
            matches = pattern.findall(text)
            for match in matches:
                results.append({
                    'treatment': match,
                    'method': 'pattern_match',
                    'confidence': 0.7,
                })
        
        for keyword in self.keywords:
            if keyword in text:
                results.append({
                    'keyword': keyword,
                    'type': 'treatment_keyword',
                    'context': self._get_context(text, keyword),
                })
        
        return results
    
    def _get_context(self, text: str, keyword: str, window: int = 5) -> str:
        """Get context around a keyword.
        
        Args:
            text: Full text.
            keyword: Keyword to find.
            window: Number of words around keyword.
            
        Returns:
            Context string.
        """
        words = text.split()
        for i, word in enumerate(words):
            if keyword in word:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                return ' '.join(words[start:end])
        return ''
    
    def extract_with_confidence(self, text: str) -> Dict[str, any]:
        """Extract treatments with confidence scores.
        
        Args:
            text: Input Sanskrit text.
            
        Returns:
            Dictionary with treatments and confidence metrics.
        """
        treatments = self.extract(text)
        
        keyword_count = sum(1 for t in treatments if 'keyword' in t)
        pattern_count = sum(1 for t in treatments if 'treatment' in t and 'method' in t)
        
        confidence = min(1.0, (keyword_count * 0.3 + pattern_count * 0.5))
        
        return {
            'treatments': treatments,
            'confidence': confidence,
            'total_extracted': len(treatments),
        }
    
    def extract_batch(self, texts: List[str]) -> List[List[Dict[str, str]]]:
        """Extract treatments from multiple texts.
        
        Args:
            texts: List of input texts.
            
        Returns:
            List of treatment lists for each text.
        """
        return [self.extract(text) for text in texts]


def demo():
    """Demonstration function for treatment extraction."""
    extractor = RuleBasedExtractor()
    
    sample_text = """
    ज्वरस्य निदानं तिक्ताम्ललवणं परिहरेत्।
    पित्तप्रकोपे शीतलं जलं पिबेत्।
    कफरोगे गुडूचीरसायनं सेवयेत्।
    """
    
    print("Sample Text:")
    print(sample_text)
    print("\nExtracted Treatments:")
    print(extractor.extract(sample_text))
    print("\nWith Confidence:")
    print(extractor.extract_with_confidence(sample_text))


if __name__ == "__main__":
    demo()

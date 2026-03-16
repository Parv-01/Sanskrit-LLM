"""Keyword-based symptom extraction from Sanskrit texts."""

from typing import List, Dict, Set, Optional


class KeywordExtractor:
    """Keyword-based extractor for Ayurvedic symptoms.
    
    Uses predefined Sanskrit symptom vocabulary to identify
    symptoms mentioned in classical Ayurvedic texts.
    
    Example:
        >>> extractor = KeywordExtractor()
        >>> symptoms = extractor.extract("कफप्रकोपः शीतले ज्वरे भवति")
        >>> print(symptoms)
    """
    
    SYMPTOM_KEYWORDS: Set[str] = {
        'ज्वरः', 'ज्वर', 'पित्तं', 'पित्त', 'कफः', 'कफ',
        'वातः', 'वात', 'शीतलं', 'शीतल', 'उष्णं', 'उष्ण',
        'वेदना', 'रुजा', 'शूल', 'दाहः', 'पाकः',
        'कासः', 'कास', 'श्वासः', 'श्वास', 'स्वास',
        'अतीसारः', 'दस्तं', 'मलबन्धः', 'उदरं',
        'हृदयं', 'हृदय', 'मूर्च्छा', 'भ्रमः',
        'मुखं', 'शिरः', 'शरीरं', 'देहः',
    }
    
    DOSHA_KEYWORDS: Dict[str, Set[str]] = {
        'वात': {'वातः', 'वात', 'वातदोषः'},
        'पित्त': {'पित्तं', 'पित्त', 'पित्तदोषः'},
        'कफ': {'कफः', 'कफ', 'कफदोषः'},
    }
    
    def __init__(self, use_ontology: bool = False):
        """Initialize the keyword symptom extractor.
        
        Args:
            use_ontology: Whether to use ontology mapping.
        """
        self.symptom_keywords = self.SYMPTOM_KEYWORDS
        self.dosha_keywords = self.DOSHA_KEYWORDS
        self.use_ontology = use_ontology
    
    def extract(self, text: str) -> List[Dict[str, any]]:
        """Extract symptoms from Sanskrit text.
        
        Args:
            text: Input Sanskrit text.
            
        Returns:
            List of extracted symptoms.
        """
        results = []
        words = text.split()
        
        for word in words:
            if word in self.symptom_keywords:
                results.append({
                    'symptom': word,
                    'type': 'direct_keyword',
                    'confidence': 0.9,
                })
        
        for dosha, keywords in self.dosha_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    results.append({
                        'dosha': dosha,
                        'keyword': keyword,
                        'type': 'dosha_keyword',
                    })
        
        return results
    
    def extract_with_severity(self, text: str) -> List[Dict[str, any]]:
        """Extract symptoms with severity indicators.
        
        Args:
            text: Input Sanskrit text.
            
        Returns:
            Symptoms with severity levels.
        """
        symptoms = self.extract(text)
        
        severity_terms = {
            'अत्यर्थ': 'severe',
            'बहु': 'moderate',
            'किञ्चित': 'mild',
            'मात्रा': 'minor',
        }
        
        for symptom in symptoms:
            for term, severity in severity_terms.items():
                if term in text:
                    symptom['severity'] = severity
        
        return symptoms
    
    def get_dominant_dosha(self, text: str) -> Optional[str]:
        """Determine dominant dosha from text.
        
        Args:
            text: Input Sanskrit text.
            
        Returns:
            Dominant dosha name or None.
        """
        dosha_counts = {dosha: 0 for dosha in self.dosha_keywords}
        
        for dosha, keywords in self.dosha_keywords.items():
            for keyword in keywords:
                dosha_counts[dosha] += text.count(keyword)
        
        if max(dosha_counts.values()) > 0:
            return max(dosha_counts, key=dosha_counts.get)
        
        return None
    
    def extract_batch(self, texts: List[str]) -> List[List[Dict[str, any]]]:
        """Extract symptoms from multiple texts.
        
        Args:
            texts: List of input texts.
            
        Returns:
            List of symptom lists.
        """
        return [self.extract(text) for text in texts]


def demo():
    """Demonstration function for symptom extraction."""
    extractor = KeywordExtractor()
    
    sample_text = "कफप्रकोपः शीतले ज्वरे भवति। पित्तदोषः प्रकोपं करोति।"
    
    print("Sample Text:")
    print(sample_text)
    print("\nExtracted Symptoms:")
    print(extractor.extract(sample_text))
    print("\nDominant Dosha:")
    print(extractor.get_dominant_dosha(sample_text))


if __name__ == "__main__":
    demo()

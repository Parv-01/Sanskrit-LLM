"""Transliteration utilities for Sanskrit."""

from typing import Dict


class Transliterator:
    """Transliteration converter for Sanskrit text.
    
    Supports conversion between Devanagari, IAST, and ISO transliteration
    schemes commonly used in Sanskrit scholarship.
    
    Example:
        >>> transliterator = Transliterator()
        >>> devanagari = transliterator.iast_to_devanagari("āyurvedaḥ")
        >>> print(devanagari)
        आयुर्वेदः
    """
    
    IAST_TO_DEVANAGARI: Dict[str, str] = {
        'a': 'अ', 'ā': 'आ', 'i': 'इ', 'ī': 'ई', 'u': 'उ', 'ū': 'ऊ',
        'ṛ': 'ऋ', 'ṝ': 'ॠ', 'ḷ': 'लृ', 'ḹ': 'लॄ',
        'e': 'ए', 'ai': 'ऐ', 'o': 'ओ', 'au': 'औ',
        'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ', 'ṅ': 'ङ',
        'c': 'च', 'ch': 'छ', 'j': 'ज', 'jh': 'झ', 'ñ': 'ञ',
        'ṭ': 'ट', 'ṭh': 'ठ', 'ḍ': 'ड', 'ḍh': 'ढ', 'ṇ': 'ण',
        't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न',
        'p': 'प', 'ph': 'फ', 'b': 'भ', 'bh': 'ब', 'm': 'म',
        'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 'w': 'व',
        'ś': 'श', 'ṣ': 'ष', 's': 'स', 'h': 'ह',
        'ṃ': 'ं', 'ḥ': 'ः', 'ṁ': 'ं',
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    }
    
    DEVANAGARI_TO_IAST: Dict[str, str] = {v: k for k, v in IAST_TO_DEVANAGARI.items()}
    
    DEVANAGARI_TO_ISO: Dict[str, str] = {
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u', 'ऊ': 'uu',
        'ऋ': 'r', 'ॠ': 'rr', 'लृ': 'lrr', 'लॄ': 'lrrr',
        'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'c', 'छ': 'ch', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
        'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
        'ं': 'm', 'ः': 'h',
    }
    
    def __init__(self):
        """Initialize the transliterator."""
        pass
    
    def iast_to_devanagari(self, text: str) -> str:
        """Convert IAST transliteration to Devanagari.
        
        Args:
            text: Input text in IAST format.
            
        Returns:
            Converted Devanagari text.
        """
        result = text
        for iast, dev in self.IAST_TO_DEVANAGARI.items():
            result = result.replace(iast, dev)
        return result
    
    def devanagari_to_iast(self, text: str) -> str:
        """Convert Devanagari to IAST transliteration.
        
        Args:
            text: Input text in Devanagari.
            
        Returns:
            Converted IAST text.
        """
        result = text
        for dev, iast in self.DEVANAGARI_TO_IAST.items():
            result = result.replace(dev, iast)
        return result
    
    def devanagari_to_iso(self, text: str) -> str:
        """Convert Devanagari to ISO 15919 transliteration.
        
        Args:
            text: Input text in Devanagari.
            
        Returns:
            Converted ISO text.
        """
        result = text
        for dev, iso in self.DEVANAGARI_TO_ISO.items():
            result = result.replace(dev, iso)
        return result
    
    def detect_script(self, text: str) -> str:
        """Detect the script of input text.
        
        Args:
            text: Input text.
            
        Returns:
            Script type: 'devanagari', 'iast', or 'unknown'.
        """
        devanagari_range = range(0x0900, 0x097F)
        
        for char in text:
            if ord(char) in devanagari_range:
                return 'devanagari'
        
        for iast_char in self.IAST_TO_DEVANAGARI.keys():
            if len(iast_char) > 1 and iast_char in text:
                return 'iast'
        
        return 'unknown'


def demo():
    """Demonstration function for transliteration."""
    transliterator = Transliterator()
    
    test_iast = "āyurvedaḥ sarvadā rakṣati"
    test_devanagari = "आयुर्वेदः सर्वदा रक्षति"
    
    print(f"IAST input: {test_iast}")
    print(f"Devanagari: {transliterator.iast_to_devanagari(test_iast)}")
    print(f"ISO: {transliterator.devanagari_to_iso(test_devanagari)}")
    print(f"Script detected: {transliterator.detect_script(test_devanagari)}")


if __name__ == "__main__":
    demo()

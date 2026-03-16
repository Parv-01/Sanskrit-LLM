"""Ontology mapping for Ayurvedic symptoms."""

from typing import Dict, List, Optional, Set


class OntologyMapper:
    """Maps extracted symptoms to Ayurvedic disease ontology.
    
    Provides mapping between extracted symptoms and structured
    disease/syndrome ontology used in Ayurvedic medicine.
    
    Example:
        >>> mapper = OntologyMapper()
        >>> mapped = mapper.map_to_ontology("ज्वरः")
        >>> print(mapped)
    """
    
    SYMPTOM_ONTOLOGY: Dict[str, Dict] = {
        'ज्वरः': {
            'id': 'AYU_001',
            'name': 'Jvara (Fever)',
            'category': 'disease',
            'related_dosha': ['पित्त', 'वात', 'कफ'],
            'synonyms': ['ज्वर', 'तापः'],
        },
        'पित्तं': {
            'id': 'AYU_002',
            'name': 'Pitta Dosa',
            'category': 'dosha',
            'characteristics': ['उष्ण', 'तीक्ष्ण', 'लघु'],
        },
        'कफः': {
            'id': 'AYU_003',
            'name': 'Kapha Dosa',
            'category': 'dosha',
            'characteristics': ['शीतल', 'गुरु', 'मृदु'],
        },
        'वातः': {
            'id': 'AYU_004',
            'name': 'Vata Dosa',
            'category': 'dosha',
            'characteristics': ['शीतल', 'रूक्ष', 'लघु'],
        },
    }
    
    DISEASE_ONTOLOGY: Dict[str, Dict] = {
        'ज्वरभेदः': {
            'id': 'AYU_D001',
            'name': 'Jvara Bheda (Types of Fever)',
            'subtypes': ['वातज्वर', 'पित्तज्वर', 'कफज्वर'],
        },
        'प्रमेहः': {
            'id': 'AYU_D002',
            'name': 'Prameha (Diabetes/Urological disorders)',
        },
    }
    
    def __init__(self):
        """Initialize the ontology mapper."""
        self.symptom_ontology = self.SYMPTOM_ONTOLOGY
        self.disease_ontology = self.DISEASE_ONTOLOGY
    
    def map_to_ontology(self, symptom: str) -> Optional[Dict]:
        """Map a symptom to ontology entry.
        
        Args:
            symptom: Extracted symptom string.
            
        Returns:
            Ontology entry or None.
        """
        if symptom in self.symptom_ontology:
            return self.symptom_ontology[symptom]
        
        for key, entry in self.symptom_ontology.items():
            if 'synonyms' in entry and symptom in entry['synonyms']:
                return entry
        
        return None
    
    def map_batch(self, symptoms: List[str]) -> List[Optional[Dict]]:
        """Map multiple symptoms to ontology.
        
        Args:
            symptoms: List of symptom strings.
            
        Returns:
            List of ontology entries.
        """
        return [self.map_to_ontology(s) for s in symptoms]
    
    def get_related_symptoms(self, symptom: str) -> List[str]:
        """Get symptoms related to the given one.
        
        Args:
            symptom: Input symptom.
            
        Returns:
            List of related symptoms.
        """
        entry = self.map_to_ontology(symptom)
        if entry and 'related_dosha' in entry:
            return entry['related_dosha']
        return []
    
    def search_ontology(self, query: str) -> List[Dict]:
        """Search ontology by keyword.
        
        Args:
            query: Search query.
            
        Returns:
            Matching ontology entries.
        """
        results = []
        query = query.lower()
        
        for key, entry in self.symptom_ontology.items():
            if query in key.lower() or query in entry.get('name', '').lower():
                results.append(entry)
        
        return results
    
    def get_disease_info(self, disease: str) -> Optional[Dict]:
        """Get disease information from ontology.
        
        Args:
            disease: Disease name.
            
        Returns:
            Disease information or None.
        """
        return self.disease_ontology.get(disease)


def demo():
    """Demonstration function for ontology mapping."""
    mapper = OntologyMapper()
    
    symptom = "ज्वरः"
    print(f"Symptom: {symptom}")
    print(f"Mapped: {mapper.map_to_ontology(symptom)}")
    print(f"Related: {mapper.get_related_symptoms(symptom)}")


if __name__ == "__main__":
    demo()

"""Dataset schema definitions for Ayurvedic knowledge."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import json


@dataclass
class SymptomSchema:
    """Schema for symptom data."""
    
    symptom_id: str
    sanskrit_name: str
    english_name: str
    description: str
    dosha_association: List[str] = field(default_factory=list)
    severity_level: Optional[str] = None
    related_symptoms: List[str] = field(default_factory=list)
    ontology_id: Optional[str] = None
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymptomSchema":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TreatmentSchema:
    """Schema for treatment data."""
    
    treatment_id: str
    sanskrit_name: str
    english_name: str
    description: str
    dosage: Optional[str] = None
    duration: Optional[str] = None
    indications: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    related_symptoms: List[str] = field(default_factory=list)
    source_text: Optional[str] = None
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreatmentSchema":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DiseaseSchema:
    """Schema for disease data."""
    
    disease_id: str
    sanskrit_name: str
    english_name: str
    description: str
    category: str
    dosha_involvement: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    treatments: List[str] = field(default_factory=list)
    ontology_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiseaseSchema":
        """Create from dictionary."""
        return cls(**data)


class DatasetSchema:
    """Schema validator for Ayurvedic datasets."""
    
    REQUIRED_SYMPTOM_FIELDS = [
        'symptom_id', 'sanskrit_name', 'english_name', 'description'
    ]
    
    REQUIRED_TREATMENT_FIELDS = [
        'treatment_id', 'sanskrit_name', 'english_name', 'description'
    ]
    
    @staticmethod
    def validate_symptom(data: Dict[str, Any]) -> bool:
        """Validate symptom data.
        
        Args:
            data: Symptom data dictionary.
            
        Returns:
            True if valid.
        """
        return all(field in data for field in DatasetSchema.REQUIRED_SYMPTOM_FIELDS)
    
    @staticmethod
    def validate_treatment(data: Dict[str, Any]) -> bool:
        """Validate treatment data.
        
        Args:
            data: Treatment data dictionary.
            
        Returns:
            True if valid.
        """
        return all(field in data for field in DatasetSchema.REQUIRED_TREATMENT_FIELDS)
    
    @staticmethod
    def create_extraction_record(
        text: str,
        symptoms: List[Dict],
        treatments: List[Dict],
        source: str,
    ) -> Dict[str, Any]:
        """Create a structured extraction record.
        
        Args:
            text: Source text.
            symptoms: Extracted symptoms.
            treatments: Extracted treatments.
            source: Source identifier.
            
        Returns:
            Structured record.
        """
        return {
            'source_text': text,
            'source_id': source,
            'extracted_symptoms': symptoms,
            'extracted_treatments': treatments,
            'extraction_metadata': {
                'symptom_count': len(symptoms),
                'treatment_count': len(treatments),
            }
        }


def demo():
    """Demonstration function for schemas."""
    symptom = SymptomSchema(
        symptom_id="SYM_001",
        sanskrit_name="ज्वरः",
        english_name="Fever",
        description="Elevated body temperature",
        dosha_association=["पित्त", "वात", "कफ"],
    )
    
    print("Symptom Schema:")
    print(symptom.to_json())


if __name__ == "__main__":
    demo()

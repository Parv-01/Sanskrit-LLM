"""ML classifier for Prakriti (Ayurvedic constitution) prediction."""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class PrakritiResult:
    """Result of Prakriti prediction."""
    
    predominant_dosha: str
    secondary_dosha: Optional[str]
    scores: Dict[str, float]
    description: str


class PrakritiClassifier:
    """Machine learning classifier for Prakriti prediction.
    
    Predicts Ayurvedic body constitution (Prakriti) based on
    symptom descriptions using feature extraction and classification.
    
    Note:
        This is a placeholder implementation. Production use requires
        training data and proper model training.
    
    Example:
        >>> classifier = PrakritiClassifier()
        >>> result = classifier.predict(["शीतलं ज्वरः", "पित्तप्रकोपः"])
        >>> print(result.predominant_dosha)
    """
    
    PRAKRITI_TYPES: List[str] = [
        'वातिक', 'पित्तिक', 'कफिक', 'वात-पित्तिक', 'वात-कफिक', 'पित्त-कफिक',
        'त्रिदोषिक',
    ]
    
    FEATURE_KEYWORDS: Dict[str, List[str]] = {
        'वात': ['वात', 'रूक्ष', 'शीतल', 'चल', 'लघु', 'विचल'],
        'पित्त': ['पित्त', 'उष्ण', 'तीक्ष्ण', 'दाह', 'रक्त', 'पीत'],
        'कफ': ['कफ', 'शीतल', 'गुरु', 'मृदु', 'स्थिर', 'स्निग्ध'],
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the Prakriti classifier.
        
        Args:
            model_path: Path to trained model (optional).
        """
        self.model_path = model_path
        self.model = None
        self.is_trained = False
    
    def _extract_features(self, symptoms: List[str]) -> np.ndarray:
        """Extract features from symptom text.
        
        Args:
            symptoms: List of symptom strings.
            
        Returns:
            Feature vector.
        """
        feature_vector = np.zeros(9)
        
        combined_text = ' '.join(symptoms)
        
        for i, (dosha, keywords) in enumerate(self.FEATURE_KEYWORDS.items()):
            for keyword in keywords:
                if keyword in combined_text:
                    feature_vector[i] += 1
        
        feature_vector[3:6] = feature_vector[0:3]
        feature_vector[6:9] = feature_vector[0:3] / (np.sum(feature_vector[0:3]) + 1e-6)
        
        return feature_vector
    
    def _rule_based_predict(self, features: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Rule-based prediction (placeholder).
        
        Args:
            features: Feature vector.
            
        Returns:
            Predicted Prakriti and scores.
        """
        vata_score = features[0]
        pitta_score = features[1]
        kapha_score = features[2]
        
        scores = {
            'वात': float(vata_score),
            'पित्त': float(pitta_score),
            'कफ': float(kapha_score),
        }
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        if max(scores.values()) == 0:
            return 'त्रिदोषिक', scores
        
        predominant = max(scores, key=scores.get)
        
        return predominant, scores
    
    def predict(self, symptoms: List[str]) -> PrakritiResult:
        """Predict Prakriti from symptoms.
        
        Args:
            symptoms: List of symptom descriptions.
            
        Returns:
            Prakriti prediction result.
        """
        features = self._extract_features(symptoms)
        
        predominant, scores = self._rule_based_predict(features)
        
        sorted_doshas = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        secondary = sorted_doshas[1][0] if len(sorted_doshas) > 1 and sorted_doshas[1][1] > 0 else None
        
        descriptions = {
            'वात': 'वातिक प्रकृति - वायु तत्व प्रधान। चल, शीतल, रूक्ष गुण।',
            'पित्त': 'पित्तिक प्रकृति - अग्नि तत्व प्रधान। उष्ण, तीक्ष्ण गुण।',
            'कफ': 'कफिक प्रकृति - जल तत्व प्रधान। शीतल, गुरु, स्थिर गुण।',
            'वात-पित्तिक': 'वात-पित्त प्रधान मिश्रित प्रकृति।',
            'वात-कफिक': 'वात-कफ प्रधान मिश्रित प्रकृति।',
            'पित्त-कफिक': 'पित्त-कफ प्रधान मिश्रित प्रकृति।',
            'त्रिदोषिक': 'त्रिदोष समान प्रकृति।',
        }
        
        return PrakritiResult(
            predominant_dosha=predominant,
            secondary_dosha=secondary,
            scores=scores,
            description=descriptions.get(predominant, ''),
        )
    
    def predict_batch(self, symptom_lists: List[List[str]]) -> List[PrakritiResult]:
        """Predict Prakriti for multiple symptom sets.
        
        Args:
            symptom_lists: List of symptom lists.
            
        Returns:
            List of prediction results.
        """
        return [self.predict(symptoms) for symptoms in symptom_lists]
    
    def get_treatment_recommendations(self, prakriti: str) -> List[str]:
        """Get lifestyle recommendations for Prakriti.
        
        Args:
            prakriti: Predicted Prakriti type.
            
        Returns:
            List of recommendations.
        """
        recommendations = {
            'वात': [
                'नियमित शयन',
                'तेल मालिश',
                'गर्म आहार',
                'वायु नियंत्रण',
            ],
            'पित्त': [
                'शीतल आहार',
                'तीत्र धूप से बचाव',
                'पित्त शमन',
                'मध्यम व्यायाम',
            ],
            'कफ': [
                'हल्का आहार',
                'व्यायाम',
                'जल सेवन',
                'कफ नाशक',
            ],
        }
        
        return recommendations.get(prakriti, [])


def demo():
    """Demonstration function for Prakriti prediction."""
    classifier = PrakritiClassifier()
    
    symptoms = [
        "शीतलं देहं रूक्षं च",
        "पित्तप्रकोपः",
    ]
    
    print("Input symptoms:", symptoms)
    result = classifier.predict(symptoms)
    
    print(f"\nPredicted Prakriti: {result.predominant_dosha}")
    print(f"Secondary: {result.secondary_dosha}")
    print(f"Scores: {result.scores}")
    print(f"Description: {result.description}")


if __name__ == "__main__":
    demo()

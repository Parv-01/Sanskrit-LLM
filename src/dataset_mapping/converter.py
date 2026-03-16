"""JSON conversion utilities for dataset mapping."""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class JSONConverter:
    """Converts extraction results to structured JSON format.
    
    Handles conversion of raw extraction results to various
    JSON formats suitable for downstream processing.
    
    Example:
        >>> converter = JSONConverter()
        >>> json_data = converter.extractions_to_json(extractions)
    """
    
    def __init__(self):
        """Initialize the JSON converter."""
        pass
    
    def extractions_to_json(
        self,
        extractions: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> str:
        """Convert extractions to JSON format.
        
        Args:
            extractions: List of extraction dictionaries.
            output_path: Optional path to save JSON.
            
        Returns:
            JSON string.
        """
        data = {
            'extractions': extractions,
            'metadata': {
                'count': len(extractions),
                'version': '1.0',
            }
        }
        
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        if output_path:
            Path(output_path).write_text(json_str, encoding='utf-8')
        
        return json_str
    
    def create_knowledge_graph(
        self,
        symptoms: List[Dict],
        treatments: List[Dict],
    ) -> Dict[str, Any]:
        """Create a knowledge graph structure.
        
        Args:
            symptoms: Extracted symptoms.
            treatments: Extracted treatments.
            
        Returns:
            Knowledge graph dictionary.
        """
        nodes = []
        edges = []
        
        for symptom in symptoms:
            nodes.append({
                'id': symptom.get('id', f"symptom_{len(nodes)}"),
                'type': 'symptom',
                'label': symptom.get('sanskrit_name', symptom.get('symptom', '')),
            })
        
        for treatment in treatments:
            nodes.append({
                'id': treatment.get('id', f"treatment_{len(nodes)}"),
                'type': 'treatment',
                'label': treatment.get('sanskrit_name', treatment.get('treatment', '')),
            })
            
            if 'related_symptoms' in treatment:
                for rel_symptom in treatment['related_symptoms']:
                    edges.append({
                        'source': treatment['id'],
                        'target': rel_symptom,
                        'relation': 'treats',
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
        }
    
    def convert_to_jsonl(
        self,
        records: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """Convert records to JSONL format.
        
        Args:
            records: List of record dictionaries.
            output_path: Path to save JSONL file.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    def load_jsonl(self, input_path: str) -> List[Dict[str, Any]]:
        """Load records from JSONL format.
        
        Args:
            input_path: Path to JSONL file.
            
        Returns:
            List of record dictionaries.
        """
        records = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        return records
    
    def merge_datasets(
        self,
        datasets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge multiple datasets.
        
        Args:
            datasets: List of dataset dictionaries.
            
        Returns:
            Merged dataset.
        """
        merged = {
            'symptoms': [],
            'treatments': [],
            'diseases': [],
        }
        
        for dataset in datasets:
            if 'symptoms' in dataset:
                merged['symptoms'].extend(dataset['symptoms'])
            if 'treatments' in dataset:
                merged['treatments'].extend(dataset['treatments'])
            if 'diseases' in dataset:
                merged['diseases'].extend(dataset['diseases'])
        
        return merged


def demo():
    """Demonstration function for JSON conversion."""
    converter = JSONConverter()
    
    sample_extraction = {
        'symptom': 'ज्वरः',
        'treatment': 'तिक्ताम्ललवण',
    }
    
    json_output = converter.extractions_to_json([sample_extraction])
    print(json_output)


if __name__ == "__main__":
    demo()

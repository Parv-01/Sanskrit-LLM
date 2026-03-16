# Dataset Schema

This document defines the schemas for structured datasets extracted from Ayurvedic texts.

## Symptom Schema

```json
{
  "symptom_id": "SYM_001",
  "sanskrit_name": "ज्वरः",
  "english_name": "Fever",
  "description": "Elevated body temperature associated with disease",
  "dosha_association": ["पित्त", "वात", "कफ"],
  "severity_level": "moderate",
  "related_symptoms": ["दाहः", "वेदना"],
  "ontology_id": "AYU_001",
  "confidence_score": 0.95,
  "source_text": "...",
  "extraction_method": "keyword"
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `symptom_id` | string | Yes | Unique identifier (SYM_XXX) |
| `sanskrit_name` | string | Yes | Original Sanskrit term |
| `english_name` | string | Yes | English translation |
| `description` | string | Yes | Detailed description |
| `dosha_association` | array | No | Associated doshas |
| `severity_level` | string | No | mild/moderate/severe |
| `related_symptoms` | array | No | Related symptom IDs |
| `ontology_id` | string | No | Ontology reference |
| `confidence_score` | float | Yes | 0.0-1.0 confidence |
| `source_text` | string | Yes | Original text |
| `extraction_method` | string | Yes | keyword/llm/manual |

## Treatment Schema

```json
{
  "treatment_id": "TRT_001",
  "sanskrit_name": "तिक्ताम्ललवणं",
  "english_name": "Bitter, sour, salty substances",
  "description": "Bitter, sour, and salty tastes recommended for balancing",
  "dosage": "परिमाणं योग्यं",
  "duration": "साताहः",
  "indications": ["ज्वरः", "पित्तदोषः"],
  "contraindications": ["वातरोगः"],
  "related_symptoms": ["SYM_001"],
  "source_text": "...",
  "confidence_score": 0.85,
  "source_reference": "चरकसंहिता"
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `treatment_id` | string | Yes | Unique identifier (TRT_XXX) |
| `sanskrit_name` | string | Yes | Sanskrit treatment name |
| `english_name` | string | Yes | English translation |
| `description` | string | Yes | Treatment description |
| `dosage` | string | No | Recommended dosage |
| `duration` | string | No | Treatment duration |
| `indications` | array | No | Conditions treated |
| `contraindications` | array | No | Avoided conditions |
| `related_symptoms` | array | No | Treated symptoms |
| `source_text` | string | Yes | Original text |
| `confidence_score` | float | Yes | 0.0-1.0 confidence |
| `source_reference` | string | No | Source text name |

## Disease Schema

```json
{
  "disease_id": "DIS_001",
  "sanskrit_name": "ज्वरभेदः",
  "english_name": "Types of fever",
  "description": "Classification of fever based on dosha involvement",
  "category": "ज्वररोग",
  "dosha_involvement": ["वात", "पित्त", "कफ"],
  "symptoms": ["SYM_001", "SYM_002"],
  "treatments": ["TRT_001", "TRT_002"],
  "ontology_id": "AYU_D001"
}
```

## Extraction Record Schema

```json
{
  "record_id": "EXT_001",
  "source_text": "...",
  "source_id": "charaka_samhita_1",
  "extracted_symptoms": [...],
  "extracted_treatments": [...],
  "extraction_metadata": {
    "extraction_date": "2026-03-16",
    "symptom_count": 5,
    "treatment_count": 3,
    "method": "rule_based"
  }
}
```

## File Formats

### JSON (Recommended)
```
data/datasets/symptoms.json
data/datasets/treatments.json
data/datasets/diseases.json
data/datasets/extractions.jsonl
```

### JSONL (Line-delimited for large datasets)
```
data/datasets/extractions.jsonl
```

## Validation Rules

1. All required fields must be present
2. IDs must follow naming convention (SYM_, TRT_, DIS_, EXT_)
3. Confidence scores must be between 0.0 and 1.0
4. Dosha associations must be from valid set: वात, पित्त, कफ
5. Source text must not be empty

## Export Formats

Supported export formats:
- JSON (default)
- JSONL (for streaming)
- CSV (tabular data)
- RDF (linked data)

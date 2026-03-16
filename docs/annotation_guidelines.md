# Annotation Guidelines

Guidelines for researchers annotating Sanskrit Ayurvedic texts.

## Introduction

This document provides guidelines for manually annotating Sanskrit Ayurvedic texts for symptom and treatment extraction.

## Annotator Qualifications

- Knowledge of Sanskrit language (reading Devanagari)
- Basic understanding of Ayurvedic concepts
- Familiarity with doshas (वात, पित्त, कफ)
- Training on annotation tool

## Text Selection Criteria

### Source Texts
1. Classical Ayurvedic texts (चरकसंहिता, सुश्रुतसंहिता, अष्टांगहृदयं)
2. Secondary commentaries
3. Modern Ayurvedic literature

### Selection Requirements
- Minimum 1000 verses/passages
- Clear attribution
- Legible text (no OCR errors)

## Annotation Workflow

### 1. Preprocessing
- Verify text quality
- Segment into verses/shlokas
- Assign source IDs

### 2. Symptom Annotation

#### Guidelines
1. Identify all symptoms mentioned
2. Record Sanskrit term exactly as in text
3. Note context (surrounding text)
4. Assign severity if mentioned
5. Map to dosha if indicated

#### Example
```
Text: "कफप्रकोपः शीतले ज्वरे भवति"

Annotation:
- symptom_id: [auto-generated]
- sanskrit_name: ज्वरः
- context: "शीतले ज्वरे" (in cold fever)
- severity: none mentioned
- dosha: कफ
```

### 3. Treatment Annotation

#### Guidelines
1. Identify treatment/ remedy mentioned
2. Record Sanskrit term exactly
3. Note dosage if mentioned
4. Note duration if mentioned
5. Link to indication (disease/symptom)

#### Example
```
Text: "ज्वरस्य निदानं तिक्ताम्ललवणं परिहरेत्"

Annotation:
- treatment_id: [auto-generated]
- sanskrit_name: तिक्ताम्ललवणं
- description: bitter, sour, salty substances
- indication: ज्वर (fever)
- action: परिहरेत् (avoid)
```

### 4. Quality Control

#### Review Process
1. Self-review annotations
2. Cross-review with second annotator
3. Resolve disagreements through discussion
4. Final review by senior researcher

#### Agreement Metrics
- Inter-annotator agreement > 80%
- Review accuracy > 95%

## Annotation Tool

Recommended tools:
1. Doccano (open source)
2. Label Studio
3. Custom web interface

## Data Format

### Manual Annotations (CSV)
```csv
source_id,verse_id,text_type,content,annotation_type,value,notes
CS_001,1.1,verse,कफप्रकोपः...,symptom,ज्वरः,fever symptom
CS_001,1.1,verse,कफप्रकोपः...,treatment,तिक्ताम्ललवणं,bitter/sour/salty
```

### Export to JSON
```json
{
  "annotations": [
    {
      "id": "ANN_001",
      "source": "CS_001",
      "type": "symptom",
      "value": "ज्वरः",
      "metadata": {...}
    }
  ]
}
```

## Common Challenges

### Sandhi Compounds
- Split compound words when possible
- Note original form if unsplit

### Ambiguous Terms
- Research classical meanings
- Note uncertainty level
- Flag for review

### OCR Errors
- Verify against published editions
- Mark as uncertain if unsure

## Ethical Considerations

1. **Attribution**: Credit original authors
2. **Context**: Preserve textual context
3. **Accuracy**: Prioritize precision over quantity
4. **Documentation**: Record all decisions

## Resources

- Sanskrit dictionaries: [Links]
- Ayurvedic ontologies: [Links]
- Reference texts: [Links]

## Contact

For questions about annotations:
- Project Lead: [TBD]
- Annotation Coordinator: [TBD]

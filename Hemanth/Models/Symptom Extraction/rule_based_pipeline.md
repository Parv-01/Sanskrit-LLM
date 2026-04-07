# Multilingual Symptom Extractor

## Pipeline & Core Concept

The core concept of this pipeline is **Rule-Based Natural Language Processing (NLP)**. It utilizes a hierarchical dictionary-matching approach combined with linguistic rules to extract medical entities from English, Hindi, and Hinglish text.

### Pipeline Stages

1. **Text Normalization**: Cleans the input by handling Unicode normalization and lowercase conversion.
2. **Hinglish-to-English Mapping**: Uses a custom dictionary to translate Romanized Hindi (Hinglish) phrases into canonical English medical terms.
3. **Negation Detection (NegEx)**: Implements the NegEx algorithm to identify negation cues (e.g., "no", "without") and stops the negation scope when encountering conjunctions or pre-defined terminators.
4. **Fuzzy String Matching**: Employs a fuzzy backend to match input tokens against a standardized symptom dictionary, allowing for minor spelling variations or transliteration inconsistencies.

## How to Run

This script is designed for CLI interaction and testing.
1. Dependencies:
   pip install rapidfuzz
2. Execution:
    python nl_to_symp.py
3. Enter the number of test cases and provide the text strings when prompted.

## References

1. Negation Detection  (NegEx — Chapman et al., 2001)
   "A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries."
2. Fuzzy Matching  (RapidFuzz / difflib fallback)
   Bachmann & Sperl (2021), RapidFuzz.
   https://github.com/maxbachmann/RapidFuzz
3. Hinglish NER  (Mave et al., 2018)
   "Language Identification and Named Entity Recognition in Hinglish Code Mixed Tweets." NAACL 2018.
   https://aclanthology.org/N18-3008/

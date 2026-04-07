# Hybrid Multilingual Symptom Extraction Pipeline

## Pipeline & Core Concept

The core concept of this pipeline is **Hybrid Semantic Retrieval**. It combines traditional rule-based preprocessing with **Semantic Embeddings** and vector similarity. By using TF-IDF and cosine similarity, the system can understand the "intent" or "context" of a phrase even if it doesn't match the dictionary exactly.

### Pipeline Stages

1. **N-Gram Windowing**: Uses a minimum 2-token window to ensure that single, common words do not trigger false positive matches.
2. **Phrase Normalization**: A pre-expansion step that rewrites irregular phrases (e.g., "eyes pain" to "eye pain") and combines body parts with descriptors (e.g., "throat" + "ache" to "sore throat").
3. **Vector Indexing**: Builds a TF-IDF vector index of canonical symptoms to perform high-speed similarity lookups.
4. **Semantic Scoring**: Calculates the cosine similarity between the input text and the symptom lexicon, returning matches that meet a specific confidence threshold.

## How to Run
This module requires scientific computing libraries for the vector operations.

1. Dependencies:
   pip install numpy scikit-learn
2. Execution:
    python symptoms.py

## References

1. Karpukhin, V., et al. (2020). 
    Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
2. Reimers, N., & Gurevych, I. (2019). 
    Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
3. Khanuja, S., et al. (2021). 
    MuRIL: Multilingual Representations for Indian Languages. arXiv.
4. Chen, Q., et al. (2019). 
    BioSentVec: Creating sentence embeddings for biomedical texts. IEEE International Conference on Healthcare Informatics.
5. Negation Detection  (NegEx — Chapman et al., 2001)
    "A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries."
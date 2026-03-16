# System Architecture

This document provides a detailed overview of the Ayurveda Sanskrit LLM system architecture.

## Overview

The system implements a complete NLP pipeline for processing classical Sanskrit Ayurvedic texts, extracting structured knowledge, and enabling question answering through Retrieval-Augmented Generation (RAG).

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. INGESTION                                                               │
│     raw_texts/ ──────► Text Loading ──────► Preprocessing                  │
│                                                                             │
│  2. PROCESSING                                                              │
│     Tokenization ──► Transliteration ──► Sandhi Splitting                  │
│                                                                             │
│  3. EXTRACTION                                                             │
│     Symptom Extraction ──► Treatment Extraction ──► Ontology Mapping        │
│                                                                             │
│  4. STRUCTURING                                                             │
│     Schema Validation ──► JSON Conversion ──► Dataset Storage              │
│                                                                             │
│  5. INDEXING                                                                │
│     Embedding Generation ──► Vector Indexing ──► FAISS Storage              │
│                                                                             │
│  6. RETRIEVAL                                                               │
│     Query Embedding ──► Similarity Search ──► Context Assembly             │
│                                                                             │
│  7. GENERATION                                                              │
│     LLM Prompt ──► Answer Generation ──► Response                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### Sanskrit Processing (`src/sanskrit_processing/`)

| Module | Purpose |
|--------|---------|
| `tokenizer.py` | Tokenizes Devanagari Sanskrit text |
| `transliteration.py` | Converts between Devanagari, IAST, ISO |
| `sandhi_splitter.py` | Splits compound words |

### Treatment Extraction (`src/treatment_extraction/`)

| Module | Purpose |
|--------|---------|
| `rule_extractor.py` | Pattern-based treatment extraction |
| `llm_extractor.py` | LLM-based treatment extraction |

### Symptom Extraction (`src/symptom_extraction/`)

| Module | Purpose |
|--------|---------|
| `keyword_extractor.py` | Keyword-based symptom extraction |
| `ontology_mapper.py` | Maps symptoms to disease ontology |

### Dataset Mapping (`src/dataset_mapping/`)

| Module | Purpose |
|--------|---------|
| `schema.py` | Defines data schemas |
| `converter.py` | Converts data to JSON formats |

### RAG Pipeline (`src/rag_pipeline/`)

| Module | Purpose |
|--------|---------|
| `embeddings.py` | Generates text embeddings |
| `vector_store.py` | FAISS/in-memory vector storage |
| `query_engine.py` | RAG query pipeline |

### Prakriti Prediction (`src/prakriti_prediction/`)

| Module | Purpose |
|--------|---------|
| `classifier.py` | ML-based Prakriti prediction |

## RAG Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   USER QUERY ─────► EMBEDDING (Query)                                      │
│                           │                                                 │
│                           ▼                                                 │
│                    ┌──────────────┐                                        │
│                    │   RETRIEVAL   │                                        │
│                    │  Vector Store │◄──── Knowledge Base                   │
│                    │    (FAISS)     │      (Indexed Texts)                  │
│                    └──────┬────────┘                                        │
│                           │                                                 │
│                           ▼                                                 │
│                    ┌──────────────┐                                        │
│                    │    CONTEXT    │                                        │
│                    │  ASSEMBLY     │                                        │
│                    └──────┬────────┘                                        │
│                           │                                                 │
│                           ▼                                                 │
│                    ┌──────────────┐                                        │
│                    │      LLM      │                                        │
│                    │  GENERATION   │                                        │
│                    └──────┬────────┘                                        │
│                           │                                                 │
│                           ▼                                                 │
│                      ANSWER                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Storage

### Raw Texts (`data/raw_texts/`)
- Original Sanskrit manuscripts in Devanagari
- Various file formats (txt, pdf, scanned images)

### Processed Texts (`data/processed_texts/`)
- Cleaned and tokenized text
- Transliteration outputs

### Datasets (`data/datasets/`)
- Structured JSON exports
- Symptom-Treatment mappings
- Disease ontologies

## Configuration

Environment variables (`.env`):
```
OPENAI_API_KEY=your_api_key
MODEL_PATH=path/to/model
DATA_PATH=path/to/data
```

## Extension Points

1. **New Extraction Rules**: Add patterns in `treatment_extraction/rule_extractor.py`
2. **Ontology Expansion**: Update `symptom_extraction/ontology_mapper.py`
3. **Model Replacement**: Swap embedding models in `rag_pipeline/embeddings.py`
4. **LLM Backend**: Modify `rag_pipeline/query_engine.py` for different LLMs

# Ayurveda Sanskrit LLM

A research project developing a prototype AI system that extracts treatment knowledge from classical Sanskrit Ayurvedic texts and maps it to diseases and symptoms using a Retrieval-Augmented Generation (RAG) pipeline.

The goal is to bridge classical Ayurvedic knowledge with modern AI systems by transforming Sanskrit medical literature into a structured knowledge base that supports symptom-based treatment retrieval and conversational AI applications.

This project is part of ongoing research in AI for Traditional Medicine and Sanskrit NLP.


--------------------------------------------------
PROJECT OBJECTIVE
--------------------------------------------------

The system aims to:

1. Extract treatment knowledge from Sanskrit Ayurvedic texts
2. Map symptoms and diseases using standardized Ayurveda disease ontology
3. Create structured datasets from classical texts
4. Build a Retrieval-Augmented Generation (RAG) pipeline
5. Enable LLM-based question answering
6. Optionally predict Prakriti (Vata / Pitta / Kapha)


--------------------------------------------------
EXAMPLE SYSTEM OUTPUT
--------------------------------------------------

Input Symptoms:

- fever
- burning sensation
- thirst


Expected Output:

Diagnosis:
Pittaja Jwara

Treatment:
Guduchi decoction

Evidence Source:
Charaka Samhita – Jwara Chikitsa


--------------------------------------------------
SYSTEM ARCHITECTURE
--------------------------------------------------

AYURVEDA SANSKRIT LLM PIPELINE

1. Sanskrit Text Ingestion
   Load classical Ayurvedic texts

2. NLP Preprocessing
   Tokenization → Transliteration → Sandhi Splitting

3. Symptom Extraction
   Sanskrit symptom detection → Ontology mapping

4. Treatment Extraction
   Rule-based + LLM extraction from Sanskrit verses

5. Dataset Construction
   Structured JSON datasets

6. Embedding Generation
   Sanskrit-aware embeddings + vector database indexing

7. RAG Query Engine
   Retrieval → Context assembly → LLM reasoning

8. Prakriti Prediction (Optional)
   Symptom features → ML classifier → Dosha prediction


--------------------------------------------------
REPOSITORY STRUCTURE
--------------------------------------------------

Sanskrit-LLM/

README.md  
requirements.txt  
.gitignore  

data/

    raw_texts/
        Sanskrit Ayurvedic source texts

    processed_texts/
        Cleaned and segmented Sanskrit verses

    datasets/
        disease_ontology.json
        symptom_dataset.json
        treatment_dataset.json
        knowledge_base.json


src/

    sanskrit_processing/
        text_cleaning.py
        verse_segmentation.py
        sandhi_splitter.py

    treatment_extraction/
        keyword_detector.py
        treatment_extractor.py

    symptom_extraction/
        symptom_extractor.py

    dataset_mapping/
        ontology_builder.py
        dataset_mapper.py

    rag_pipeline/
        embeddings.py
        vector_db.py
        rag_engine.py

    prakriti_prediction/
        feature_engineering.py
        model_training.py


notebooks/

    data_exploration.ipynb
    pipeline_testing.ipynb
    model_experiments.ipynb


experiments/

    rag_experiments/
    prakriti_experiments/


docs/

    system_architecture.md
    dataset_schema.md
    annotation_guidelines.md


paper/

    paper_draft.md


figures/


references/



--------------------------------------------------
DATASET STRUCTURE
--------------------------------------------------

Example dataset entry:

{
  "disease": "Pittaja Jwara",
  "symptoms": ["fever", "burning sensation", "thirst"],
  "treatment": ["Guduchi decoction"],
  "source_text": "Charaka Samhita"
}


--------------------------------------------------
PARALLEL DEVELOPMENT STRATEGY
--------------------------------------------------

To allow efficient collaboration, the project is divided into two independent pipelines.


--------------------------------------------------
PIPELINE A — SANSKRIT TEXT PROCESSING
--------------------------------------------------

Developer: Hemanth

Responsibilities:

- Collect Sanskrit Ayurvedic texts
- Clean OCR or raw text
- Segment texts into verses
- Extract treatment information from verses


Directories owned by Hemanth:

data/raw_texts/  
data/processed_texts/  
src/sanskrit_processing/  
src/treatment_extraction/


Expected outputs:

data/processed_texts/verses_dataset.json  
data/datasets/treatment_dataset.json


--------------------------------------------------
PIPELINE B — DISEASE ONTOLOGY AND SYMPTOM MAPPING
--------------------------------------------------

Developer: Gowtham

Responsibilities:

- Process Ayurveda disease datasets
- Build disease ontology
- Create symptom datasets
- Map symptoms to diseases


Directories owned by Gowtham:

data/datasets/  
src/symptom_extraction/  
src/dataset_mapping/


Expected outputs:

data/datasets/disease_ontology.json  
data/datasets/symptom_dataset.json


--------------------------------------------------
FINAL INTEGRATION
--------------------------------------------------

treatment_dataset
+
symptom_dataset
+
disease_ontology

↓

knowledge_base.json

This dataset powers the RAG retrieval engine.


--------------------------------------------------
GIT WORKFLOW
--------------------------------------------------

Branch structure:

main  
│  
dev  
│  
├── feature/hemanth-text-pipeline  
└── feature/gowtham-disease-ontology  


Rules:

Hemanth commits only to:

feature/hemanth-text-pipeline


Gowtham commits only to:

feature/gowtham-disease-ontology


Pull request flow:

feature branch → dev → main


Only the Project Lead merges code into main.


--------------------------------------------------
PULL REQUEST REQUIREMENTS
--------------------------------------------------

Every PR must include:

- Description of the feature implemented
- Files modified
- Dataset updates (if any)
- Example outputs

Example commit message:

feat: add verse segmentation for Charaka Samhita


--------------------------------------------------
INSTALLATION
--------------------------------------------------

Prerequisites:

Python 3.10+  
Git  


Setup:

git clone <repository-url>

cd Sanskrit-LLM

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt


--------------------------------------------------
EXAMPLE USAGE
--------------------------------------------------

Sanskrit Tokenization Example:

from src.sanskrit_processing import SanskritTokenizer

tokenizer = SanskritTokenizer()

tokens = tokenizer.tokenize("आयुर्वेदः सर्वदा रक्षति")

print(tokens)



Treatment Extraction Example:

from src.treatment_extraction import RuleBasedExtractor

extractor = RuleBasedExtractor()

treatments = extractor.extract("ज्वरस्य कषाय पानं हितम्")

print(treatments)



RAG Query Example:

from src.rag_pipeline import RAGEngine

engine = RAGEngine()

answer = engine.query("What treatments are recommended for fever?")

print(answer)



--------------------------------------------------
WEEKLY DELIVERABLES
--------------------------------------------------

Week 2

Hemanth:

verses_dataset.json  
treatment_verse_candidates.json  


Gowtham:

disease_ontology.json  
symptom_dataset.json  



--------------------------------------------------
TEAM
--------------------------------------------------

Project Lead  
Junior Research Associate  

Student Developer  
Hemanth  

Student Developer  
Gowtham  



--------------------------------------------------
LICENSE
--------------------------------------------------

This project is intended for academic research and experimentation purposes.



--------------------------------------------------
CITATION
--------------------------------------------------

If you use this work, please cite:

@misc{ayurveda_sanskrit_llm_2026,
title = {Ayurveda Sanskrit LLM},
author = {Research Team},
year = {2026},
institution = {Research Institution}
}

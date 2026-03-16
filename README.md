# Ayurveda Sanskrit LLM

A research project developing a prototype AI system that extracts treatment information from classical Sanskrit Ayurvedic texts and maps it with symptoms and disease ontology using a Retrieval-Augmented Generation (RAG) pipeline.

## System Architecture

The system implements a complete NLP pipeline for Sanskrit Ayurvedic text processing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AYURVEDA SANSKRIT LLM PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SANSKRIT TEXT INGESTION                                                │
│     └─> Load raw Sanskrit texts from data/raw_texts/                       │
│                                                                             │
│  2. NLP PREPROCESSING                                                       │
│     └─> Tokenization → Transliteration → Sandhi Splitting                 │
│                                                                             │
│  3. SYMPTOM EXTRACTION                                                      │
│     └─> Keyword-based extraction → Ontology mapping                        │
│                                                                             │
│  4. TREATMENT EXTRACTION                                                   │
│     └─> Rule-based extraction → LLM augmentation                           │
│                                                                             │
│  5. DATASET MAPPING                                                        │
│     └─> Structured JSON conversion → Schema validation                     │
│                                                                             │
│  6. EMBEDDING GENERATION                                                   │
│     └─> Sanskrit-aware embeddings → Vector database indexing             │
│                                                                             │
│  7. RAG QUERY ENGINE                                                       │
│     └─> Retrieval → Context assembly → LLM generation                     │
│                                                                             │
│  8. PRAKRITI PREDICTION (Optional)                                         │
│     └─> Symptom features → ML classifier → Dosha prediction              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
Sanskrit-LLM/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── data/
│   ├── raw_texts/              # Original Sanskrit manuscripts
│   ├── processed_texts/        # Cleaned and tokenized text
│   └── datasets/               # Structured extracted knowledge
├── src/
│   ├── sanskrit_processing/    # Text preprocessing utilities
│   ├── treatment_extraction/  # Treatment extraction modules
│   ├── symptom_extraction/    # Symptom extraction modules
│   ├── dataset_mapping/       # Data transformation utilities
│   ├── rag_pipeline/          # RAG implementation
│   └── prakriti_prediction/    # Dosha prediction module
├── notebooks/
│   ├── data_exploration.ipynb    # Text analysis & visualization
│   ├── pipeline_testing.ipynb    # End-to-end pipeline tests
│   └── model_experiments.ipynb   # Embedding & LLM experiments
├── experiments/
│   ├── rag_experiments/        # RAG evaluation results
│   └── prakriti_experiments/   # Classification experiments
├── docs/
│   ├── system_architecture.md   # Detailed architecture docs
│   ├── dataset_schema.md       # Data schema definitions
│   └── annotation_guidelines.md # Text annotation rules
├── paper/
│   └── paper_draft.md          # Research paper template
├── figures/                    # Visualization assets
└── references/                 # Related papers & resources
```

## Installation Instructions

### Prerequisites
- Python 3.10+
- Git
- (Optional) CUDA-capable GPU for transformer models

### Setup Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd Sanskrit-LLM

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download Sanskrit models
# Transformers will be downloaded automatically when first used
```

## Usage Instructions

### Processing Sanskrit Text

```python
from src.sanskrit_processing import SanskritTokenizer

tokenizer = SanskritTokenizer()
tokens = tokenizer.tokenize("आयुर्वेदः सर्वदा रक्षति")
print(tokens)
```

### Running Extraction Pipeline

```python
from src.treatment_extraction import RuleBasedExtractor
from src.symptom_extraction import KeywordExtractor

# Initialize extractors
treatment_extractor = RuleBasedExtractor()
symptom_extractor = KeywordExtractor()

# Process text
text = "कफप्रकोपः शीतले ज्वरे भवति"
treatments = treatment_extractor.extract(text)
symptoms = symptom_extractor.extract(text)
```

### Running RAG Query

```python
from src.rag_pipeline import RAGQueryEngine

engine = RAGQueryEngine()
answer = engine.query("What treatments are recommended for fever?")
print(answer)
```

### Running Notebooks

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

## Collaboration Resources

| Resource | Link |
|----------|------|
| Research Document | [Google Docs - TBD] |
| Task Tracker | [Notion - TBD] |
| Project Slides | [TBD] |
| Paper Draft | `paper/paper_draft.md` |

## Contribution Guide

### Branching Strategy

```
main                    # Stable, production-ready code
├── dev                 # Integration branch for features
│   ├── feature/hemanth     # Hemanth's feature branch
│   └── feature/gowtham     # Gowtham's feature branch
```

### Branch Naming Conventions
- `feature/<name>` - New features
- `fix/<name>` - Bug fixes
- `experiment/<name>` - Research experiments
- `docs/<name>` - Documentation updates

### Workflow

1. **Create Feature Branch**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

2. **Implement Feature**
   - Write code following PEP8 guidelines
   - Add docstrings to all functions
   - Include type hints where appropriate

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add symptom extraction module"
   ```

4. **Open Pull Request**
   - Push to remote: `git push origin feature/your-feature-name`
   - Create PR against `dev` branch
   - Fill in PR template with description

5. **Code Review**
   - At least one approval required
   - Address review comments

6. **Merge into Dev**
   - Squash commits if needed
   - Delete feature branch after merge

7. **Project Lead Merges to Main**
   - Only project lead merges `dev` → `main`
   - Requires passing tests

### Code Style Rules

- Python 3.10+ compliance
- PEP8 formatting
- All functions require docstrings
- Use type hints where applicable
- Maximum line length: 100 characters

## Team

| Role | Name |
|------|------|
| Project Lead | [TBD] |
| Junior Research Associate | [TBD] |
| Student Developer | Hemanth |
| Student Developer | Gowtham |

## License

This project is for research purposes. See LICENSE for details.

## Citation

If you use this work, please cite:

```bibtex
@misc{sanskrit-llm-2026,
  title = {Ayurveda Sanskrit LLM},
  author = {Hemanth, Gowtham, and Team},
  year = {2026},
  institution = {Research Institution}
}
```

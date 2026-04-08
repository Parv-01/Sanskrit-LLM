# Surya-OCR Sanskrit Pipeline

## Pipeline and Core Concept
The core concept of this pipeline is Deep Learning-based Document Analysis using the Surya-OCR engine. Unlike standard Tesseract-based approaches, Surya uses high-performance transformer models for both text detection and recognition. It is specifically designed to handle complex layouts and multilingual scripts (Sanskrit/Devanagari) with high density.

### Pipeline Stages
1. Dynamic API Selection: Automatically detects the installed version of Surya (Foundation, Predictor, or Legacy) to ensure compatibility across different environment setups.
2. Page-to-Image Conversion: Uses PyMuPDF (Fitz) to render PDF pages into images at a configurable DPI for processing.
3. Layout Detection & Recognition: Utilizes a dual-model approach where one model identifies text lines/blocks and the second (recognition) model decodes the Devanagari characters.
4. Adaptive Batching: Groups page images into batches for optimized CPU/GPU utilization, reducing total processing time for large documents.

## How to Run

### Dependencies:
  pip install surya-ocr pymupdf pillow tqdm

### Execution:
  python surya_ocr.py --input book.pdf --output result.txt
  python surya_ocr.py --input book.pdf --pages 10-25 --dpi 250
  
## Reference

1. Surya-OCR Project: A specialized document OCR toolkit built for high-accuracy multilingual recognition.
  https://github.com/vikasreacts/surya
2. PyMuPDF Documentation: High-performance library used for document rendering and metadata extraction.

## Remarks
Surya-OCR providing superior results for Devanagari compared to traditional engines because it understands the visual structure of the script better. However, it is resource-intensive; it is recommended to keep batch-size low (1 or 2) if running on a standard CPU to prevent memory overflow.

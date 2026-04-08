# Tesseract Indic OCR Pipeline

## Pipeline and Core Concept
The core concept of this pipeline is Tesseract-based Raster OCR. Unlike direct text extraction, this script treats every PDF page as an image (rasterization), applying image enhancement techniques before passing them to the Tesseract engine. It is specifically configured to use Indic language packs (Sanskrit and Hindi) to handle the complex ligatures and character structures of Devanagari script.

### Pipeline Stages
1. Rasterization: Converts PDF pages into high-resolution images (default 300 DPI) using the Poppler library to ensure character edges are sharp.
2. Image Enhancement: Utilizes OpenCV to convert images to grayscale and applies Otsu's Binarization. This removes background noise and creates a high-contrast black-and-white image for the OCR engine.
3. Multi-Language OCR: Passes the processed image to Tesseract using a combined language flag (san+hin+eng). This allows the engine to recognize Sanskrit characters while correctly identifying modern Hindi words or English annotations.
4. Text Aggregation: Collects the identified text page-by-page and writes it to a UTF-8 encoded text file to preserve Devanagari Unicode formatting.

## How to Run
### Dependencies:
  pip install pytesseract pdf2image opencv-python numpy Pillow
### System Requirements:
  Tesseract OCR: Install the Tesseract engine on your OS and ensure the san (Sanskrit) and hin (Hindi) traineddata files are in your tessdata folder.
  Poppler: Required for pdf2image.
  Windows: Add the Poppler bin/ folder to your System PATH.
  Linux: sudo apt-get install poppler-utils

### Execution:
  python devnagari_ocr.py input_file.pdf output_file.txt
  python devnagari_ocr.py input_file.pdf output_file.txt --first-page 1 --last-page 10

  
## Reference
1. Tesseract OCR Engine: Smith, R. (2007). An Overview of the Tesseract OCR Engine. Ninth International Conference on Document Analysis and Recognition (ICDAR).
2. Otsu's Method: Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. IEEE Transactions on Systems, Man, and Cybernetics.

## Remarks
While this approach is more stable than the CNN-based model for general documents, its accuracy is heavily dependent on the quality of the "san" traineddata. Performance may decrease on historical manuscripts with non-standard fonts or heavy physical degradation.

# Devanagari OCR CNN Pipeline

## Pipeline and Core Concept
The core concept of this pipeline is **Convolutional Neural Network (CNN)** based Optical Character Recognition. It utilizes deep learning to identify individual Devanagari characters from document images. Unlike standard OCR, this approach is specifically trained on the spatial features of Hindi and Sanskrit scripts.

### Pipeline Stages
1. Document Transformation: Converts PDF pages into high-resolution images for feature extraction.
2. Image Preprocessing: Uses OpenCV for noise reduction, binarization, and segmenting text lines into individual character blocks.
3. CNN Classification: Processes each character segment through a PyTorch-based Convolutional Neural Network to identify consonants, vowels, and numerals.
4. Unicode Mapping: Translates model predictions into standard Devanagari Unicode characters for text output.

## How to Run
1. Dependencies:
    pip install torch torchvision opencv-python pillow pdf2image tqdm
2. Execution:
    python CNN_based.py input_file.pdf -o CNN_output.txt

## Reference

1. A Systematic Framework for Sanskrit Character Recognition Using Deep Learning
    https://elcvia.cvc.uab.cat/article/view/1850

## Remarks

The performance of the CNN-based pipeline is currently suboptimal, as noisy source text is triggering a high volume of character misclassifications and "garbage" outputs. To improve accuracy, the model requires more robust preprocessing or a transition to a more noise-tolerant architecture that can better distinguish between script features and document artifacts.
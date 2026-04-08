"""
Sanskrit/Devanagari Raster OCR Pipeline
This script treats all PDF pages as images (ignoring embedded text layers)
and processes them using Tesseract OCR with Indic language support.
"""

import os
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image

def process_tesseract_only(pdf_path, output_path, lang='san+hin+eng', dpi=300):
    print(f"Starting Tesseract-only extraction for: {pdf_path}")
    
    try:
        # 1. Convert PDF pages to PIL Images
        pages = convert_from_path(pdf_path, dpi=dpi)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(pages):
                print(f"Processing Page {i+1}/{len(pages)}...")
                
                # 2. Image Preprocessing for better OCR accuracy
                # Convert PIL to OpenCV format
                open_cv_image = np.array(page)
                gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
                
                # Apply thresholding to clean up noise/background
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 3. Tesseract Extraction
                # Config: --psm 1 (Automatic page segmentation with OSD)
                custom_config = r'--oem 3 --psm 1'
                text = pytesseract.image_to_string(binary, lang=lang, config=custom_config)
                
                # 4. Write to file
                f.write(f"--- Page {i + 1} ---\n")
                f.write(text)
                f.write("\n\n")
                
        print(f"Extraction complete. Results saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure you have 'san', 'hin', and 'eng' traineddata in your Tesseract-OCR/tessdata folder
    input_pdf = "charaka_samhita.pdf" 
    output_txt = "extracted_sanskrit.txt"
    
    process_tesseract_only(input_pdf, output_txt)
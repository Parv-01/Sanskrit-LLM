"""
Sanskrit PDF → Text Pipeline  (fixed)
======================================
Stage 1 : Image pre-processing       (OpenCV — Paper 6)
Stage 2 : Segmentation               (OpenCV projection profiles — Paper 6)
Stage 3 : OCR                        (Tesseract -l san — Paper 7)
Stage 4 : OCR post-correction        (ByT5-Sanskrit — Nehrdich et al. EMNLP 2024)
           HuggingFace: buddhist-nlp/byt5-sanskrit-analyzer-hackathon
           Note: this model works on IAST (roman transliteration), not raw Devanagari,
           so we transliterate Devanagari → IAST before correction, then back.

Install:
    pip install pdf2image opencv-python-headless pytesseract transformers torch indic-transliteration tqdm Pillow

System (Linux):
    sudo apt-get install tesseract-ocr tesseract-ocr-san poppler-utils

System (Windows):
    1. Download Tesseract installer from https://github.com/UB-Mannheim/tesseract/wiki
       During install, check "Additional language data" and pick Sanskrit (san)
       Add Tesseract to PATH (e.g. C:\\Program Files\\Tesseract-OCR)
    2. Download poppler for Windows from https://github.com/oschwartz10612/poppler-windows/releases
       Extract and add the bin/ folder to PATH

Usage:
    python sanskrit_ocr_pipeline.py Sushrut_Sanhita.pdf -o output.txt
    python sanskrit_ocr_pipeline.py Sushrut_Sanhita.pdf -o output.txt --dpi 400
    python sanskrit_ocr_pipeline.py Sushrut_Sanhita.pdf -o output.txt --skip-correction
    python sanskrit_ocr_pipeline.py Sushrut_Sanhita.pdf -o output.txt --start-page 5 --end-page 50
    python sanskrit_ocr_pipeline.py Sushrut_Sanhita.pdf -o output.txt --devanagari-out
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Image pre-processing  (Paper 6, Section 3.1)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_image(pil_image: Image.Image, dpi_hint: int = 300) -> np.ndarray:
    """
    Pre-processing chain from Paper 6:
      1. Grayscale conversion
      2. Adaptive binarization  (Gaussian blur + Otsu threshold)
      3. Noise removal          (morphological opening)
      4. Deskewing              (Hough-line angle estimation)
      5. Shirorekha detection   (informational; removal is opt-in)
    """
    img = np.array(pil_image)

    # Step 1 — Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()

    # Step 2 — Binarization
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3 — Noise removal (kernel size scales with DPI)
    k = max(1, int(dpi_hint / 150))
    kernel = np.ones((k, k), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 4 — Deskew
    cleaned = _deskew(cleaned)

    # Step 5 — Shirorekha removal (uncomment if needed)
    # cleaned = _remove_shirorekha(cleaned)

    return cleaned


def _deskew(binary: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=binary.shape[1] // 4,
        maxLineGap=10,
    )
    if lines is None:
        return binary

    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in (l[0] for l in lines)
        if x2 != x1
    ]
    if not angles:
        return binary

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return binary

    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _remove_shirorekha(binary: np.ndarray) -> np.ndarray:
    """Remove the Devanagari header line (shirorekha) — Paper 6 Section 3.2."""
    row_sums = binary.sum(axis=1)
    top_zone = row_sums[:int(binary.shape[0] * 0.35)]
    if top_zone.max() == 0:
        return binary
    shirorekha_rows = np.where(top_zone > top_zone.max() * 0.6)[0]
    result = binary.copy()
    result[shirorekha_rows, :] = 0
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Segmentation  (Paper 6, Section 3.2)
# ═══════════════════════════════════════════════════════════════════════════════

def segment_lines(binary: np.ndarray, min_line_height: int = 10) -> list:
    """Horizontal projection profile → list of line images."""
    row_profile = binary.sum(axis=1)
    lines, in_line, line_start = [], False, 0

    for i, val in enumerate(row_profile):
        if val > 0 and not in_line:
            in_line, line_start = True, i
        elif val == 0 and in_line:
            in_line = False
            chunk = binary[line_start:i, :]
            if chunk.shape[0] >= min_line_height:
                lines.append(chunk)

    if in_line:
        chunk = binary[line_start:, :]
        if chunk.shape[0] >= min_line_height:
            lines.append(chunk)

    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — OCR  (Paper 7 — Tesseract with Sanskrit language pack)
# ═══════════════════════════════════════════════════════════════════════════════

def ocr_page(binary: np.ndarray, lang: str = "san") -> str:
    """
    Runs Tesseract on a pre-processed binary page.
    Paper 7 uses '--oem 1 -l san' (LSTM engine + Sanskrit language data).
    Returns raw Unicode Devanagari text.
    """
    # Tesseract expects black-ink on white background
    pil_img = Image.fromarray(cv2.bitwise_not(binary))
    config = f"--oem 1 --psm 3 -l {lang}"
    return pytesseract.image_to_string(pil_img, config=config).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — OCR Post-Correction  (Nehrdich et al. EMNLP 2024)
# ═══════════════════════════════════════════════════════════════════════════════

class SanskritOCRCorrector:
    """
    Wraps the ByT5-Sanskrit model for OCR post-correction.

    Paper  : Nehrdich, Hellwig, Keutzer — EMNLP Findings 2024
             "One Model is All You Need: ByT5-Sanskrit"
             arXiv: 2409.13920
    GitHub : https://github.com/sebastian-nehrdich/byt5-sanskrit-analyzers
    HF org : https://huggingface.co/buddhist-nlp

    ── Why IAST, not Devanagari? ──
    The ByT5-Sanskrit model was pretrained and fine-tuned on IAST roman
    transliteration, not raw Devanagari bytes.  (Nehrdich et al. 2024, Section 3:
    "We use IAST transliteration for pretraining as well as all of the fine-tuning
    tasks, as this yields clear efficiency advantages compared to Devanagari when
    training on the individual byte level, with half the bytes needed.")
    We therefore:
        Tesseract Devanagari  →  IAST  →  ByT5 correction  →  IAST output
    and optionally back-convert to Devanagari at the end.
    """

    # Confirmed HuggingFace model IDs (Buddhist NLP group, March 2026):
    #   buddhist-nlp/byt5-sanskrit                      — pretrained base (no OCR fine-tune)
    #   buddhist-nlp/byt5-sanskrit-analyzer-hackathon   — fine-tuned multitask  ← USE THIS
    DEFAULT_MODEL = "buddhist-nlp/byt5-sanskrit-analyzer-hackathon"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        self._transliterate = transliterate
        self._sanscript = sanscript
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name or self.DEFAULT_MODEL

        log.info("Loading ByT5-Sanskrit from '%s' on %s ...", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        log.info("Model loaded.")

    def _dev_to_iast(self, text: str) -> str:
        return self._transliterate(
            text, self._sanscript.DEVANAGARI, self._sanscript.IAST
        )

    def _iast_to_dev(self, text: str) -> str:
        return self._transliterate(
            text, self._sanscript.IAST, self._sanscript.DEVANAGARI
        )

    def correct(
        self,
        devanagari_text: str,
        max_input_len: int = 512,
        max_output_len: int = 512,
        return_devanagari: bool = False,
    ) -> str:
        """
        Corrects raw Tesseract Devanagari output line by line.

        Args:
            devanagari_text   : raw OCR output from Stage 3
            max_input_len     : max tokens per line fed to the model
            max_output_len    : max new tokens model may generate
            return_devanagari : if True, convert IAST output back to Devanagari

        Returns:
            Corrected text — IAST by default, Devanagari if return_devanagari=True
        """
        import torch

        if not devanagari_text.strip():
            return devanagari_text

        corrected_lines = []
        for line in devanagari_text.splitlines():
            stripped = line.strip()
            if not stripped:
                corrected_lines.append(line)
                continue

            iast_line = self._dev_to_iast(stripped)

            inputs = self.tokenizer(
                iast_line,
                return_tensors="pt",
                max_length=max_input_len,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_output_len,
                    num_beams=4,
                    early_stopping=True,
                )

            corrected_iast = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            corrected_lines.append(
                self._iast_to_dev(corrected_iast) if return_devanagari else corrected_iast
            )

        return "\n".join(corrected_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_path: str,
    output_path: str,
    dpi: int = 300,
    lang: str = "san",
    skip_correction: bool = False,
    corrector_model: Optional[str] = None,
    return_devanagari: bool = False,
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> None:
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    if not pdf_path.exists():
        log.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Stage 4: load corrector ──
    corrector = None
    if not skip_correction:
        try:
            corrector = SanskritOCRCorrector(model_name=corrector_model)
        except Exception as e:
            log.warning("Could not load ByT5-Sanskrit corrector: %s", e)
            log.warning("Stage 4 disabled — output will be raw Tesseract Devanagari text.")

    # ── Render PDF ──
    log.info("Rendering '%s' at %d dpi ...", pdf_path.name, dpi)
    pages = convert_from_path(
        str(pdf_path), dpi=dpi,
        first_page=start_page, last_page=end_page,
    )
    log.info("Loaded %d pages.", len(pages))

    parts = []
    for page_num, pil_page in enumerate(
        tqdm(pages, desc="Processing pages"), start=start_page
    ):
        log.info("─── Page %d ───", page_num)

        binary = preprocess_image(pil_page, dpi_hint=dpi)          # Stage 1
        lines = segment_lines(binary)                               # Stage 2
        log.debug("  %d text lines detected", len(lines))

        raw_text = ocr_page(binary, lang=lang)                      # Stage 3
        if not raw_text:
            log.warning("  Page %d: Tesseract returned no text", page_num)
            continue

        if corrector is not None:                                    # Stage 4
            final_text = corrector.correct(
                raw_text, return_devanagari=return_devanagari
            )
        else:
            final_text = raw_text

        parts.append(f"\n\n# Page {page_num}\n\n{final_text}")

    full_text = "\n".join(parts)
    output_path.write_text(full_text, encoding="utf-8")
    log.info("Done — written to '%s'  (%d chars)", output_path, len(full_text))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sanskrit PDF → text via paper-backed OCR pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf", help="Path to input Sanskrit PDF")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .txt path (default: <pdf_stem>_output.txt)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Render DPI. Use 400 for old prints like Sushruta Samhita.")
    parser.add_argument("--lang", default="san",
                        help="Tesseract language code. 'san' = Sanskrit.")
    parser.add_argument("--skip-correction", action="store_true",
                        help="Skip Stage 4 (fast test mode).")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model for Stage 4. "
                             "Default: buddhist-nlp/byt5-sanskrit-analyzer-hackathon")
    parser.add_argument("--devanagari-out", action="store_true",
                        help="Convert Stage 4 IAST output back to Devanagari. "
                             "Default output is IAST roman.")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    output = args.output or (Path(args.pdf).stem + "_output.txt")

    process_pdf(
        pdf_path=args.pdf,
        output_path=output,
        dpi=args.dpi,
        lang=args.lang,
        skip_correction=args.skip_correction,
        corrector_model=args.model,
        return_devanagari=args.devanagari_out,
        start_page=10,
        end_page=30,
    )


if __name__ == "__main__":
    main()
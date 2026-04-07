"""
Devanagari / Sanskrit + English PDF -> Text Pipeline (v3 - Final)

How it works:
  - Splits PDF into chunks of 30 pages
  - For each chunk, samples the FIRST page to check if OCR-enabled
    (usable text = >50% of chars are Devanagari / English / punctuation)
  - If OCR-enabled  -> reads embedded text directly (fast, accurate)
  - If scanned      -> rasterizes pages and runs Tesseract OCR
  - Each chunk is decided independently (handles mixed PDFs)
  - Preserves English text as-is alongside Devanagari
  - Strips noise characters: { } ~ ` | [ ] ^ < > backslash
  - Drops lines that have no Devanagari and no English words

Usage:
  python devanagari_ocr.py input.pdf output.txt
  python devanagari_ocr.py input.pdf output.txt --first-page 5 --last-page 80
  python devanagari_ocr.py input.pdf output.txt --dpi 400
  python devanagari_ocr.py input.pdf output.txt --debug

Dependencies:
  pip install pdf2image pytesseract opencv-python Pillow numpy pypdf
  Tesseract with san + hin + eng language packs installed
"""

import os
import re
import argparse
from pathlib import Path

import pypdf
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

# ── Windows: Tesseract executable path ───────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── CONFIGURATION (edit here or override via CLI flags) ───────────────────────
TESS_LANG      = "san+hin+eng"  # Tesseract language models to use
TESS_PSM       = 6              # Page segmentation: 6=single block, 3=auto
TESS_OEM       = 1              # Engine: 1=LSTM neural net (best)
RENDER_DPI     = 300            # DPI for rasterising scanned pages
CHUNK_SIZE     = 30             # Pages per chunk
OCR_THRESHOLD  = 0.50           # Fraction of "good" chars to consider page OCR-enabled

# Characters stripped from ALL output regardless of source
NOISE_CHARS = set('{}~`|[]^<>\\')

# ─────────────────────────────────────────────────────────────────────────────


# ── CHARACTER CLASSIFIERS ─────────────────────────────────────────────────────

def is_devanagari(c: str) -> bool:
    return '\u0900' <= c <= '\u097F'

def is_english_word_char(c: str) -> bool:
    return c.isascii() and (c.isalpha() or c.isdigit())


# ── TEXT QUALITY CHECK ────────────────────────────────────────────────────────

def meaningful_ratio(text: str) -> float:
    """
    Returns the fraction of non-whitespace characters that are
    Devanagari, ASCII alphanumeric, or standard punctuation.
    Used to decide whether a page's embedded text is usable.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    good = sum(
        1 for c in chars
        if is_devanagari(c)
        or (c.isascii() and (c.isalpha() or c.isdigit()))
        or c in '.,;:!?()"\'-\u0964\u0965'   # danda, double-danda, common punctuation
    )
    return good / len(chars)


# ── NOISE CLEANING ────────────────────────────────────────────────────────────

def strip_noise(text: str) -> str:
    """
    1. Remove every character listed in NOISE_CHARS globally.
    2. Drop lines that contain neither Devanagari nor an English word (>= 2 letters).
    3. Strip leading/trailing junk from surviving lines.
    4. Collapse 3+ consecutive blank lines into 2.
    """
    # Step 1: strip noise characters
    cleaned = ''.join(c for c in text if c not in NOISE_CHARS)

    # Step 2-3: filter lines
    kept = []
    for line in cleaned.split('\n'):
        s = line.strip()
        if not s:
            kept.append('')
            continue

        has_deva    = any(is_devanagari(c) for c in s)
        has_english = bool(re.search(r'[A-Za-z]{2,}', s))

        if has_deva or has_english:
            # Strip residual leading/trailing non-word chars
            s = re.sub(r'^[^\u0900-\u097FA-Za-z0-9]+', '', s)
            s = re.sub(r'[^\u0900-\u097FA-Za-z0-9\u0964\u0965.!?,;:()\'"]+$', '', s)
            if s:
                kept.append(s)
        # else: pure symbol/number line — silently dropped

    # Step 4: collapse excess blank lines
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(kept)).strip()


# ── CHUNK-LEVEL OCR DETECTION ─────────────────────────────────────────────────

def is_page_ocr_enabled(reader: pypdf.PdfReader, page_index: int, chunk_end: int) -> tuple:
    """
    Sample pages in a chunk starting from page_index until a non-empty page is found.
    Returns (ocr_enabled: bool, sampled_page_index: int).
    Skips pages with 0 chars and tries the next one.
    """
    idx = page_index
    while idx < chunk_end:
        try:
            text = reader.pages[idx].extract_text() or ""
        except Exception:
            text = ""

        ratio = meaningful_ratio(text)

        if len(text.strip()) == 0:
            print(f"  [SAMPLE] p.{idx+1} is empty, trying next page...")
            idx += 1
            continue

        # Non-empty page found — use it to decide strategy
        return (ratio > OCR_THRESHOLD), idx, ratio

    # All pages in chunk were empty — default to image OCR
    print(f"  [SAMPLE] All pages in chunk are empty, defaulting to image-OCR")
    return False, page_index, 0.0


# ── STRATEGY A: Embedded text ─────────────────────────────────────────────────

def extract_embedded(reader: pypdf.PdfReader, page_index: int) -> str:
    """Read and clean one page from the PDF text layer."""
    try:
        raw = reader.pages[page_index].extract_text() or ""
    except Exception:
        raw = ""
    return strip_noise(raw)


# ── STRATEGY B: Image OCR ─────────────────────────────────────────────────────

def preprocess(pil_img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a clean grayscale numpy array for Tesseract.
    Applies contrast enhancement + Otsu threshold only when the image
    is NOT already binary (avoids destroying clean digital scans).
    """
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)

    hist     = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    bw_ratio = (hist[0] + hist[255]) / hist.sum()

    if bw_ratio <= 0.70:                              # not already B&W
        gray = cv2.GaussianBlur(gray, (3, 3), 0)     # mild denoise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)                     # local contrast boost
        _, gray = cv2.threshold(                      # Otsu binarisation
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    return gray


def ocr_image(pil_img: Image.Image, debug_path: str = None) -> str:
    """Preprocess + Tesseract OCR one PIL image. Returns cleaned text."""
    img = preprocess(pil_img)
    if debug_path:
        cv2.imwrite(debug_path, img)
    config = f"--oem {TESS_OEM} --psm {TESS_PSM} -c preserve_interword_spaces=1"
    raw = pytesseract.image_to_string(Image.fromarray(img), lang=TESS_LANG, config=config)
    return strip_noise(raw)


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run(
    pdf_path:    str,
    output_path: str,
    first_page:  int  = None,
    last_page:   int  = None,
    debug:       bool = False,
) -> None:

    pdf_path    = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if debug:
        debug_dir = output_path.parent / "debug_pages"
        debug_dir.mkdir(exist_ok=True)
        print(f"[INFO] Debug images -> {debug_dir}")

    reader      = pypdf.PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    # Convert to 0-indexed range
    p_start = (first_page - 1) if first_page else 0
    p_end   = min(last_page if last_page else total_pages, total_pages)

    print(f"[INFO] '{pdf_path.name}' — {total_pages} total pages")
    print(f"[INFO] Processing pages {p_start + 1} to {p_end}\n")

    with open(output_path, "w", encoding="utf-8") as out:

        page_idx = p_start

        while page_idx < p_end:
            chunk_start = page_idx
            chunk_end   = min(page_idx + CHUNK_SIZE, p_end)   # exclusive
            
            # ── Decide strategy by sampling first non-empty page of chunk ──
            ocr_enabled, sample_page, sample_ratio = is_page_ocr_enabled(
                reader, chunk_start, chunk_end
            )
            strategy = "embedded" if ocr_enabled else "image-OCR"

            print(f"[CHUNK {chunk_start+1}-{chunk_end}]  "
                  f"sample p.{sample_page+1} -> ratio={sample_ratio:.0%} -> {strategy}")

            # ── Rasterise whole chunk upfront if using image OCR ───────────
            chunk_images = None
            if not ocr_enabled:
                chunk_images = convert_from_path(
                    str(pdf_path),
                    dpi         = RENDER_DPI,
                    fmt         = "jpeg",
                    first_page  = chunk_start + 1,   # pdf2image is 1-indexed
                    last_page   = chunk_end,
                    thread_count= 4,
                )

            # ── Process each page ──────────────────────────────────────────
            for i, pg_idx in enumerate(range(chunk_start, chunk_end)):
                page_num = pg_idx + 1   # human-readable 1-indexed

                if ocr_enabled:
                    text = extract_embedded(reader, pg_idx)
                else:
                    dbg  = str(debug_dir / f"page_{page_num:04d}.png") if debug_dir else None
                    text = ocr_image(chunk_images[i], debug_path=dbg)

                print(f"  p.{page_num}: {len(text.strip()):>5} chars  [{strategy}]")

                out.write(f"{'='*60}\nPage {page_num}\n{'='*60}\n{text}\n\n")
                out.flush()   # write immediately so file is readable during long runs

            page_idx = chunk_end

    print(f"\n[DONE] Saved -> {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global TESS_LANG, RENDER_DPI

    parser = argparse.ArgumentParser(
        description="Devanagari + English PDF -> Text (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python devanagari_ocr.py book.pdf book.txt
  python devanagari_ocr.py book.pdf book.txt --first-page 10 --last-page 50
  python devanagari_ocr.py book.pdf book.txt --dpi 400
  python devanagari_ocr.py book.pdf book.txt --debug
        """,
    )
    parser.add_argument("input_pdf",   help="Input PDF path")
    parser.add_argument("output_txt",  help="Output .txt path")
    parser.add_argument("--first-page", type=int, default=None,
                        help="First page to process (1-indexed, inclusive)")
    parser.add_argument("--last-page",  type=int, default=None,
                        help="Last page to process (1-indexed, inclusive)")
    parser.add_argument("--dpi",  type=int, default=RENDER_DPI,
                        help=f"Render DPI for scanned pages (default {RENDER_DPI})")
    parser.add_argument("--lang", default=TESS_LANG,
                        help=f"Tesseract language string (default '{TESS_LANG}')")
    parser.add_argument("--debug", action="store_true",
                        help="Save preprocessed images to debug_pages/ folder")

    args       = parser.parse_args()
    TESS_LANG  = args.lang
    RENDER_DPI = args.dpi

    run(
        pdf_path    = args.input_pdf,
        output_path = args.output_txt,
        first_page  = 7,
        last_page   = args.last_page,
        debug       = args.debug,
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Sanskrit Book OCR Pipeline using Surya-OCR  (v4 — supports all API versions)
==============================================================================
Handles all three surya API generations:
  v0.13+  : FoundationPredictor required, langs auto-detected (newest)
  v0.6-0.12: RecognitionPredictor()/DetectionPredictor() no foundation
  v0.4-0.5 : run_ocr + load_model functions (oldest)

Requirements:
    pip install surya-ocr pymupdf pillow tqdm

Usage:
    python sanskrit_ocr_pipeline.py --input book.pdf --output output.txt
    python sanskrit_ocr_pipeline.py --input book.pdf --output output.txt --pages 1-50
    python sanskrit_ocr_pipeline.py --input book.pdf --output output.txt --dpi 200 --batch-size 2
"""

import argparse
import importlib
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Dependency check
# ─────────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg, mod in [("surya-ocr","surya"),("pymupdf","fitz"),("Pillow","PIL"),("tqdm","tqdm")]:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing dependencies. Install with:")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)


def detect_surya_api():
    """
    Returns:
      'foundation'  surya >= 0.13  (FoundationPredictor required, no langs arg)
      'predictor'   surya 0.6-0.12 (RecognitionPredictor() / DetectionPredictor())
      'legacy'      surya 0.4-0.5  (run_ocr + load_model)
    """
    try:
        importlib.import_module("surya.foundation")
        importlib.import_module("surya.recognition")
        importlib.import_module("surya.detection")
        return "foundation"
    except ImportError:
        pass
    try:
        importlib.import_module("surya.recognition")
        importlib.import_module("surya.detection")
        return "predictor"
    except ImportError:
        pass
    try:
        importlib.import_module("surya.ocr")
        return "legacy"
    except ImportError:
        pass
    print("Could not detect a working surya API. Please reinstall: pip install surya-ocr")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# PDF  →  PIL Images
# ─────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path, page_range, dpi, keep_images, image_dir):
    import fitz
    from PIL import Image
    import io

    doc   = fitz.open(str(pdf_path))
    total = len(doc)
    s, e  = max(1, page_range[0]), min(total, page_range[1])
    print(f"PDF has {total} pages. Processing pages {s}-{e} at {dpi} DPI.")

    if keep_images:
        image_dir.mkdir(parents=True, exist_ok=True)
        print(f"Page images will be saved to: {image_dir}")

    mat, pages = fitz.Matrix(dpi / 72, dpi / 72), []
    for i in range(s - 1, e):
        pix = doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        if keep_images:
            img.save(image_dir / f"page_{i+1:04d}.png")
        pages.append((i + 1, img))

    doc.close()
    return pages


# ─────────────────────────────────────────────────────────────
# OCR  —  Foundation API  (surya >= 0.13)
# ─────────────────────────────────────────────────────────────

def run_ocr_foundation(pages, batch_size, verbose):
    from tqdm import tqdm

    foundation_mod = importlib.import_module("surya.foundation")
    rec_mod        = importlib.import_module("surya.recognition")
    det_mod        = importlib.import_module("surya.detection")

    FoundationPredictor  = foundation_mod.FoundationPredictor
    RecognitionPredictor = rec_mod.RecognitionPredictor
    DetectionPredictor   = det_mod.DetectionPredictor

    print("\nLoading Surya models (foundation API >= 0.13) ...")
    t0  = time.time()
    fp  = FoundationPredictor()
    rec = RecognitionPredictor(fp)
    det = DetectionPredictor()
    print(f"Models loaded in {time.time()-t0:.1f}s")

    results = []
    with tqdm(total=len(pages), desc="OCR", unit="page", disable=verbose) as pbar:
        for i in range(0, len(pages), batch_size):
            batch = pages[i : i + batch_size]
            nums  = [p[0] for p in batch]
            imgs  = [p[1] for p in batch]

            # No langs argument in 0.13+
            for page_num, res in zip(nums, rec(imgs, det_predictor=det)):
                text = "\n".join(line.text for line in res.text_lines)
                if verbose:
                    print(f"  Page {page_num:4d}: {text[:80].replace(chr(10),' ')} ...")
                results.append((page_num, text))
            pbar.update(len(batch))

    return results


# ─────────────────────────────────────────────────────────────
# OCR  —  Predictor API  (surya 0.6 – 0.12)
# ─────────────────────────────────────────────────────────────

def run_ocr_predictor(pages, lang, batch_size, verbose):
    from tqdm import tqdm

    rec_mod = importlib.import_module("surya.recognition")
    det_mod = importlib.import_module("surya.detection")

    RecognitionPredictor = rec_mod.RecognitionPredictor
    DetectionPredictor   = det_mod.DetectionPredictor

    print(f"\nLoading Surya models (predictor API 0.6-0.12, lang={lang}) ...")
    t0  = time.time()
    rec = RecognitionPredictor()
    det = DetectionPredictor()
    print(f"Models loaded in {time.time()-t0:.1f}s")

    results = []
    with tqdm(total=len(pages), desc="OCR", unit="page", disable=verbose) as pbar:
        for i in range(0, len(pages), batch_size):
            batch      = pages[i : i + batch_size]
            nums       = [p[0] for p in batch]
            imgs       = [p[1] for p in batch]
            langs_list = [[lang]] * len(imgs)

            for page_num, res in zip(nums, rec(imgs, langs_list, det)):
                text = "\n".join(line.text for line in res.text_lines)
                if verbose:
                    print(f"  Page {page_num:4d}: {text[:80].replace(chr(10),' ')} ...")
                results.append((page_num, text))
            pbar.update(len(batch))

    return results


# ─────────────────────────────────────────────────────────────
# OCR  —  Legacy API  (surya 0.4 / 0.5)
# ─────────────────────────────────────────────────────────────

def run_ocr_legacy(pages, lang, batch_size, verbose):
    from tqdm import tqdm

    run_ocr = importlib.import_module("surya.ocr").run_ocr

    try:
        det_mod = importlib.import_module("surya.model.detection.model")
    except ImportError:
        det_mod = importlib.import_module("surya.model.detection.segformer")

    rec_model_mod = importlib.import_module("surya.model.recognition.model")
    rec_proc_mod  = importlib.import_module("surya.model.recognition.processor")

    print(f"\nLoading Surya models (legacy API, lang={lang}) ...")
    t0       = time.time()
    det_proc = det_mod.load_processor()
    det_m    = det_mod.load_model()
    rec_m    = rec_model_mod.load_model()
    rec_proc = rec_proc_mod.load_processor()
    print(f"Models loaded in {time.time()-t0:.1f}s")

    results = []
    with tqdm(total=len(pages), desc="OCR", unit="page", disable=verbose) as pbar:
        for i in range(0, len(pages), batch_size):
            batch      = pages[i : i + batch_size]
            nums       = [p[0] for p in batch]
            imgs       = [p[1] for p in batch]
            langs_list = [[lang]] * len(imgs)

            for page_num, res in zip(nums, run_ocr(imgs, langs_list, det_m, det_proc, rec_m, rec_proc)):
                text = "\n".join(line.text for line in res.text_lines)
                if verbose:
                    print(f"  Page {page_num:4d}: {text[:80].replace(chr(10),' ')} ...")
                results.append((page_num, text))
            pbar.update(len(batch))

    return results


# ─────────────────────────────────────────────────────────────
# Save output
# ─────────────────────────────────────────────────────────────

def save_text(results, output_path, pdf_path, lang, dpi):
    results.sort(key=lambda x: x[0])
    header = (
        f"# Sanskrit OCR Output\n"
        f"# Source  : {pdf_path.name}\n"
        f"# Language: {lang}\n"
        f"# DPI     : {dpi}\n"
        f"# Pages   : {results[0][0]}-{results[-1][0]}  ({len(results)} pages)\n"
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for num, text in results:
            f.write(f"\n{'─'*40}\n  PAGE {num}\n{'─'*40}\n\n{text}\n")

    print(f"\nSaved {len(results)} pages -> {output_path}  ({output_path.stat().st_size/1024:.1f} KB, UTF-8)")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_page_range(value):
    try:
        if "-" in value:
            a, b = value.split("-", 1)
            return int(a), int(b)
        n = int(value)
        return n, n
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid page range '{value}'. Use e.g. '1-50'.")


def main():
    check_dependencies()
    api = detect_surya_api()
    print(f"Detected surya API: {api}")

    p = argparse.ArgumentParser(description="Sanskrit PDF OCR pipeline using Surya-OCR")
    p.add_argument("--input",       required=True,  type=Path)
    p.add_argument("--output",      required=True,  type=Path)
    p.add_argument("--pages",       default=None,   type=str,  help="e.g. 1-100")
    p.add_argument("--dpi",         default=200,    type=int)
    p.add_argument("--batch-size",  default=2,      type=int,  help="Keep at 1-2 for CPU")
    p.add_argument("--lang",        default="sa",   type=str,  help="sa = Sanskrit/Devanagari (ignored on surya >= 0.13 which auto-detects)")
    p.add_argument("--keep-images", action="store_true")
    p.add_argument("--image-dir",   default=Path("./page_images"), type=Path)
    p.add_argument("--verbose",     action="store_true")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    import fitz
    with fitz.open(str(args.input)) as doc:
        total = len(doc)
    page_range = parse_page_range(args.pages) if args.pages else (1, total)

    t0    = time.time()
    pages = pdf_to_images(args.input, page_range, args.dpi, args.keep_images, args.image_dir)

    if api == "foundation":
        results = run_ocr_foundation(pages, args.batch_size, args.verbose)
    elif api == "predictor":
        results = run_ocr_predictor(pages, args.lang, args.batch_size, args.verbose)
    else:
        results = run_ocr_legacy(pages, args.lang, args.batch_size, args.verbose)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_text(results, args.output, args.input, args.lang, args.dpi)

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s  ({elapsed/len(pages):.1f}s/page)")


if __name__ == "__main__":
    main()
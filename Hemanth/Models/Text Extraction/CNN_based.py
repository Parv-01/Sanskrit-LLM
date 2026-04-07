"""
"A Systematic Framework for Sanskrit Character Recognition Using Deep Learning"
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
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Devanagari Unicode character map
# 46 classes used in Paper 6 (36 consonants + 10 numerals)
# Extended here with vowels for better coverage
# ─────────────────────────────────────────────────────────────────────────────

DEVANAGARI_CLASSES = [
    # Vowels (16)
    "अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ", "अं", "अः",
    "ऋ", "ॠ", "ऌ", "ॡ",
    # Consonants (36)
    "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व",
    "श", "ष", "स", "ह",
    "ळ", "क्ष", "ज्ञ",
    # Digits (10)
    "०", "१", "२", "३", "४", "५", "६", "७", "८", "९",
]

NUM_CLASSES = len(DEVANAGARI_CLASSES)   # 62
IMG_SIZE    = 32                        # Paper 6 uses 32×32 input patches


# ─────────────────────────────────────────────────────────────────────────────
# CNN Architecture — Paper 6, Section 4.2
#
# "The proposed CNN consists of three convolutional layers each followed by
#  max-pooling, a flatten layer, and two fully connected layers with dropout."
#
#  Input  : 1 × 32 × 32  (grayscale)
#  Conv1  : 32 filters, 3×3, ReLU → MaxPool 2×2  → 32 × 15 × 15
#  Conv2  : 64 filters, 3×3, ReLU → MaxPool 2×2  → 64 × 6 × 6
#  Conv3  : 128 filters, 3×3, ReLU → MaxPool 2×2 → 128 × 2 × 2
#  Flatten: 512
#  FC1    : 512 → 256, ReLU, Dropout 0.5
#  FC2    : 256 → NUM_CLASSES
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn(num_classes: int = NUM_CLASSES):
    """Build the Paper 6 CNN. Returns a torch.nn.Module."""
    import torch.nn as nn

    class DevanagariCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Conv block 1
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),          # → 32 × 16 × 16

                # Conv block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),          # → 64 × 8 × 8

                # Conv block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),          # → 128 × 4 × 4
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return DevanagariCNN()


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights_path: Optional[str] = None):
    """
    Load the CNN with weights.

    Priority:
      1. weights_path argument (explicit path)
      2. devanagari_cnn.pth   in the same directory as this script
      3. Download from HuggingFace (Sugam-Arora/devanagari-character-recognition)
         This is a ~8 MB pretrained model on the UCI Devanagari dataset.

    Returns (model, device) or (None, None) if PyTorch unavailable.
    """
    try:
        import torch
    except ImportError:
        log.warning("PyTorch not installed — CNN unavailable. Use --tesseract-fallback.")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cnn()

    # --- Try explicit path first ---
    candidates = []
    if weights_path:
        candidates.append(Path(weights_path))
    candidates.append(Path(__file__).parent / "devanagari_cnn.pth")

    for path in candidates:
        if path.exists():
            log.info("Loading CNN weights from '%s' ...", path)
            state = torch.load(str(path), map_location=device)
            # Handle both plain state_dict and checkpoint dicts
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval()
            model.to(device)
            log.info("CNN ready on %s.", device)
            return model, device

    # --- Download pretrained weights ---
    log.info("No local weights found — downloading pretrained Devanagari CNN ...")
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id="Sugam-Arora/devanagari-character-recognition",
            filename="pytorch_model.bin",
        )
        state = torch.load(local_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        model.to(device)
        log.info("Pretrained CNN loaded on %s.", device)
        return model, device
    except Exception as e:
        log.warning("Could not download pretrained CNN: %s", e)
        log.warning("Falling back to Tesseract OCR for character recognition.")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Grayscale + Binarization + Noise removal  (Paper 6, Section 3.1)
# ─────────────────────────────────────────────────────────────────────────────

def binarize(pil_image: Image.Image, dpi: int = 300) -> np.ndarray:
    """
    Returns binary image: 255 = ink, 0 = background.
    """
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

    # Gaussian blur suppresses scanner noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's method: automatically finds optimal threshold
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening removes isolated noise pixels
    # Kernel size scales with DPI (larger kernel for higher DPI scans)
    k = max(1, dpi // 150)
    kernel = np.ones((k, k), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Deskewing  (Paper 6, Section 3.1)
# ─────────────────────────────────────────────────────────────────────────────

def deskew(binary: np.ndarray) -> np.ndarray:
    """
    Estimates page skew angle using Hough line transform and rotates to correct.
    Paper 6 mentions deskewing as a pre-processing step before segmentation.
    """
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=80,
        minLineLength=binary.shape[1] // 5,
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

    angle = float(np.median(angles))
    if abs(angle) < 0.3:        # negligible skew, skip rotation
        return binary

    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    log.debug("  Deskewed by %.2f°", angle)
    return rotated


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Shirorekha removal  (Paper 6, Section 3.2)
# ─────────────────────────────────────────────────────────────────────────────

def remove_shirorekha(binary: np.ndarray) -> np.ndarray:
    """
    Paper 6, Section 3.2:
    "The horizontal line (shirorekha) connecting characters is detected using
     horizontal projection profiles and removed before character segmentation,
     as its presence degrades character classification accuracy."

    The shirorekha is the topmost dense horizontal run in each text band.
    We detect it via the top-third projection profile peak and zero those rows.
    """
    row_sums = binary.sum(axis=1).astype(float)

    # Only look in the top 40% of the image where shirorekha lives
    top_h = int(binary.shape[0] * 0.40)
    top_profile = row_sums[:top_h]

    if top_profile.max() == 0:
        return binary

    # Rows whose ink density exceeds 60% of the peak are the shirorekha
    threshold = top_profile.max() * 0.60
    shirorekha_rows = np.where(top_profile > threshold)[0]

    result = binary.copy()
    result[shirorekha_rows, :] = 0
    log.debug("  Removed shirorekha at rows %s", shirorekha_rows[:3])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — Line segmentation  (Paper 6, Section 3.3)
# ─────────────────────────────────────────────────────────────────────────────

def segment_lines(binary: np.ndarray, min_height: int = 8) -> list:
    """
    Paper 6, Section 3.3:
    "Lines are extracted using horizontal projection profiles.
     Rows with zero ink count mark line boundaries."

    Returns list of (y_start, y_end, line_image) tuples.
    """
    profile = binary.sum(axis=1)
    lines = []
    in_line = False
    y_start = 0

    for y, val in enumerate(profile):
        if val > 0 and not in_line:
            in_line, y_start = True, y
        elif val == 0 and in_line:
            in_line = False
            chunk = binary[y_start:y, :]
            if chunk.shape[0] >= min_height:
                lines.append((y_start, y, chunk))

    if in_line:
        chunk = binary[y_start:, :]
        if chunk.shape[0] >= min_height:
            lines.append((y_start, binary.shape[0], chunk))

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Word segmentation  (Paper 6, Section 3.3)
# ─────────────────────────────────────────────────────────────────────────────

def segment_words(line_img: np.ndarray, min_width: int = 4) -> list:
    """
    Paper 6, Section 3.3:
    "Words within a line are separated using vertical projection profiles.
     Columns with zero ink count mark word boundaries."

    Returns list of word images.
    """
    profile = line_img.sum(axis=0)
    words = []
    in_word = False
    x_start = 0

    for x, val in enumerate(profile):
        if val > 0 and not in_word:
            in_word, x_start = True, x
        elif val == 0 and in_word:
            in_word = False
            word = line_img[:, x_start:x]
            if word.shape[1] >= min_width:
                words.append(word)

    if in_word:
        word = line_img[:, x_start:]
        if word.shape[1] >= min_width:
            words.append(word)

    return words


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — Zone-based character segmentation  (Paper 6, Section 3.4)
# ─────────────────────────────────────────────────────────────────────────────

def segment_characters(word_img: np.ndarray, min_width: int = 4) -> list:
    """
    Paper 6, Section 3.4 — Zone-based segmentation with median bisection:

    "The word image is divided into three horizontal zones:
      - Upper zone  : vowel modifiers (matras) above the baseline
      - Middle zone : main character body
      - Lower zone  : subscript modifiers below the baseline

     Characters are isolated using vertical projection profiles within the
     middle zone, then the full column strip (all three zones) is extracted
     as the character patch."

    Returns list of character patch images (variable width, full word height).
    """
    h, w = word_img.shape

    # Define three horizontal zones (Paper 6, Section 3.4)
    upper_end  = h // 4          # top 25%  — vowel matras
    middle_end = (h * 3) // 4    # mid 50%  — main body
    # lower zone = middle_end:h

    # Use only the middle zone for vertical projection (avoids matra bleed)
    middle_zone = word_img[upper_end:middle_end, :]
    profile = middle_zone.sum(axis=0)

    chars = []
    in_char = False
    x_start = 0

    for x, val in enumerate(profile):
        if val > 0 and not in_char:
            in_char, x_start = True, x
        elif val == 0 and in_char:
            in_char = False
            char_patch = word_img[:, x_start:x]
            if char_patch.shape[1] >= min_width:
                chars.append(char_patch)

    if in_char:
        char_patch = word_img[:, x_start:]
        if char_patch.shape[1] >= min_width:
            chars.append(char_patch)

    return chars


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — CNN character classification  (Paper 6, Section 4)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_patch(char_img: np.ndarray, size: int = IMG_SIZE) -> "torch.Tensor":
    """
    Resize character patch to size×size and convert to normalised tensor.
    Paper 6 uses 32×32 greyscale input patches.
    """

    # Pad to square preserving aspect ratio
    h, w = char_img.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = char_img

    # Resize to model input size
    resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)

    # Normalise to [0, 1] and add batch + channel dims  → (1, 1, 32, 32)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0).unsqueeze(0)


def classify_character(
    char_img: np.ndarray,
    model,
    device,
    threshold: float = 0.3,
) -> str:
    """
    Run the CNN on a single character patch.
    Returns the predicted Unicode character, or '?' if confidence is below threshold.
    """
    import torch

    with torch.no_grad():
        tensor = prepare_patch(char_img).to(device)
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    if conf.item() < threshold:
        return "?"

    return DEVANAGARI_CLASSES[idx.item() % len(DEVANAGARI_CLASSES)]


# ─────────────────────────────────────────────────────────────────────────────
# Tesseract fallback (Paper 7)
# ─────────────────────────────────────────────────────────────────────────────

def tesseract_ocr(binary: np.ndarray, lang: str = "san") -> str:
    """
    Fallback when CNN weights are unavailable.
    Uses Tesseract with LSTM engine and Sanskrit language data (Paper 7).
    """
    try:
        import pytesseract
    except ImportError:
        log.error("pytesseract not installed. Run: pip install pytesseract")
        return ""

    pil_img = Image.fromarray(cv2.bitwise_not(binary))
    config = f"--oem 1 --psm 3 -l {lang}"
    return pytesseract.image_to_string(pil_img, config=config).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Full page → text  (assembles all stages for one page)
# ─────────────────────────────────────────────────────────────────────────────

def page_to_text(
    pil_page: Image.Image,
    model,
    device,
    dpi: int = 300,
    use_tesseract_fallback: bool = False,
    tesseract_lang: str = "san",
) -> str:
    """
    Runs the full Paper 6 pipeline on one page image.
    """
    # Stage 1 — Binarize
    binary = binarize(pil_page, dpi=dpi)

    # Stage 2 — Deskew
    binary = deskew(binary)

    # Stage 3 — Remove shirorekha
    binary = remove_shirorekha(binary)

    # Tesseract fallback path
    if use_tesseract_fallback or model is None:
        return tesseract_ocr(binary, lang=tesseract_lang)

    # CNN path: Stages 4 → 7
    page_text_lines = []

    # Stage 4 — Line segmentation
    text_lines = segment_lines(binary)
    log.debug("  %d lines", len(text_lines))

    for _, _, line_img in text_lines:
        line_chars = []

        # Stage 5 — Word segmentation
        words = segment_words(line_img)

        for word_img in words:
            word_chars = []

            # Stage 6 — Character segmentation
            chars = segment_characters(word_img)

            if not chars:
                # Fallback: treat the whole word as one character patch
                chars = [word_img]

            # Stage 7 — CNN classification
            for char_img in chars:
                predicted = classify_character(char_img, model, device)
                word_chars.append(predicted)

            line_chars.append("".join(word_chars))

        # Re-join words with a space
        page_text_lines.append(" ".join(line_chars))

    return "\n".join(page_text_lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: str,
    output_path: str,
    dpi: int = 300,
    weights: Optional[str] = None,
    use_tesseract_fallback: bool = False,
    tesseract_lang: str = "san",
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> None:

    pdf_path    = Path(pdf_path)
    output_path = Path(output_path)

    if not pdf_path.exists():
        log.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load CNN (or fall back to Tesseract)
    model, device = (None, None) if use_tesseract_fallback else load_model(weights)

    # Render pages
    log.info("Rendering '%s' at %d dpi ...", pdf_path.name, dpi)
    pages = convert_from_path(
        str(pdf_path), dpi=dpi,
        first_page=start_page, last_page=end_page,
    )
    log.info("Loaded %d pages.", len(pages))

    parts = []
    for page_num, pil_page in enumerate(
        tqdm(pages, desc="Pages"), start=start_page
    ):
        log.info("─── Page %d ───", page_num)
        text = page_to_text(
            pil_page, model, device,
            dpi=dpi,
            use_tesseract_fallback=use_tesseract_fallback,
            tesseract_lang=tesseract_lang,
        )
        if text:
            parts.append(f"\n\n# Page {page_num}\n\n{text}")
        else:
            log.warning("  Page %d: no text extracted", page_num)

    full_text = "\n".join(parts)
    output_path.write_text(full_text, encoding="utf-8")
    log.info("Done — '%s'  (%d chars)", output_path, len(full_text))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sanskrit PDF → text using Paper 6 CNN pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf",  help="Path to input Sanskrit PDF")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .txt path (default: <stem>_output.txt)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Render DPI. Use 400 for old/noisy prints.")
    parser.add_argument("--weights", default=None,
                        help="Path to local CNN weights (.pth). "
                             "Auto-downloads if not provided.")
    parser.add_argument("--tesseract-fallback", action="store_true",
                        help="Skip CNN entirely and use Tesseract OCR (Paper 7). "
                             "Fast but lower quality. Requires: pip install pytesseract "
                             "and tesseract-ocr-san system package.")
    parser.add_argument("--tesseract-lang", default="san",
                        help="Tesseract language code (used only with --tesseract-fallback).")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page",   type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    output = args.output or (Path(args.pdf).stem + "_output.txt")

    process_pdf(
        pdf_path=args.pdf,
        output_path=output,
        dpi=args.dpi,
        weights=args.weights,
        use_tesseract_fallback=args.tesseract_fallback,
        tesseract_lang=args.tesseract_lang,
        start_page=10,
        end_page=20,
    )


if __name__ == "__main__":
    main()

import logging
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

LINE_THRESHOLD = 15  # px vertical tolerance for grouping words into the same line; tune if needed

# Discover Tesseract binary on Windows if not already in system PATH
if not shutil.which("tesseract"):
    for candidate_path in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]:
        if Path(candidate_path).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate_path
            break



@dataclass
class OCRResult:
    raw_text: str
    page_count: int
    confidence: float | None
    ocr_engine: str
    processing_time_ms: int
    layout: dict[str, Any] | None


def _is_native_pdf(doc: fitz.Document) -> bool:
    """Check if the PDF document contains extractable text across its pages."""
    total_text_length = 0
    for page in doc:
        text = page.get_text().strip()
        total_text_length += len(text)
        if total_text_length > 50:
            return True
    return total_text_length > 0


def _correct_coarse_orientation(gray: np.ndarray) -> np.ndarray:
    """Detect 90/180/270 degree rotation using pytesseract OSD and correct it."""
    try:
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
        rotate_angle = osd.get("rotate", 0)

        if rotate_angle == 90:
            return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate_angle == 180:
            return cv2.rotate(gray, cv2.ROTATE_180)
        elif rotate_angle == 270:
            return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    except Exception as e:
        logger.debug(f"Coarse orientation detection skipped or failed: {e}")

    return gray


def _deskew_fine(gray: np.ndarray) -> np.ndarray:
    """Detect fine tilt angle (sub-90 degree) using text contours and rotate with warpAffine."""
    try:
        # Threshold foreground text pixels
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(thresh)
        if coords is None or len(coords) < 50:
            return gray

        # Find minAreaRect angle with width/height disambiguation (OpenCV 5.0 convention)
        rect = cv2.minAreaRect(coords)
        (w_box, h_box) = rect[1]
        angle = rect[-1]
        if w_box < h_box:
            angle = angle - 90.0
        angle = ((angle + 90) % 180) - 90  # normalize into (-90, 90]

        # Rotate only if significant tilt detected
        if 0.3 < abs(angle) <= 45.0:
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                gray,
                rotation_matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,  # White document background
            )
            return rotated
    except Exception as e:
        logger.debug(f"Fine deskew failed, returning original image: {e}")

    return gray


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    """Complete image preprocessing pipeline:
    1. Grayscale
    2. Coarse orientation check (0/90/180/270 deg)
    3. Fine-angle deskew (sub-90 deg)
    4. Denoise
    5. Adaptive binarization
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Coarse orientation
    oriented = _correct_coarse_orientation(gray)

    # 3. Fine deskew
    deskewed = _deskew_fine(oriented)

    # 4. Denoise
    try:
        denoised = cv2.fastNlMeansDenoising(deskewed, h=10, templateWindowSize=7, searchWindowSize=21)
    except Exception:
        denoised = cv2.medianBlur(deskewed, 3)

    # 5. Adaptive binarization
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )

    return binary


def _reconstruct_text_from_data(data: dict) -> str:

    """Reconstruct reading-order text from Tesseract's image_to_data output.

    Groups words into lines by vertical (top) position, then sorts words
    within each line left-to-right. This works directly off word
    coordinates rather than Tesseract's internal block/paragraph layout
    analysis, which avoids the row/column scrambling that
    pytesseract.image_to_string() can produce on tables and multi-column
    layouts.

    Known limitation: words that fall in the same horizontal band are
    merged into one line, so genuine side-by-side columns or table cells
    at the same vertical height will still be concatenated onto a single
    line rather than kept in separate cells. True grid/column detection
    is out of scope for this fix.
    """
    
    words: list[dict[str, Any]] = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        try:
            conf_val = float(data["conf"][i])
        except (ValueError, TypeError):
            continue
        if conf_val < 0:
            continue
        words.append({
            "text": text,
            "left": data["left"][i],
            "top": data["top"][i],
        })

    if not words:
        return ""

    words.sort(key=lambda w: w["top"])

    lines: list[list[dict[str, Any]]] = []
    for word in words:
        placed = False
        for line in lines:
            line_top = sum(w["top"] for w in line) / len(line)
            if abs(word["top"] - line_top) <= LINE_THRESHOLD:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    lines.sort(key=lambda line: sum(w["top"] for w in line) / len(line))

    line_texts = []
    for line in lines:
        line.sort(key=lambda w: w["left"])
        line_texts.append(" ".join(w["text"] for w in line))

    return "\n".join(line_texts)




def _collect_page_data(
    processed_image: np.ndarray,
    page_num: int,
) -> tuple[str, float, list[dict[str, Any]]]:
    """Run Tesseract OCR on a processed image page, extracting full text
    (reconstructed in reading order from word coordinates), normalized page
    confidence (0.0 to 1.0), and page-tagged word confidences.
    """
    data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)

    word_confidences: list[dict[str, Any]] = []
    conf_values: list[float] = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf_str = data["conf"][i]

        try:
            conf_val = float(conf_str)
        except (ValueError, TypeError):
            continue

        if word and conf_val >= 0:
            # Normalize 0-100 Tesseract scale to 0.0-1.0 scale
            normalized_conf = round(conf_val / 100.0, 4)
            word_confidences.append({
                "text": word,
                "confidence": normalized_conf,
                "page": page_num,
            })
            conf_values.append(normalized_conf)

    page_text = _reconstruct_text_from_data(data)
    page_confidence = round(sum(conf_values) / len(conf_values), 4) if conf_values else 0.0
    return page_text, page_confidence, word_confidences


def _extract_native_pdf_text(file_path: Path, doc: fitz.Document) -> OCRResult:
    """Extract embedded text directly from a native PDF."""
    start_time = time.perf_counter()
    page_texts: list[str] = []
    page_count = len(doc)
    page_confidences: list[dict[str, Any]] = []

    for idx, page in enumerate(doc):
        text = page.get_text().strip()
        page_texts.append(text)
        page_confidences.append({
            "page": idx + 1,
            "confidence": 1.0,
        })

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    raw_text = "\n\n".join(t for t in page_texts if t)

    layout = {
        "pages": page_count,
        "tables_detected": 0,
        "page_confidences": page_confidences,
        "word_confidences": None,
    }

    return OCRResult(
        raw_text=raw_text,
        page_count=page_count,
        confidence=1.0,
        ocr_engine="pymupdf-native",
        processing_time_ms=elapsed_ms,
        layout=layout,
    )


def _extract_tesseract_from_pdf(file_path: Path, doc: fitz.Document) -> OCRResult:
    """Render PDF pages to images, preprocess, and execute Tesseract OCR."""
    start_time = time.perf_counter()
    page_texts: list[str] = []
    page_confidences: list[dict[str, Any]] = []
    all_word_confidences: list[dict[str, Any]] = []
    page_count = len(doc)

    for idx, page in enumerate(doc):
        page_num = idx + 1
        # Render page at 300 DPI for high-accuracy OCR
        pix = page.get_pixmap(dpi=300)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        processed = _preprocess_image(img_bgr)
        text, page_conf, word_confs = _collect_page_data(processed, page_num=page_num)

        page_texts.append(text)
        page_confidences.append({
            "page": page_num,
            "confidence": page_conf,
        })
        all_word_confidences.extend(word_confs)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    raw_text = "\n\n".join(t for t in page_texts if t)

    all_confs = [w["confidence"] for w in all_word_confidences]
    overall_confidence = round(sum(all_confs) / len(all_confs), 4) if all_confs else 0.0

    layout = {
        "pages": page_count,
        "tables_detected": 0,
        "page_confidences": page_confidences,
        "word_confidences": all_word_confidences,
    }

    return OCRResult(
        raw_text=raw_text,
        page_count=page_count,
        confidence=overall_confidence,
        ocr_engine="tesseract",
        processing_time_ms=elapsed_ms,
        layout=layout,
    )


def _extract_tesseract_from_image(file_path: Path) -> OCRResult:
    """Preprocess and run Tesseract OCR on a single image file."""
    start_time = time.perf_counter()
    image = cv2.imread(str(file_path))
    if image is None:
        pil_img = Image.open(file_path).convert("RGB")
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    processed = _preprocess_image(image)
    text, page_conf, word_confs = _collect_page_data(processed, page_num=1)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    layout = {
        "pages": 1,
        "tables_detected": 0,
        "page_confidences": [{"page": 1, "confidence": page_conf}],
        "word_confidences": word_confs,
    }

    return OCRResult(
        raw_text=text,
        page_count=1,
        confidence=page_conf,
        ocr_engine="tesseract",
        processing_time_ms=elapsed_ms,
        layout=layout,
    )


_PADDLE_OCR: Any | None = None
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").strip().lower()
PADDLEOCR_DEVICE = os.getenv("PADDLEOCR_DEVICE", "cpu").strip()


def _get_paddle_ocr() -> Any:
    """Load PaddleOCR lazily so the Tesseract baseline needs no new dependency."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install backend/requirements-ocr-paddle.txt "
            "and the appropriate PaddlePaddle runtime."
        ) from exc

    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        logger.info("Loading PaddleOCR with device=%s", PADDLEOCR_DEVICE)
        _PADDLE_OCR = PaddleOCR(
            lang="en",
            device=PADDLEOCR_DEVICE,
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    return _PADDLE_OCR


def _paddle_result_dict(result: Any) -> dict[str, Any]:
    """Normalize PaddleOCR result objects to a dictionary."""
    if isinstance(result, dict):
        return result

    value = getattr(result, "json", None)
    if callable(value):
        value = value()
    if isinstance(value, str):
        import json
        value = json.loads(value)

    if isinstance(value, dict):
        if len(value) == 1:
            nested = next(iter(value.values()))
            if isinstance(nested, dict) and (
                "rec_texts" in nested or "rec_boxes" in nested
            ):
                return nested
        return value

    raise TypeError("Unsupported PaddleOCR result format")


def _extract_paddle_page(
    image: np.ndarray,
    page_num: int,
) -> tuple[str, float, list[dict[str, Any]]]:
    """Run PaddleOCR and preserve recognized text, scores, and coordinates."""
    results = list(_get_paddle_ocr().predict(image))
    if not results:
        return "", 0.0, []

    result = _paddle_result_dict(results[0])
    texts = result.get("rec_texts") or []
    scores = result.get("rec_scores") or []
    boxes = result.get("rec_boxes") or []

    words: list[dict[str, Any]] = []
    for i, value in enumerate(texts):
        text = str(value).strip()
        if not text:
            continue

        score = float(scores[i]) if i < len(scores) else 0.0
        box = boxes[i] if i < len(boxes) else None
        coords = np.asarray(box).reshape(-1).tolist() if box is not None else []

        if len(coords) >= 4:
            x1, y1, x2, y2 = map(float, coords[:4])
        else:
            x1 = y1 = x2 = y2 = 0.0

        words.append({
            "text": text,
            "confidence": round(score, 4),
            "page": page_num,
            "left": x1,
            "top": y1,
            "right": x2,
            "bottom": y2,
        })

    # Coordinate-aware reading order. Full table/column parsing is intentionally
    # left for the later PP-Structure/layout step.
    words.sort(key=lambda item: (item["top"], item["left"]))
    text = "\n".join(item["text"] for item in words)
    confidence = (
        round(sum(item["confidence"] for item in words) / len(words), 4)
        if words else 0.0
    )
    return text, confidence, words


def _extract_paddle_from_pdf(doc: fitz.Document) -> OCRResult:
    start_time = time.perf_counter()
    page_texts: list[str] = []
    page_confidences: list[dict[str, Any]] = []
    all_word_confidences: list[dict[str, Any]] = []

    for idx, page in enumerate(doc):
        page_num = idx + 1
        pix = page.get_pixmap(dpi=300)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            (pix.height, pix.width, pix.n)
        )

        if pix.n == 4:
            image = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif pix.n == 3:
            image = arr
        else:
            image = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

        text, page_conf, word_confs = _extract_paddle_page(image, page_num)
        page_texts.append(text)
        page_confidences.append({"page": page_num, "confidence": page_conf})
        all_word_confidences.extend(word_confs)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    all_confs = [item["confidence"] for item in all_word_confidences]
    confidence = round(sum(all_confs) / len(all_confs), 4) if all_confs else 0.0

    return OCRResult(
        raw_text="\n\n".join(text for text in page_texts if text),
        page_count=len(doc),
        confidence=confidence,
        ocr_engine="paddleocr",
        processing_time_ms=elapsed_ms,
        layout={
            "pages": len(doc),
            "tables_detected": 0,
            "page_confidences": page_confidences,
            "word_confidences": all_word_confidences,
        },
    )


def _extract_paddle_from_image(file_path: Path) -> OCRResult:
    start_time = time.perf_counter()
    image = cv2.imread(str(file_path))
    if image is None:
        image = np.asarray(Image.open(file_path).convert("RGB"))
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    text, confidence, word_confs = _extract_paddle_page(image, 1)
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    return OCRResult(
        raw_text=text,
        page_count=1,
        confidence=confidence,
        ocr_engine="paddleocr",
        processing_time_ms=elapsed_ms,
        layout={
            "pages": 1,
            "tables_detected": 0,
            "page_confidences": [{"page": 1, "confidence": confidence}],
            "word_confidences": word_confs,
        },
    )

def extract_text(file_path: Path, file_type: str) -> OCRResult:
    """Extract text using the configured OCR engine for scanned documents."""
    if not file_path.exists():
        raise FileNotFoundError(f"Document file not found at: {file_path}")

    if file_type == "application/pdf" or file_path.suffix.lower() == ".pdf":
        doc = fitz.open(file_path)
        try:
            if _is_native_pdf(doc):
                return _extract_native_pdf_text(file_path, doc)

            if OCR_ENGINE == "paddleocr":
                try:
                    return _extract_paddle_from_pdf(doc)
                except Exception as exc:
                    logger.warning("PaddleOCR failed; falling back to Tesseract: %s", exc)

            return _extract_tesseract_from_pdf(file_path, doc)
        finally:
            doc.close()

    if (
        file_type in ("image/jpeg", "image/png", "image/jpg")
        or file_path.suffix.lower() in (".jpg", ".jpeg", ".png")
    ):
        if OCR_ENGINE == "paddleocr":
            try:
                return _extract_paddle_from_image(file_path)
            except Exception as exc:
                logger.warning("PaddleOCR failed; falling back to Tesseract: %s", exc)

        return _extract_tesseract_from_image(file_path)

    raise ValueError(f"Unsupported file type for OCR: {file_type}")

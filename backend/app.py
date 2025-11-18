"""
Fresh minimal backend (EasyOCR, no Tesseract, no Docker).
Place this file in backend/ next to requirements.txt and run-win.bat.

Serves UI from ../frontend (so your index.html, css, and JS remain unchanged).
Endpoint:
  POST /api/verify  -> multipart form with key 'file' (image or PDF)
  GET  /            -> serves frontend/index.html
  GET  /<path:...>  -> serves other static frontend files
"""

import os
import io
import hashlib
import traceback
from flask import Flask, request, jsonify, send_file, send_from_directory, abort
from PIL import Image, ImageOps
import numpy as np

# Optional libs (if installed). We handle their absence gracefully.
try:
    import easyocr
    EASYOCR_OK = True
    # Create global reader once to avoid reloading models per request
    reader = easyocr.Reader(["en"], gpu=False)
except Exception:
    EASYOCR_OK = False
    reader = None

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_OK = True
except Exception:
    PDF2IMAGE_OK = False

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(BACKEND_DIR, "..", "frontend"))

# If frontend doesn't exist, fall back to serving files from backend folder root
if not os.path.isdir(FRONTEND_DIR):
    FRONTEND_DIR = BACKEND_DIR

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    template_folder=FRONTEND_DIR,
)

def sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()

def pil_from_bytes(b: bytes):
    """Open bytes as a PIL RGB image."""
    return Image.open(io.BytesIO(b)).convert("RGB")

def preprocess_for_ocr(pil_img, min_dim=1200):
    """Basic preprocessing: grayscale, autocontrast, upscale small images for better OCR."""
    img = pil_img.convert("L")
    w, h = img.size
    # Upscale small images
    if max(w, h) < min_dim:
        scale = int(min_dim / max(w, h)) + 1
        img = img.resize((w * scale, h * scale), Image.BICUBIC)
    img = ImageOps.autocontrast(img)
    return img

def ocr_easyocr_pil(pil_img):
    """Run EasyOCR on a PIL image. Returns (text, avg_confidence, raw_results)."""
    if not EASYOCR_OK or reader is None:
        raise RuntimeError("EasyOCR not installed or failed to initialize.")
    arr = np.array(pil_img.convert("RGB"))
    results = reader.readtext(arr, detail=1)  # list of (bbox, text, conf)
    texts = []
    confs = []
    for r in results:
        if len(r) >= 2:
            texts.append(r[1])
        if len(r) >= 3 and isinstance(r[2], (int, float)):
            confs.append(r[2])
    full_text = "\n".join(texts)
    avg_conf = (sum(confs) / len(confs)) if confs else None
    return full_text, avg_conf, results

def convert_pdf_bytes_to_pil_list(pdf_bytes, dpi=200):
    """Use pdf2image to convert PDF to list of PIL images (requires poppler)."""
    if not PDF2IMAGE_OK:
        raise RuntimeError("pdf2image is not installed. Install pdf2image and poppler to enable PDF uploads.")
    return convert_from_bytes(pdf_bytes, dpi=dpi)

# Lightweight heuristics
KNOWN_ISSUERS = [
    "institute", "university", "college", "certified", "certificate", "issued", "degree",
    "coursera", "edx", "udemy", "iit", "mit", "stanford", "google", "microsoft"
]

def detect_issuer_in_text(text: str):
    t = (text or "").lower()
    found = set()
    for kw in KNOWN_ISSUERS:
        if kw in t:
            found.add(kw)
    return sorted(found)

def quick_trust_score(word_count:int, issuer_count:int, has_dates:bool):
    score = 30
    if word_count > 50:
        score += 30
    score += min(30, issuer_count * 10)
    if has_dates:
        score += 10
    return max(0, min(100, score))

# ---- Routes to serve frontend files without changing UI ----
@app.route("/", methods=["GET"])
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)
    return "index.html not found in frontend folder.", 404

@app.route("/<path:filename>", methods=["GET"])
def serve_frontend_file(filename):
    # serve files from frontend folder (css, js, assets). Don't allow escaping base dir.
    safe_path = os.path.normpath(os.path.join(FRONTEND_DIR, filename))
    if not safe_path.startswith(os.path.abspath(FRONTEND_DIR)):
        return abort(403)
    if os.path.exists(safe_path):
        return send_file(safe_path)
    return abort(404)

# ---- OCR API ----
@app.route("/api/verify", methods=["POST"])
def api_verify():
    """
    Accepts multipart/form-data with key 'file' containing an image or a PDF.
    Returns JSON with:
      - file_name, file_hash, page_count
      - full_text, snippet, word_count
      - detected_issuers, has_dates, avg_confidence, trust_score
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "no file part named 'file' in request"}), 400
        f = request.files["file"]
        filename = f.filename or "upload"
        raw = f.read()
        file_hash = sha256_bytes(raw)

        texts = []
        confidences = []
        page_count = 0

        is_pdf = filename.lower().endswith(".pdf") or raw[:4] == b"%PDF"
        if is_pdf:
            if not PDF2IMAGE_OK:
                return jsonify({"error": "pdf_uploaded_but_pdf2image_missing", "message": "Install pdf2image and poppler to enable PDF support."}), 400
            try:
                pages = convert_pdf_bytes_to_pil_list(raw, dpi=200)
            except Exception as e:
                tb = traceback.format_exc()
                return jsonify({"error": "pdf_conversion_failed", "details": str(e), "trace": tb}), 500
            for p in pages:
                page_count += 1
                pre = preprocess_for_ocr(p)
                try:
                    txt, conf, meta = ocr_easyocr_pil(pre)
                    texts.append(txt)
                    if conf is not None:
                        confidences.append(conf)
                except Exception:
                    texts.append("")
        else:
            # image
            try:
                pil = pil_from_bytes(raw)
            except Exception as e:
                return jsonify({"error": "cannot_read_image", "details": str(e)}), 400
            pre = preprocess_for_ocr(pil)
            try:
                txt, conf, meta = ocr_easyocr_pil(pre)
                texts.append(txt)
                if conf is not None:
                    confidences.append(conf)
                page_count = 1
            except Exception as e:
                tb = traceback.format_exc()
                return jsonify({"error": "ocr_failed", "details": str(e), "trace": tb}), 500

        full_text = "\n\n---PAGE---\n\n".join(texts).strip()
        snippet = (full_text[:1000] + "...") if len(full_text) > 1000 else full_text
        words = [w for w in full_text.split() if w.strip()]
        word_count = len(words)
        issuers = detect_issuer_in_text(full_text)
        has_dates = any(tok.isdigit() and (len(tok) == 4 and 1900 <= int(tok) <= 2100) for tok in words) or any(k in (full_text or "").lower() for k in ["issued on", "date of", "on:"])
        avg_conf = (sum(confidences)/len(confidences)) if confidences else None
        trust = quick_trust_score(word_count, len(issuers), has_dates)

        return jsonify({
            "file_name": filename,
            "file_hash": file_hash,
            "page_count": page_count,
            "full_text": full_text,
            "extracted_text_snippet": snippet,
            "word_count": word_count,
            "detected_issuers": issuers,
            "has_dates": has_dates,
            "avg_ocr_confidence": avg_conf,
            "trust_score": trust
        }), 200

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": "internal_server_error", "details": str(e), "trace": tb}), 500

# ---- startup ----
if __name__ == "__main__":
    print("Starting fresh EasyOCR backend (no Tesseract, no Docker).")
    print("Frontend served from:", FRONTEND_DIR)
    print("EasyOCR available:", EASYOCR_OK)
    print("pdf2image/poppler available:", PDF2IMAGE_OK)
    # Debug True for local dev; set False for production
    app.run(host="0.0.0.0", port=5000, debug=True)

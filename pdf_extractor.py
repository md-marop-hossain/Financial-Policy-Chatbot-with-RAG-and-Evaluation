import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io, os, re, unicodedata

# ----------Windows Tesseract Configuration ----------
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

lang = "eng"

pdf_path = "data/For Task - Policy file.pdf"
output_path = "data/cleaned_text.txt"
os.makedirs("data", exist_ok=True)

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_with_ocr_fallback(pdf_path: str, lang: str = "eng") -> str:
    """
    1) Try native text extraction (fast, accurate for digital PDFs).
    2) If a page has little/no text, fallback to OCR for that page.
    3) Prefix each page’s content with [Page N] for source tracking.
    """
    doc = fitz.open(pdf_path)
    all_text = []

    for i, page in enumerate(doc):
        print(f"Processing page {i + 1}/{len(doc)}")
        
        text = page.get_text("text") or ""
        text = clean_text(text)
       
        if len(text) < 20:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            try:
                text = pytesseract.image_to_string(img, lang=lang)
            except pytesseract.TesseractError:
                text = pytesseract.image_to_string(img, lang="eng")
            text = clean_text(text)

        if text:
            all_text.append(f"[Page {i + 1}] {text}")

    return "\n\n".join(all_text)

if __name__ == "__main__":
    result = extract_text_with_ocr_fallback(pdf_path, lang=lang)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    print("✅ Extraction completed (English) and saved to", output_path)

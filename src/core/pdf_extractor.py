import io
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

def extract_text_with_ocr_fallback(pdf_bytes_or_file) -> str:
    """
    Extracts text from a PDF. If the PDF contains native text layers, it uses pdfplumber.
    If the text is too sparse (likely a scanned image-based PDF), it falls back to OCR using pytesseract.
    
    Args:
        pdf_bytes_or_file: File-like object (e.g., from st.file_uploader)
    
    Returns:
        str: Extracted text.
    """
    # 1. Try native extraction first
    try:
        pdf_bytes_or_file.seek(0)
        text_parts = []
        with pdfplumber.open(pdf_bytes_or_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
        
        native_text = "\n".join(text_parts)
        
        # Heuristic: If we extracted less than 20 chars per page on average, it's probably scanned
        avg_chars_per_page = len(native_text) / max(1, len(pdf.pages))
        if avg_chars_per_page > 20:
            return native_text.strip()
    except Exception as e:
        print(f"Native extraction failed: {e}")
        # Reset pointer for fallback
        pdf_bytes_or_file.seek(0)
    
    # 2. Fallback to OCR because text was sparse or native extraction failed
    print("Falling back to OCR extraction...")
    try:
        pdf_bytes_or_file.seek(0)
        pdf_bytes = pdf_bytes_or_file.read()
        
        # Convert PDF bytes to list of PIL Images
        images = convert_from_bytes(pdf_bytes)
        ocr_text_parts = []
        
        for i, img in enumerate(images):
            # Extract text from image using Tesseract
            page_text = pytesseract.image_to_string(img)
            ocr_text_parts.append(page_text)
            
        return "\n".join(ocr_text_parts).strip()
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return ""

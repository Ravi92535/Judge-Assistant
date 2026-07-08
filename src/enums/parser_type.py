from enum import Enum

class ParserType(Enum):
    PDF = "pdf_parser"
    OCR = "ocr_parser"
    DOCX = "docx_parser"
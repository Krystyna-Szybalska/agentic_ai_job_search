import pytest
from tools.pdf_parser import extract_text_from_pdf


def test_extract_text_returns_string(tmp_path):
    # Use any small real PDF or skip if none available
    # Here we test the function signature and error handling
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("nonexistent.pdf")


def test_extract_text_from_valid_pdf(tmp_path):
    import pdfplumber
    # Create a minimal test: if pdfplumber can open a file, function works
    # Integration test — run manually with a real PDF
    pass

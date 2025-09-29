# data_loader.py

from pathlib import Path
from typing import Union
from docx import Document
import fitz  # PyMuPDF


def _read_pdf(path: Union[str, Path]) -> str:
    text_parts = []
    with fitz.open(str(path)) as pdf:
        for page in pdf:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)


def _read_docx(path: Union[str, Path]) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_txt(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_resume(file_path: Union[str, Path]) -> str:
    """
    Load and return raw text from a resume file (.pdf, .docx, .txt).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _read_pdf(path)
    elif ext == ".docx":
        text = _read_docx(path)
    elif ext == ".txt":
        text = _read_txt(path)
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")

    return text.strip()
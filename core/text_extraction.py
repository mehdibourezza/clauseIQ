"""
Turns an uploaded file (PDF or plain text) into the contract text we analyse.

Scanned PDFs contain images, not text: pypdf returns an empty string for them.
We reject those explicitly instead of sending an empty contract to the LLM.
OCR (e.g. Azure Document Intelligence) is out of scope for the prototype.
"""

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_EXTENSIONS = (".pdf", ".txt")


class DocumentError(ValueError):
    """The uploaded file cannot be turned into contract text. The message is user-facing."""


def extract_text(filename: str, data: bytes) -> str:
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except PdfReadError as exc:
            raise DocumentError(f"Unreadable PDF: {exc}") from exc
    elif extension == ".txt":
        text = data.decode("utf-8", errors="replace")
    else:
        raise DocumentError(
            f"Unsupported file type '{extension}'. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if not text.strip():
        raise DocumentError("No extractable text (scanned PDFs are not supported yet).")
    return text

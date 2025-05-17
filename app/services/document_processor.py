import os
import io
import uuid
from pathlib import Path

import pytesseract
from PIL import Image
from pypdf import PdfReader
import docx  # python-docx

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import UPLOAD_DIR, PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """Service for processing various document types and extracting text content."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def save_uploaded_file(self, file):
        """Save an uploaded file to disk and return the file path."""
        file_ext = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename

        with open(file_path, "wb") as f:
            f.write(file.file.read())

        return file_path, unique_filename

    def process_document(self, file):
        """Process a document file and extract text content."""
        file_path, unique_id = self.save_uploaded_file(file)
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext == ".pdf":
            text = self._extract_text_from_pdf(file_path)
        elif file_ext == ".docx":
            text = self._extract_text_from_docx(file_path)
        elif file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            text = self._extract_text_from_image(file_path)
        elif file_ext in [".txt", ".csv", ".md", ".json", ".log"]:
            # For text-like files just read text
            text = self._extract_text_from_text_file(file_path)
        else:
            # For unsupported types, try to read as text or raise error
            try:
                text = self._extract_text_from_text_file(file_path)
            except Exception:
                raise ValueError(f"Unsupported file type: {file_ext}")

        text_file_path = PROCESSED_DIR / f"{unique_id}.txt"
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(text)

        doc = {
            "id": unique_id,
            "filename": file.filename,
            "text": text,
            "path": str(file_path),
            "processed_path": str(text_file_path)
        }

        chunks = self._chunk_text(text, doc)

        return doc, chunks

    def _extract_text_from_pdf(self, file_path):
        text = ""
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text += f"Page {i+1}:\n{page_text}\n\n"
        return text

    def _extract_text_from_docx(self, file_path):
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)

    def _extract_text_from_image(self, file_path):
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text

    def _extract_text_from_text_file(self, file_path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _chunk_text(self, text, metadata):
        docs = self.text_splitter.create_documents([text], [metadata])
        return docs

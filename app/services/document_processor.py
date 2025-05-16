import os
import io
import pytesseract
from PIL import Image
from pypdf import PdfReader
import uuid
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import UPLOAD_DIR, PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """Service for processing various document types and extracting text content."""

    def __init__(self):
        """Initialize the document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def save_uploaded_file(self, file):
        """Save an uploaded file to disk and return the file path."""
        # Generate a unique filename to avoid conflicts
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save file to disk
        with open(file_path, "wb") as f:
            f.write(file.file.read())
            
        return file_path, unique_filename

    def process_document(self, file):
        """Process a document file and extract text content."""
        file_path, unique_id = self.save_uploaded_file(file)
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # Extract text based on file type
        if file_ext == ".pdf":
            text = self._extract_text_from_pdf(file_path)
        elif file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            text = self._extract_text_from_image(file_path)
        else:
            # For text files or other formats, just read as text
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        
        # Save the extracted text for future reference
        text_file_path = PROCESSED_DIR / f"{unique_id}.txt"
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Create document with metadata
        doc = {
            "id": unique_id,
            "filename": file.filename,
            "text": text,
            "path": str(file_path),
            "processed_path": str(text_file_path)
        }
        
        # Split text into chunks
        chunks = self._chunk_text(text, doc)
        
        return doc, chunks

    def _extract_text_from_pdf(self, file_path):
        """Extract text from a PDF file."""
        text = ""
        reader = PdfReader(file_path)
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text += f"Page {i+1}:\n{page_text}\n\n"
        
        return text

    def _extract_text_from_image(self, file_path):
        """Extract text from an image file using OCR."""
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text

    def _chunk_text(self, text, metadata):
        """Split text into manageable chunks with metadata."""
        docs = self.text_splitter.create_documents([text], [metadata])
        return docs
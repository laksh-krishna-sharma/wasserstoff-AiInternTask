from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    id: str
    filename: str
    path: Optional[str] = None
    processed_path: Optional[str] = None


class DocumentResponse(BaseModel):
    """Response model for a single document."""
    doc_id: str
    filename: str
    extracted_answer: str
    citation: str
    relevance: int


class ThemeModel(BaseModel):
    """Model for a theme identified across documents."""
    theme_name: str
    theme_description: str
    supporting_documents: List[str]
    confidence: int


class QueryResponse(BaseModel):
    """Response model for a query."""
    document_responses: List[DocumentResponse]
    identified_themes: List[ThemeModel]
    synthesized_answer: str


class DocumentList(BaseModel):
    """List of documents."""
    documents: List[DocumentMetadata]


class QueryRequest(BaseModel):
    """Request model for a query."""
    query: str
    document_ids: Optional[List[str]] = None  # Optional list of document IDs to filter by
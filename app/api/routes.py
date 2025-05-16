from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from typing import List, Optional
import json
import os

from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.query_processor import QueryProcessor
from app.models.models import DocumentList, QueryResponse, QueryRequest

router = APIRouter()
document_processor = DocumentProcessor()
vector_store = VectorStore()
query_processor = QueryProcessor()


@router.post("/documents/upload", response_model=DocumentList)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload multiple documents and process them.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    processed_docs = []
    
    for file in files:
        try:
            # Process document and extract text
            doc, chunks = document_processor.process_document(file)
            
            # Add document chunks to vector store
            vector_store.add_documents(chunks)
            
            # Add processed document to response
            processed_docs.append({
                "id": doc["id"],
                "filename": doc["filename"],
                "path": doc["path"],
                "processed_path": doc["processed_path"]
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")
    
    return {"documents": processed_docs}


@router.get("/documents", response_model=DocumentList)
async def get_documents():
    """
    Get all uploaded documents.
    """
    try:
        collection = vector_store.get_all_documents()
        
        # Extract unique document IDs and metadata
        documents = {}
        for i, metadata in enumerate(collection["metadatas"]):
            doc_id = metadata.get("id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "id": doc_id,
                    "filename": metadata.get("filename", "Unknown"),
                    "path": metadata.get("path", ""),
                    "processed_path": metadata.get("processed_path", "")
                }
        
        return {"documents": list(documents.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query against the documents.
    """
    try:
        # Process query with themes
        response = query_processor.process_query_with_themes(request.query)
        
        # Filter by document IDs if provided
        if request.document_ids:
            response.document_responses = [
                doc for doc in response.document_responses 
                if doc.doc_id in request.document_ids
            ]
            
            # Update themes based on filtered documents
            for theme in response.identified_themes:
                theme.supporting_documents = [
                    doc_id for doc_id in theme.supporting_documents 
                    if doc_id in request.document_ids
                ]
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
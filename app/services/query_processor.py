from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from app.config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY
from app.services.vector_store import VectorStore


class DocumentResponse(BaseModel):
    """Schema for a response from a single document."""
    doc_id: str = Field(description="Unique identifier for the document")
    filename: str = Field(description="Original filename of the document")
    extracted_answer: str = Field(description="Relevant information extracted from the document")
    citation: str = Field(description="Page, paragraph, or section where the information is found")
    relevance: int = Field(description="Relevance score from 1-10", ge=1, le=10)


class ThemeResponse(BaseModel):
    """Schema for a theme identified across documents."""
    theme_name: str = Field(description="Name of the identified theme")
    theme_description: str = Field(description="Description of the theme")
    supporting_documents: List[str] = Field(description="List of document IDs supporting this theme")
    confidence: int = Field(description="Confidence score from 1-10", ge=1, le=10)


class SynthesizedResponse(BaseModel):
    """Schema for the final synthesized response."""
    document_responses: List[DocumentResponse] = Field(description="Responses from individual documents")
    identified_themes: List[ThemeResponse] = Field(description="Themes identified across documents")
    synthesized_answer: str = Field(description="Final synthesized answer to the query")


class QueryProcessor:
    """Service for processing queries against documents and identifying themes."""

    def __init__(self):
        """Initialize the query processor."""
        self.vector_store = VectorStore()
        self.llm = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            groq_api_key=GROQ_API_KEY
        )

    def process_query(self, query, top_k=10):
        """Process a query against documents and return individual responses."""
        # Get relevant document chunks
        document_chunks = self.vector_store.similarity_search(query, k=top_k)
        
        # Group chunks by document
        doc_chunks = {}
        for doc, score in document_chunks:
            doc_id = doc.metadata.get("id")
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = {
                    "chunks": [],
                    "filename": doc.metadata.get("filename"),
                    "score": score
                }
            doc_chunks[doc_id]["chunks"].append(doc.page_content)
        
        # Extract answers from each document
        document_responses = []
        for doc_id, doc_data in doc_chunks.items():
            answer = self._extract_answer_from_document(
                query, 
                doc_data["chunks"], 
                doc_id, 
                doc_data["filename"]
            )
            document_responses.append(answer)
        
        # Sort by relevance
        document_responses.sort(key=lambda x: x.relevance, reverse=True)
        
        return document_responses

    def _extract_answer_from_document(self, query, chunks, doc_id, filename):
        """Extract an answer from document chunks."""
        # Combine chunks into context
        context = "\n\n".join(chunks)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analyzer. 
            Extract the most relevant information from the document context that answers the query.
            Include specific page numbers, paragraphs, or sections if possible.
            If the document doesn't contain relevant information, state that clearly.
            
            Rate the relevance of the document to the query on a scale of 1-10, 
            where 10 is perfectly relevant and 1 is not relevant at all."""),
            ("user", """
            Query: {query}
            
            Document Context: {context}
            
            Return your answer in the following format:
            {{
                "doc_id": "{doc_id}",
                "filename": "{filename}",
                "extracted_answer": "The relevant information from the document",
                "citation": "Specific location (page, paragraph, section)",
                "relevance": relevance_score_1_to_10
            }}
            """)
        ])
        
        # Run chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(query=query, context=context, doc_id=doc_id, filename=filename)
        
        try:
            # Parse result to DocumentResponse
            parsed_result = json.loads(result)
            return DocumentResponse(**parsed_result)
        except Exception as e:
            print(f"Error parsing document response: {e}")
            # Return a fallback response
            return DocumentResponse(
                doc_id=doc_id,
                filename=filename,
                extracted_answer="Error processing this document.",
                citation="Unknown",
                relevance=1
            )

    def identify_themes(self, document_responses, query):
        """Identify themes across document responses."""
        if not document_responses:
            return []
        
        # Create context for theme identification
        context = []
        for resp in document_responses:
            if resp.relevance >= 3:  # Only include somewhat relevant docs
                context.append(f"Document {resp.doc_id} ({resp.filename}): {resp.extracted_answer}")
        
        context_text = "\n\n".join(context)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying themes across documents.
            Analyze the information from multiple documents and identify common themes that emerge.
            For each theme, provide a clear name, description, and list of documents that support it.
            Rate your confidence in each theme on a scale of 1-10."""),
            ("user", """
            Query: {query}
            
            Information from documents:
            {context}
            
            Identify 2-5 common themes across these documents. Return your answer as a JSON list:
            [
                {{
                    "theme_name": "Theme 1",
                    "theme_description": "Description of theme 1",
                    "supporting_documents": ["doc_id1", "doc_id2"],
                    "confidence": confidence_score_1_to_10
                }},
                ...
            ]
            """)
        ])
        
        # Run chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(query=query, context=context_text)
        
        try:
            # Parse result to list of ThemeResponse
            parsed_result = json.loads(result)
            themes = [ThemeResponse(**theme) for theme in parsed_result]
            return themes
        except Exception as e:
            print(f"Error parsing theme response: {e}")
            return []

    def synthesize_answer(self, document_responses, themes, query):
        """Synthesize a final answer based on document responses and themes."""
        # Create context
        doc_context = []
        for resp in document_responses:
            if resp.relevance >= 3:  # Only include somewhat relevant docs
                doc_context.append(f"Document {resp.doc_id} ({resp.filename}): {resp.extracted_answer}")
        
        doc_context_text = "\n\n".join(doc_context)
        
        theme_context = []
        for theme in themes:
            theme_context.append(f"Theme: {theme.theme_name}\nDescription: {theme.theme_description}\nSupporting Documents: {', '.join(theme.supporting_documents)}")
        
        theme_context_text = "\n\n".join(theme_context)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at synthesizing information from multiple documents.
            Create a comprehensive, well-structured answer to the query based on the information from documents and identified themes.
            Include specific citations to documents (DOC001, DOC002, etc.) to support your points.
            Organize your answer by themes when possible."""),
            ("user", """
            Query: {query}
            
            Document Information:
            {doc_context}
            
            Identified Themes:
            {theme_context}
            
            Provide a comprehensive answer to the query, organizing information by themes and citing specific documents.
            """)
        ])
        
        # Run chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(query=query, doc_context=doc_context_text, theme_context=theme_context_text)
        
        return result

    def process_query_with_themes(self, query):
        """Process a query with theme identification and synthesis."""
        # Get individual document responses
        document_responses = self.process_query(query)
        
        # Identify themes
        themes = self.identify_themes(document_responses, query)
        
        # Synthesize answer
        synthesized_answer = self.synthesize_answer(document_responses, themes, query)
        
        # Create final response
        response = SynthesizedResponse(
            document_responses=document_responses,
            identified_themes=themes,
            synthesized_answer=synthesized_answer
        )
        
        return response
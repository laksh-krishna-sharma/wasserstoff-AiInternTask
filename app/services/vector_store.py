import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import DB_DIR, CHROMA_COLLECTION_NAME


class VectorStore:
    """Service for interacting with the vector database."""

    def __init__(self):
        """Initialize the vector store with Hugging Face embeddings."""
        # Use open-source HuggingFace embeddings instead of OpenAI
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize or load the vector store
        self.vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(DB_DIR)
        )

    def add_documents(self, documents):
        """Add documents to the vector store."""
        try:
            ids = self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            return ids
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def similarity_search(self, query, k=5):
        """
        Search for documents similar to the query.
        Returns a list of (document, similarity_score) tuples.
        """
        try:
            docs = self.vectorstore.similarity_search_with_score(query, k=k)
            return docs
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []

    def get_all_documents(self):
        """Get all documents from the vector store."""
        try:
            # This is a simplified approach - in a real application, you might 
            # need pagination or other techniques for large document sets
            collection = self.vectorstore.get()
            return collection
        except Exception as e:
            print(f"Error retrieving documents from vector store: {e}")
            return {"ids": [], "documents": [], "metadatas": []}
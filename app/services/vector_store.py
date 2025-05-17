import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import DB_DIR, CHROMA_COLLECTION_NAME

class VectorStore:
    def __init__(self):
        # Ensure DB_DIR exists (should already be created by your config)
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(DB_DIR)
        )

    def add_documents(self, documents):
        try:
            ids = self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            return ids
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def similarity_search(self, query, k=5):
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []

    def get_all_documents(self):
        try:
            return self.vectorstore.get()
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return {"ids": [], "documents": [], "metadatas": []}

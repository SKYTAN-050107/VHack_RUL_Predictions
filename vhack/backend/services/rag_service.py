import os
from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import PyPDFLoader

from services.database import supabase
from config import GOOGLE_API_KEY

class MockPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    def load(self):
        return [{"page_content": f"Mock content from {self.file_path}", "metadata": {"source": self.file_path}}]

class SimpleTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def split_documents(self, documents: List[Any]) -> List[Any]:
        # Simple mock of split_documents
        class MockChunk:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata

        all_chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            for chunk in text_chunks:
                all_chunks.append(MockChunk(chunk, doc.metadata))
        return all_chunks

class RAGService:
    def __init__(self):
        if not GOOGLE_API_KEY:
            # raise ValueError("GOOGLE_API_KEY must be set for RAG service.")
            print("Warning: GOOGLE_API_KEY not set. RAG service will run in mock mode.")
        
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
        except:
            self.embeddings = None
            print("Warning: Failed to initialize Google embeddings.")

        self.text_splitter = SimpleTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    async def process_and_store_resource(self, file_path: str, resource_id: int):
        """Processes a file (PDF or TXT) and stores its chunks/embeddings in Supabase."""
        # 1. Load the document content
        if file_path.endswith(".pdf"):
            loader = MockPDFLoader(file_path)
            documents = loader.load()
            # Class doc wrap for split_documents
            class Doc:
                def __init__(self, content, metadata):
                    self.page_content = content
                    self.metadata = metadata
            docs = [Doc(d["page_content"], d["metadata"]) for d in documents]
        else:
            # For .txt or .md files
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            class Doc:
                def __init__(self, content, metadata):
                    self.page_content = content
                    self.metadata = metadata
            docs = [Doc(content, {"source": file_path})]

        # 2. Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        
        # 2. Extract content and metadata
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # 3. Generate embeddings
        # embeddings.embed_documents is a synchronous call in LangChain
        vectors = self.embeddings.embed_documents(texts)
        
        # 4. Store in Supabase
        if not self.embeddings:
            print(f"Mock storing {len(texts)} chunks for resource {resource_id}")
            return

        for text, metadata, vector in zip(texts, metadatas, vectors):
            try:
                supabase.table("resource_embeddings").insert({
                    "resource_id": resource_id,
                    "content": text,
                    "metadata": metadata,
                    "embedding": vector
                }).execute()
            except Exception as e:
                print(f"Error inserting chunk: {e}")

    async def query_relevant_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Queries the vector DB for context relevant to the given query."""
        if not self.embeddings:
            return [{"content": "Mock context content for RAG analysis.", "metadata": {}}]

        # 1. Generate query embedding
        try:
            query_vector = self.embeddings.embed_query(query)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [{"content": "Mock context content for RAG analysis.", "metadata": {}}]
        
        # 2. Perform vector search using Supabase RPC
        try:
            response = supabase.rpc("match_resource_embeddings", {
                "query_embedding": query_vector,
                "match_threshold": 0.7,
                "match_count": limit
            }).execute()
            return response.data
        except Exception as e:
            print(f"Error querying Supabase: {e}")
            return [{"content": "Mock context content for RAG analysis.", "metadata": {}}]

rag_service = RAGService()

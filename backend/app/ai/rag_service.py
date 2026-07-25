import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss


class RAGService:
    def __init__(self):
        # Don't load the model during app startup
        self.model = None
        self.dimension = 384  # Embedding dimension of all-MiniLM-L6-v2

    def get_model(self):
        """
        Lazily load the embedding model only when needed.
        """
        if self.model is None:
            print("Loading SentenceTransformer model...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 800,
        overlap: int = 150,
    ) -> List[str]:
        """
        Splits a document into overlapping chunks.
        """
        words = text.split()
        chunks = []

        step = max(1, chunk_size - overlap)

        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def build_vector_store(self, text: str) -> Dict[str, Any]:
        """
        Creates a FAISS vector index from the document.
        """
        chunks = self.chunk_text(text)

        if not chunks:
            return {
                "index": None,
                "chunks": []
            }

        model = self.get_model()

        embeddings = model.encode(
            chunks,
            convert_to_numpy=True
        ).astype("float32")

        index = faiss.IndexFlatL2(self.dimension)
        index.add(embeddings)

        return {
            "index": index,
            "chunks": chunks
        }

    def search_similar(
        self,
        index: faiss.Index,
        chunks: List[str],
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search.
        """
        if index is None or not chunks:
            return []

        model = self.get_model()

        query_vector = model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        distances, indices = index.search(query_vector, top_k)

        results = []

        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(chunks):
                results.append({
                    "chunk": chunks[idx],
                    "score": float(1.0 / (1.0 + distances[0][i]))
                })

        return results


# Safe to create because the model is NOT loaded in __init__
rag_service = RAGService()
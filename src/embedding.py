from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

print(
    "Embedding dimension:",
    embedding_model.get_embedding_dimension()
)


# ============================================================
# CONNECT TO PINECONE
# ============================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(
    PINECONE_INDEX_NAME
)

print("Connected to Pinecone!")
print("Index:", PINECONE_INDEX_NAME)
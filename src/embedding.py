from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ============================================================
# EMBEDDING DIMENSION
# ============================================================

# Compatible with the installed SentenceTransformer versions.
embedding_dimension = embedding_model.get_sentence_embedding_dimension()

print("Embedding dimension:", embedding_dimension)


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


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text: str):
    """
    Convert text into a 384-dimensional embedding vector.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    vector = embedding_model.encode(
        text,
        convert_to_numpy=True
    )

    return vector.tolist()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_text = "What is Agentic AI?"

    vector = create_embedding(test_text)

    print("Embedding created successfully!")
    print("Vector length:", len(vector))
    print("First 5 values:", vector[:5])
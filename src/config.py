import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")

# Pinecone configuration
PINECONE_INDEX_NAME = "agentic-ai-rag"
PINECONE_NAMESPACE = "agentic-ai"


# Validate required keys
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env")
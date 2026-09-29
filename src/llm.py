import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in .env")


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY
)


def ask_llm(prompt: str) -> str:
    response = llm.invoke(prompt)

    return response.content
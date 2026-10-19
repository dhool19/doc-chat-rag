import os 
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in .env file.")

DATA_DIR = "data"
COLLECTION_NAME="my-collection"
PERSIST_DIRECTORY="./chroma_db"

CHUNK_SIZE=1000
CHUNK_OVERLAP=200

EMBEDDING_MODEL="text-embedding-3-small"
CHAT_MODEL="gpt-4o-mini"

RETRIEVER_K = 4

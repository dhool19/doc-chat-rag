from pathlib import Path

from langchain_community.document_loaders import(
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from config import(
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    PERSIST_DIRECTORY,
)

def load_docs():
    pdf_dir = Path(DATA_DIR)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Data directory {pdf_dir} does not exist.")

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF file found in {DATA_DIR}.")

    docs = []
    for pdf in pdf_files:
        print(f" Loading: {pdf.name}")
        docs.extend(PyPDFLoader(str(pdf)).load())

    return docs

def build_vector_store():
    documents = load_docs()
    print(f"Loaded docs {len(documents)} pages.")

    text_splitters = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,        
    )

    splits = text_splitters.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=splits,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    print(f"Vectors store saved to {PERSIST_DIRECTORY} and {vector_store._collection.count()} vectors. ")

if __name__ == "__main__":
    build_vector_store()



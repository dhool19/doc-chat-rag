# doc-chat-rag
A conversational RAG (Retrieval-Augmented Generation) chatbot built with LangChain, LangSmith, ChromaDB, and Streamlit. It allows you to upload PDF documents, ask questions about them, and handle natural follow-up questions using chat history.

## Features

- **Document Ingestion:** Loads and splits local PDF files.
- **Vector Search:** Stores document embeddings locally using ChromaDB.
- **Chat History:** Remembers context for follow-up questions.
- **Streamlit UI:** Simple, interactive chat interface.

## TechStack

- **Python** (Core language)
- **LangChain** (RAG orchestration and prompt chaining)
- **LangSmith** (Tracing and debugging LLM application)
- **ChromaDB** (Local vector database)
- **OpenAI API** 
  - LLM: `gpt-4o-mini`
  - Embeddings: `text-embedding-3-small`
- **Streamlit** (Frontend UI)

## Setup & Installation

### Clone the respository and activate a virtual environment

``` bash
python -m venv venv
```
#### On Windows:
``` bash
venv\Scripts\activate
```
#### On Mac/linux:
``` bash
source venv/bin/activate
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

### Configure Environment Variables
Create a `.env` file in the root directory and add your OPENAI API key and LangSmith credentials:

```env
OPENAI_API_KEY=your_openai_api_key

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=my-langchain-app
```

### Build the vector database
Run this once(or whenever you add a new document):

``` bash
python ingest.py
```

### Run the app

``` bash
python -m streamlit run app.py
```
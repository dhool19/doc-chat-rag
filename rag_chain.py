from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from pathlib import Path
from config import(
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    PERSIST_DIRECTORY,
    RETRIEVER_K,
    CHAT_MODEL,
)

_session_store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str):
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id] 

def build_rag_chain():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.5)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might "
         "reference context in the chat history, formulate a standalone "
         "question which can be understood without the chat history. "
         "Do NOT answer the question — just reformulate it if needed, "
         "otherwise return it as is."),
         MessagesPlaceholder("chat_history"),
         ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about the provided"
         "documents. Use ONLY the retrieved context below to answer."
         "If the answer is not in the context, say you don't know.\n\n"
         "Context:\n{context}"),
         MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(
        llm, qa_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever, 
        question_answer_chain,
    )

    converstional_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return converstional_rag_chain

def vector_store_exists() -> bool:
    """True if the Chroma DB has already been built."""
    return Path(PERSIST_DIRECTORY).exists() and any(Path(PERSIST_DIRECTORY).iterdir())

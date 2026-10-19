import uuid
import streamlit as st
from rag_chain import(
    build_rag_chain,
    vector_store_exists,
    get_session_history
)

@st.cache_resource(show_spinner="Loading RAG chain...")

def load_chain():
    return build_rag_chain()

def main():
    st.title("RAG Chabot")

    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    if not vector_store_exists():
        st.error("Vector store not found.")
        return

    chain = load_chain()
    
    prompt = st.chat_input("Write your prompt")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            result = chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )

        response=result["answer"]
        st.chat_message("assistant").markdown(response)

        source_docs = result.get("context", [])
        if source_docs:
            unique_docs = []
            seen_ids = set()

            for doc in source_docs:
                source_name = doc.metadata.get("sorce", "Document")
                page_num = doc.metadata.get("page", "Unknown")
                unique_id = f"{source_name}_{page_num}"

                if unique_id not in seen_ids:
                    seen_ids.add(unique_id)
                    unique_docs.append(doc)

            with st.expander("Sources "):
                for i, doc in enumerate(unique_docs, 1):
                    source_name = doc.metadata.get("source", f"Document {i}")
                    page_num = doc.metadata.get("page", None)

                    if page_num is not None:
                        st.markdown(f"**{i}. {source_name} (Page {page_num + 1})**")
                    else:
                        st.markdown(f"**{i}. {source_name}**")

                    if i < len(unique_docs):
                        st.divider()
                    

        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Clear chat"):
        st.session_state.messages = []
        get_session_history(st.session_state.session_id).clear()
        st.rerun()

if __name__ == "__main__":
    main()
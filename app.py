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
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Clear chat"):
        st.session_state.messages = []
        get_session_history(st.session_state.session_id).clear()
        st.rerun()

if __name__ == "__main__":
    main()
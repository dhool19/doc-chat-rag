from rag_chain import build_rag_chain

def main():
    print("Welcome! (type 'exit' or 'quit' to stop)\n")
    chain = build_rag_chain()

    session_id = "client-user"

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n Bye")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye")
            break

        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        print(f"\nAssitant: {response['answer']}\n")

        sources = {doc.metadata.get("source", "unknown") for doc in response["context"]}
        print(f"Sources: {', '.join(sorted(sources))}\n")

if __name__ =="__main__":
    main()

        
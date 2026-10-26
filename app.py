
import streamlit as st

from rag_pipeline import (
    initialize_components,
    answer_question,
)

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Knowledge Assistant")
st.write(
    "Ask questions about your documents. "
    "Answers are generated using retrieved information "
    "from the indexed PDFs."
)

st.caption(
    "Knowledge base: NIST AI Risk Management Framework "
    "and AWS Data Warehousing"
)

# -----------------------------
# Initialize RAG components
# -----------------------------
@st.cache_resource(show_spinner="Loading AI models and knowledge base...")
def load_components():
    return initialize_components()


try:
    client, embedding_model, collection = load_components()
except Exception as error:
    st.error(f"Could not initialize the assistant: {error}")
    st.info(
        "Check your .env file, Gemini API key, "
        "installed packages, and ChromaDB index."
    )
    st.stop()

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display source information for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['source']}** "
                        f"(Page {source['page']})"
                    )
                    st.caption(
                        source["text"][:500]
                        + ("..." if len(source["text"]) > 500 else "")
                    )

# -----------------------------
# Ask a question
# -----------------------------
question = st.chat_input("Ask a question about your documents...")

if question:
    # Show and save the user's question
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve context and generate the answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                answer, passages = answer_question(
                    question,
                    client,
                    embedding_model,
                    collection,
                )

                st.markdown(answer)

                sources = []
                seen = set()

                for passage in passages:
                    key = (passage["source"], passage["page"])

                    if key not in seen:
                        seen.add(key)
                        sources.append({
                            "source": passage["source"],
                            "page": passage["page"],
                            "text": passage["text"],
                        })

                if sources:
                    with st.expander("View retrieved sources"):
                        for source in sources:
                            st.markdown(
                                f"- **{source['source']}** "
                                f"(Page {source['page']})"
                            )
                            st.caption(
                                source["text"][:500]
                                + (
                                    "..."
                                    if len(source["text"]) > 500
                                    else ""
                                )
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as error:
                error_message = (
                    "Sorry, something went wrong while answering "
                    "your question. Please try again."
                )
                st.error(error_message)
                st.caption(f"Technical details: {error}")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("About this assistant")
    st.write(
        "This application uses Retrieval-Augmented Generation (RAG) "
        "to find relevant passages in your documents before generating "
        "an answer with an LLM."
    )

    st.subheader("Indexed documents")
    st.markdown(
        "- NIST AI Risk Management Framework\n"
        "- AWS Data Warehousing"
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

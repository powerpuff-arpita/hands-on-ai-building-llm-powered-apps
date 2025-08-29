from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.vectorstores.base import VectorStore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

from utils import process_file, create_search_engine
from prompt import PROMPT, WELCOME_MESSAGE


load_dotenv()


# Page configuration
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None


def create_qa_chain(vector_store: VectorStore, api_key: str) -> Tuple[Any, Any]:
    """Create the QA chain with the vector store using LCEL.

    Args:
        vector_store: The vector store containing document embeddings
        api_key: OpenAI API key

    Returns:
        Tuple containing:
            - chain: The LCEL chain for question answering
            - retriever: The document retriever
    """
    llm = ChatOpenAI(
        model='gpt-4.1-mini',
        temperature=0,
        streaming=True,
        max_tokens=8192,
        api_key=api_key
    )

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs: List[Document]) -> str:
        """Format retrieved documents for the prompt.

        Args:
            docs: List of retrieved documents

        Returns:
            Formatted string containing document content and sources
        """
        formatted = []
        for doc in docs:
            content = doc.page_content
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"Content: {content}\nSource: {source}")
        return "\n\n".join(formatted)

    def get_question(inputs: Dict[str, Any]) -> str:
        return inputs["question"]

    def get_chat_history(inputs: Dict[str, Any]) -> List[Any]:
        return inputs["chat_history"]

    chain = (
        {
            "context": get_question | retriever | format_docs,
            "question": get_question,
            "chat_history": get_chat_history
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def format_answer_with_sources(response: str, retrieved_docs: List[Document]) -> Tuple[str, List[Dict[str, str]]]:
    """Format the answer with source information.

    Args:
        response: The LLM response containing the answer
        retrieved_docs: List of documents retrieved from the vector store

    Returns:
        Tuple containing:
            - answer: The formatted answer string
            - source_contents: List of source dictionaries with name and content
    """
    answer = response
    source_contents = []

    sources_text = ""
    if "SOURCES:" in answer:
        parts = answer.split("SOURCES:")
        if len(parts) > 1:
            sources_text = parts[1].strip()

    if sources_text and retrieved_docs:
        source_map = {}
        for doc in retrieved_docs:
            source_name = doc.metadata.get("source", "unknown")
            source_map[source_name] = doc.page_content

        found_sources = []
        for source in sources_text.split(","):
            source_name = source.strip().replace(".", "")
            if source_name in source_map:
                found_sources.append(source_name)
                source_contents.append({
                    "name": source_name,
                    "content": source_map[source_name]
                })

    return answer, source_contents


def get_chat_history_messages(messages: List[Dict[str, str]]) -> List[Any]:
    """Convert Streamlit messages to LangChain message format.

    Args:
        messages: List of Streamlit message dictionaries with 'role' and 'content' keys

    Returns:
        List of LangChain message objects (HumanMessage or AIMessage)
    """
    chat_history = []
    for msg in messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    return chat_history


def main() -> None:
    """Main Streamlit application function for PDF Q&A Assistant.

    Handles file upload, processing, and chat interface for asking questions
    about uploaded PDF documents using RAG (Retrieval Augmented Generation).
    """
    st.title("📚 PDF Q&A Assistant")
    st.markdown(WELCOME_MESSAGE)

    # Sidebar for file upload
    with st.sidebar:
        st.header("🔑 API Configuration")

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
            help="Enter your OpenAI API key to use the application"
        )

        if api_key:
            st.session_state.openai_api_key = api_key
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter your OpenAI API key to continue")

        st.divider()

        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF file to ask questions about its content",
            disabled=not st.session_state.openai_api_key
        )

        if uploaded_file is not None and st.session_state.openai_api_key:
            if st.session_state.processed_file != uploaded_file.name:
                with st.status("Processing PDF...", expanded=True) as status:
                    st.write("📄 Reading PDF content...")

                    try:
                        docs = process_file(
                            uploaded_file.getvalue(), "application/pdf")
                        st.write(f"✅ Extracted {len(docs)} text chunks")

                        st.write("🔍 Creating vector store...")
                        vector_store, _ = create_search_engine(
                            uploaded_file.getvalue(), "application/pdf", api_key=st.session_state.openai_api_key)

                        st.session_state.vector_store = vector_store
                        st.session_state.docs = docs
                        st.session_state.processed_file = uploaded_file.name

                        status.update(
                            label="✅ PDF processed successfully!", state="complete")

                    except Exception as e:
                        status.update(
                            label="❌ Error processing PDF", state="error")
                        st.error(f"Error: {str(e)}")
                        return

            st.success(f"📄 **{uploaded_file.name}** is ready for questions!")

    if st.session_state.vector_store is not None and st.session_state.openai_api_key:
        st.write("🧠 Setting up Q&A chain...")
        chain, retriever = create_qa_chain(
            st.session_state.vector_store, st.session_state.openai_api_key)

        # Store in session state
        st.session_state.chain = chain
        st.session_state.retriever = retriever

    # Chat interface
    if st.session_state.chain is not None:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.text(message["content"])

                # Display sources if available
                if "sources" in message and message["sources"]:
                    for source in message["sources"]:
                        with st.expander(f"📄 Source: {source['name']}"):
                            st.text(source["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the PDF..."):
            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.text(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chat_history = get_chat_history_messages(
                            st.session_state.messages)

                        # Get retrieved documents for source processing
                        retrieved_docs = st.session_state.retriever.invoke(
                            prompt)

                        # Invoke the LCEL chain
                        response = st.session_state.chain.invoke({
                            "question": prompt,
                            "chat_history": chat_history
                        })

                        answer, source_contents = format_answer_with_sources(
                            response, retrieved_docs
                        )

                        st.text(answer)

                        # Display sources
                        if source_contents:
                            for source in source_contents:
                                with st.expander(f"📄 Source: {source['name']}"):
                                    st.text(source["content"])

                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": source_contents
                        })

                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        import logging
                        logging.error(e, exc_info=True)
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

    else:
        if not st.session_state.openai_api_key:
            st.info(
                "🔑 Please enter your OpenAI API key in the sidebar to get started!")
        else:
            st.info("👆 Please upload a PDF file to get started!")


if __name__ == "__main__":
    main()

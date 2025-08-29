import chromadb
import tempfile
import os
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain.vectorstores.base import VectorStore
from langchain_openai import OpenAIEmbeddings


def process_file(file_data, file_type: str = None) -> list:
    """
    Process a PDF file and split it into documents.
    
    Args:
        file_data: Either a file path (str) or file bytes
        file_type: Optional file type, defaults to checking if PDF
    
    Returns:
        List of processed documents
    
    Raises:
        TypeError: If file is not a PDF
        ValueError: If PDF parsing fails
    """
    if file_type and file_type != "application/pdf":
        raise TypeError("Only PDF files are supported")
    
    # Handle both file path and file bytes
    if isinstance(file_data, bytes):
        # Create a temporary file for the PDF bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_data)
            tmp_file_path = tmp_file.name
        
        try:
            loader = PDFPlumberLoader(tmp_file_path)
            documents = loader.load()
        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)
    else:
        # Assume it's a file path
        loader = PDFPlumberLoader(file_data)
        documents = loader.load()
    
    # Clean up extracted text to fix common PDF extraction issues
    for doc in documents:
        # Fix common spacing issues from PDF extraction
        doc.page_content = doc.page_content.replace('\n', ' ')  # Replace newlines with spaces
        doc.page_content = ' '.join(doc.page_content.split())  # Normalize whitespace
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    if not docs:
        raise ValueError("PDF file parsing failed.")
    return docs


def create_search_engine(file_data, file_type: str = None, api_key: str = None) -> tuple[VectorStore, list]:
    """
    Create a vector store search engine from a PDF file.
    
    Args:
        file_data: Either a file path (str) or file bytes
        file_type: Optional file type for validation
        api_key: OpenAI API key for embeddings
    
    Returns:
        Tuple of (search_engine, docs) where:
        - search_engine: The Chroma vector store
        - docs: The processed documents
    """
    # Process the file
    docs = process_file(file_data, file_type)
    
    encoder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    
    # Initialize Chromadb client and settings, reset to ensure we get a clean
    # search engine
    client = chromadb.EphemeralClient()
    client_settings = Settings(
        allow_reset=True,
        anonymized_telemetry=False
    )
    search_engine = Chroma(
        client=client,
        client_settings=client_settings
    )
    search_engine._client.reset()
    
    search_engine = Chroma.from_documents(
        client=client,
        documents=docs,
        embedding=encoder,
        client_settings=client_settings
    )
    
    return search_engine, docs
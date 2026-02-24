import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()


def run_indexing():
    """
    Load PDF, split into chunks, embed and store in Pinecone.
    Returns the number of chunks indexed.
    """
    # Step 1: Document Loader
    path = "assets/AIPerupe_Academy_details.pdf"
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Step 2: Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents=documents)

    # Step 3: Embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Step 4: VectorStore - Store embeddings in Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "langchain-pinecone-asistente-de-ventas")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY must be set")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name,
    )
    return len(chunks)


if __name__ == "__main__":
    run_indexing()

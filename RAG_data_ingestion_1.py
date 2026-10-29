
import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

# Configuration
INDEX_NAME = "nmit-test-vectors"
DIMENSION = 1024


def create_pinecone_index():
    """Create a regular vector index if it doesn't already exist."""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if not pc.has_index(INDEX_NAME):
        print(f"Creating Pinecone index: {INDEX_NAME}")

        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print("Index creation requested.")
    else:
        print(f"Index already exists: {INDEX_NAME}")

    # Wait until the index is ready before ingesting.
    import time

    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

    print("Pinecone index is ready.")
    return pc


if __name__ == "__main__":
    print("Index name:", INDEX_NAME)
    print(
        "Pinecone API key loaded:",
        bool(os.environ.get("PINECONE_API_KEY")),
    )

    print("Ingesting...")

    # 1. Load the document
    loader = UnstructuredLoader(
        file_path=(
            "C:/Users/Sohan Kumar R N/Desktop/"
            "RAG_NMIT/mediumblog1.txt"
        ),
        chunking_strategy="basic",
        max_characters=1000000,
    )

    documents = loader.load()

    # 2. Split the document into chunks
    print("Splitting...")

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )

    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")

    if not texts:
        raise ValueError("No text chunks were created.")

    # 3. Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=DIMENSION,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    # 4. Create or verify the Pinecone vector index
    create_pinecone_index()

    # 5. Store the chunks and their embeddings
    print("Ingesting into Pinecone...")

    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    print("Finished ingestion successfully!")

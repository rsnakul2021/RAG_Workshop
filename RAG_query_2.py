
import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

# ============================================================
# CONFIGURATION
# Must match the index and embedding model used during ingestion
# ============================================================
INDEX_NAME = "nmit-test-vectors"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1024

# 1. Initialize embeddings
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    dimensions=EMBEDDING_DIMENSIONS,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)

# 2. Initialize the LLM
llm = ChatOpenAI(
    model="gpt-6.1-terra",
    api_key=os.environ["OPENAI_API_KEY"],
)

# 3. Connect to the same Pinecone index used during ingestion
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

# 4. Create the retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 5. Create the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context.

If the context does not contain enough information, say that you
don't have enough information to answer.

Context:
{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================
# IMPLEMENTATION 1: WITHOUT LCEL
# ============================================================
def retrieval_chain_without_lcel(query: str):
    """Retrieve documents and answer manually."""

    # Step 1: Retrieve relevant documents from Pinecone
    docs = retriever.invoke(query)

    # Step 2: Format documents into context
    context = format_docs(docs)

    # Step 3: Build the prompt
    messages = prompt_template.format_messages(
        context=context,
        question=query,
    )

    # Step 4: Send the prompt to the LLM
    response = llm.invoke(messages)

    # Step 5: Return the answer
    return response.content


# ============================================================
# IMPLEMENTATION 2: WITH LCEL
# ============================================================
def create_retrieval_chain_with_lcel():
    """Build a composable RAG chain using LCEL."""

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("Retrieving...")
    query = "What is Pinecone in machine learning?"

    # --------------------------------------------------------
    # OPTION 0: Raw LLM (no RAG)
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)

    result_raw = llm.invoke(
        [HumanMessage(content=query)]
    )

    print("\nAnswer:")
    print(result_raw.content)

    # --------------------------------------------------------
    # OPTION 1: RAG WITHOUT LCEL
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)

    result_without_lcel = retrieval_chain_without_lcel(query)

    print("\nAnswer:")
    print(result_without_lcel)

    # --------------------------------------------------------
    # OPTION 2: RAG WITH LCEL
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()

    result_with_lcel = chain_with_lcel.invoke(
        {"question": query}
    )

    print("\nAnswer:")
    print(result_with_lcel)

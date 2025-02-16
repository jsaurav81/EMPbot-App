# Import necessary modules and constants
from constants import index_name, embeddings
from langchain_pinecone import PineconeVectorStore


def get_retriever(search_filters):
    """
    Initializes the Pinecone vector store retriever based on the given search filters.

    Args:
        search_filters (list): A list containing search filter options.
        The first element (boolean) specifies if MMR (Maximal Marginal Relevance) is enabled.

    Returns:
        tuple: A tuple containing the Pinecone vector store (docsearch) and the retriever.
    """

    # Extract the MMR (Maximal Marginal Relevance) filter from the search filters list
    mmr_is_true = search_filters[0]

    # Load an existing Pinecone index using its name and embedding model
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,  # Name of the Pinecone index
        embedding=embeddings,  # Embedding model used for searching
    )

    # Check if MMR is enabled, and return the appropriate retriever
    if mmr_is_true:
        retriever = docsearch.as_retriever(search_type="mmr")  # Use MMR-based retrieval
    else:
        retriever = docsearch.as_retriever()  # Use standard retrieval

    # Return the vector store and the configured retriever
    return docsearch, retriever

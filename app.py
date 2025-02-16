# Import necessary libraries
import streamlit as st
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain import hub

from prompt_temp import prompt_answer
from reranking import get_reranked_answer
from retriever import get_retriever
from constants import langfuse_handler
from langchain_community.chat_models import ChatOpenAI


def retrieval_answers(query, filters):
    """
    Retrieves answers based on the query and search filters.

    Args:
        query (str): The user input query.
        filters (dict): The filter settings for retrieval, including flags
                        for reranking and process chain (PC).

    Returns:
        tuple: The answer and sources obtained from the retrieval process.
    """
    # Extract filter values
    *search_filters, pc_is_true, rr_is_true, twt = filters.values()

    # If reranking is enabled, get reranked answer
    if rr_is_true:
        return get_reranked_answer(query, twt)

    # Retrieve documents using the retriever
    index, retriever = get_retriever(search_filters)

    # Generate answer based on process chain flag
    if pc_is_true:
        answer = prompt_answer(query, retriever)
    else:
        answer = get_answer(query, retriever)

    # Get similar documents from the index
    sources = get_similar_docs(query, index)
    return answer, sources


def get_answer(query, retriever):
    """
    Generates an answer using the provided retriever and a pre-defined prompt.

    Args:
        query (str): The user input query.
        retriever: The retriever object for fetching context.

    Returns:
        str: The generated answer.
    """
    # Initialize the language model with specific configuration
    llm = ChatOpenAI(model="gpt-4o")

    # Pull the RAG (Retrieval-Augmented Generation) prompt from the hub
    prompt = hub.pull("rlm/rag-prompt")

    # Set up parallel processing for context retrieval and question input
    setup_and_retrieval = RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
    )

    # Create a chain combining setup, prompt, and language model
    rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()

    # Invoke the chain with the query
    return rag_chain.invoke(query, config={"callbacks": [langfuse_handler]})


def get_similar_docs(query, index, k=3, score=True):
    """
    Retrieves similar documents based on the query using the vector index.

    Args:
        query (str): The user input query.
        index: The Pinecone index for similarity search.
        k (int): The number of top similar documents to retrieve. Default is 3.
        score (bool): Flag to include similarity scoring. Default is True.

    Returns:
        str: Formatted answer containing information about similar documents.
    """
    # Initialize the language model with a specific configuration
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

    if score:
        # Perform similarity search with scoring
        similar_docs = index.similarity_search_with_score(query, k=k)

        # Define the template for formatting the output
        template = """
        Format the following sources with the given similarity score, 
        filename, context, and metadata in short. Restrict the context section 
        to the first two lines only. Add relevant emojis where necessary:

        {text}

        Write the headings in bold. For example:

        **Source 1**
        - **Similarity Score**: 0.92
        - **Filename**: research_paper_2024.pdf
        - **Metadata**: This paper was published in the Journal of Advanced Computing in 2024.
        - **Context**: This paper explores advanced machine learning techniques for predictive analytics in healthcare.
        """

        # Create a prompt template for the language model
        prompt = PromptTemplate(template=template, input_variables=["text"])

        # Create an LLM chain for generating the formatted response
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        # Generate the answer using the LLM chain
        answer = llm_chain.run(similar_docs, callbacks=[langfuse_handler])
    else:
        # Perform similarity search without scoring
        similar_docs = index.similarity_search(query, k=k)
        answer = f"Found {len(similar_docs)} similar documents."

    return answer

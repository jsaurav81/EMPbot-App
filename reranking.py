# Import necessary libraries
import re
from datetime import date

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_pinecone import PineconeVectorStore

from constants import index_name, embeddings


def get_reranked_contexts(data_objects, twt):
    """
    Reranks contexts based on a weighted scoring system using similarity scores
    and a normalized date-based value.

    Args:
        data_objects (list): A list of tuples containing documents and their similarity scores.
        twt (float): The weight parameter for the recency-based reranking.

    Returns:
        list: A list of top 5 document data objects with weighted scores.
    """

    # Initialize list to store document data with weighted scores
    document_data_objects = []

    # Iterate over data objects to calculate weighted scores
    for document, score in data_objects:
        # Extract numerical part from 'source' in metadata
        source_metadata = document.metadata.get("source", "")
        start_index = source_metadata.find("-") + 1  # Find position of '-'

        # Extract digits after '-' using regex
        numbers = re.findall(r"\d", source_metadata[start_index:])
        numerical_part = "".join(numbers[:8])  # Join the first 8 digits

        # Convert numerical part to an integer for normalization
        numerical_value = (
            int(numerical_part)
            if (numerical_part.isdigit() and int(numerical_part) > 0)
            else 0
        )

        # Normalize numerical value (considering a max value of 100 for normalization)
        max_date = int(date.today().strftime("%Y%m%d"))
        min_date = 20000101
        normalized_value = (numerical_value - min_date) / (max_date - min_date)

        # Generate weighted score using similarity score and normalized value
        weighted_score = (1 - twt) * score + twt * normalized_value

        # Create a dictionary of document data with calculated weighted score
        doc_data = {
            "page_content": document.page_content,
            "metadata": document.metadata,
            "score": score,
            "weighted_score": weighted_score,
        }
        document_data_objects.append(doc_data)

    # Sort documents by weighted score in descending order
    document_data_objects.sort(key=lambda x: x["weighted_score"], reverse=True)

    # Return top 5 results based on weighted score
    top_results = document_data_objects[:5]
    return top_results


def get_sources(context):
    """
    Formats the sources based on given context details with similarity score, weighted score,
    filename, and metadata.

    Args:
        context (list): The list of context details to format.

    Returns:
        str: Formatted string of sources with provided details.
    """

    template = """ 
    Format the following sources with the given similarity score, weighted score, filename, 
    context, and metadata in short. Restrict the context section to the first two lines only.
    Add relevant emojis where necessary:

    {context}

    Write the headings in bold. For example:

    **Source 1**
    - **Similarity Score**: 0.92
    - **Weighted Score**: 0.85
    - **Filename**: research_paper_2024.pdf
    - **Metadata**: This paper was published in the Journal of Advanced Computing in 2024.
    - **Context**: This paper explores advanced machine learning techniques for predictive analytics in healthcare.
    """

    # Create a prompt using the provided template
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o")
    output_parser = StrOutputParser()

    # Build the chain and generate the formatted source output
    chain = prompt | model | output_parser
    ans = chain.invoke({"context": context[:3]})

    return ans


def get_reranked_answer(query, twt):
    """
    Retrieves and reranks answers based on similarity scores and weighted recency-based scoring.

    Args:
        query (str): The user query for searching similar documents.
        twt (float): The weight parameter for the recency-based reranking.

    Returns:
        tuple: The generated answer and formatted sources.
    """

    # Load the existing Pinecone index for document search
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )

    # Perform similarity search on the query
    docs = docsearch.similarity_search_with_score(query)

    # Rerank contexts based on weighted scoring
    context = get_reranked_contexts(docs, twt)

    # Define a template for generating an answer from the context
    template = """ 
    Answer the question from the following context:
    {context}
    Question: {question}
    """

    # Create a prompt, model, and output parser for generating the answer
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o")
    output_parser = StrOutputParser()

    # Build the chain and generate the answer
    chain = prompt | model | output_parser
    ans = chain.invoke({"context": context, "question": query})

    # Retrieve formatted sources based on the context
    source = get_sources(context)

    return ans, source

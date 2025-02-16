# Import necessary modules
import os
import streamlit as st
from constants import index_name, embeddings
from app import retrieval_answers
from ingest import ingest_pdf, upsert_pdf

# Sidebar configuration for the Streamlit app
with st.sidebar:
    st.title("🖥️💬 EMP Chatbot")

    # Check if the OpenAI API key is provided in the environment variables
    if "OPENAI_API_KEY" in os.environ:
        st.success("API key already provided!", icon="✅")
    else:
        st.error("Please provide API key!", icon="⚠️")

    # File upload form for adding PDFs to the vector database
    with st.form("my_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Choose a PDF file", accept_multiple_files=True, type="pdf"
        )

        # Process PDF files if the form is submitted
        if st.form_submit_button("Add to VectorDB"):
            if uploaded_files:
                ingest_pdf(uploaded_files)  # Extract text from PDF
                upsert_pdf()  # Insert/Update data in the vector store

    # Options for retrieval settings
    st.subheader("Show answers with (choose one):")
    mmr_is_true = st.toggle("Maximal Marginal Relevance search")
    rr_is_true = st.toggle("Reranking by recency")
    pc_is_true = st.toggle("Get process chain")

    # Set default weight for recency reranking
    twt = 0
    if rr_is_true:
        twt = st.slider("Adjust the weight", 0.0, 1.0, value=0.2, step=0.1)

    # Define search filters based on user input
    filters = {
        "mmrIsTrue": mmr_is_true,
        "pcIsTrue": pc_is_true,
        "rrIsTrue": rr_is_true,
        "twt": twt,
    }

    # GitHub Codespaces link
    st.markdown(
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)]"
        "(https://github.com/andi677/ai-faps-saurav-jha/)"
    )

# Main title of the app
st.title("💬 EMP Chatbot")

# Initialize chat history if not already present in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Accept user input from the chat input box
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)


def generate_response(response):
    """
    Display the response from the assistant in real-time.

    Args:
        response (list): The response content to be displayed incrementally.
    """
    placeholder = st.empty()  # Create an empty placeholder for dynamic updates
    full_response = ""  # Initialize an empty string for the complete response

    # Incrementally display each part of the response
    for item in response:
        full_response += item
        placeholder.markdown(full_response)

    # Update the final response in the placeholder
    placeholder.markdown(full_response)

    # Add the full response to the session state messages
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve response and sources using the provided filters
            response, sources = retrieval_answers(prompt, filters)
            generate_response(response)  # Display the generated response

    # Display the sources of information used for the response
    st.success(sources, icon="✅")

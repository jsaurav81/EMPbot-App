# Electric Motor Production Chatbot (EMPbot)
**Developing a Question-Answering Chatbot on Electric Motor Production Using Large Language Models and Retrieval-Augmented Generation**

EMPbot is a specialized question-answering chatbot designed to support engineers, planners, and professionals in the field of electric motor manufacturing. Built using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), EMPbot provides accurate, contextually relevant information, simplifying access to specialized knowledge in electric motor production. This project, created as part of a project at FAPS, Friedrich-Alexander-Universität Erlangen-Nürnberg.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Installation](#installation)
---

## Overview
EMPbot allows users to ask questions related to electric motor production and receive detailed, relevant answers. By combining LLMs with document retrieval capabilities, EMPbot can:
- Provide answers in natural language from a specialized knowledge base.
- Offer step-by-step explanations through a "Process Chain Mode" to outline complex production steps.
- Integrate new documents into the knowledge base for a continuously updated information source.
- Adjust relevance based on document recency for up-to-date results.

---

## Project Structure

### Root Directory
#### Files
- **stream_lit.py**: The main frontend code, using Streamlit for the user interface.
- **app.py**: Facilitates interactions with the LLM to generate answers based on user queries.
- **requirements.txt**: Contains all required Python packages for the project.

- **retriever.py**: Initializes the Pinecone vector store retriever and applies search filters to find relevant documents.
- **reranking.py**: Implements recency-based reranking, prioritizing newer and contextually appropriate information.
- **prompt_temp.py**: Contains the code for "Process Chain Mode," which structures responses in a step-by-step format.
- **ingest.py**: Manages document ingestion, allowing users to upload and permanently add new documents to the knowledge base.
#### Folders
- **pdf_database**: Contains all documents in the knowledge base.
- **xplore_api**: Contains code script to fetch DOIs from IEEE api.
- **uploaded_pdfs**: Stores uploaded pdf files by one-click upsert feature.
#### Configuration
- **constants.py**: Stores essential project constants, including the OpenAI API key, and settings for pinecone.

---

## Usage

### 1. Querying the Chatbot
   - Enter questions in natural language about electric motor production.
   - EMPbot retrieves and generates accurate answers from its knowledge base.

### 2. Process Chain Mode
   - Activate "Process Chain Mode" to get step-by-step answers about production processes.

### 3. Document Upload and Upsert
   - Use the drag-and-drop feature to upload new documents.
   - Click "One-click Upsert" to store new documents permanently in the knowledge base.

### 4. Recency and Relevance Adjustments
   - Adjust settings to control the recency and relevance of search results for context-aware answers.


## Installation
### 1. Creating a Virtual Environment: You can create a virtual environment using the following command:
   ```bash
   python -m venv myenv
   ```
   
### 2. Install Requirements: Once you have the requirements.txt file, you can install the dependencies listed in it using pip:
   ```bash
   pip install -r requirements.txt
   ```
### 3. Insert your openAI api key in constants,py file
   
### 4. Start the application using
   ```bash
   streamlit run stream_lit.py
   ```

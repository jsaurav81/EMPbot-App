# Import necessary libraries
import time
import os
import shutil
import streamlit as st

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from PyPDF2 import PdfReader
from constants import source_dir, index_name, embeddings, destination_dir


def save_file(uploaded_files):
    """
    Saves uploaded PDF files to the source directory.

    Args:
        uploaded_files (list): List of uploaded PDF files.
    """
    # Remove and recreate the source directory to ensure it's empty
    shutil.rmtree(source_dir, ignore_errors=True)
    os.makedirs(source_dir, exist_ok=True)

    # Save each uploaded file to the source directory
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Construct the full file path
            file_path = os.path.join(source_dir, uploaded_file.name)

            # Write the content of the uploaded file to the destination path
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())


def to_text():
    """
    Converts PDF files in the source directory to text chunks.

    Returns:
        list: List of text chunks from PDF documents.
    """
    # Load PDF files from the source directory
    loader = DirectoryLoader(source_dir, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split documents into manageable text chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts


def count_pdf_files(directory):
    """
    Counts the number of PDF files in the specified directory.

    Args:
        directory (str): The directory path to count PDF files.

    Returns:
        int: Number of PDF files in the directory.
    """
    return len(
        [
            name
            for name in os.listdir(directory)
            if name.lower().endswith(".pdf")
            and os.path.isfile(os.path.join(directory, name))
        ]
    )


def get_new_name(filename, folder_path):
    """
    Extracts the creation date from PDF metadata and generates a new file name.

    Args:
        filename (str): The name of the PDF file.
        folder_path (str): The folder path where the PDF file is located.

    Returns:
        str: New name based on the creation date extracted from metadata.
    """
    metadata = PdfReader(os.path.join(folder_path, filename)).metadata
    creation_date_value = metadata.get("/CreationDate", "")

    # Extract the first 8 digits from the creation date
    first_8_digits = "".join(filter(str.isdigit, creation_date_value))[:8]
    return first_8_digits


def rename_files():
    """
    Renames PDF files in the source directory based on a specific naming convention.
    """
    index = count_pdf_files("pdf_database")
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(".pdf"):
            index += 1
            new_name = get_new_name(filename, source_dir)
            new_file_name = f"{index}-{new_name}.pdf"

            # Construct old and new file paths
            old_path = os.path.join(source_dir, filename)
            new_path = os.path.join(source_dir, new_file_name)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} to {new_file_name}\n")


def add_to_vectordb(texts):
    """
    Adds text chunks to the Pinecone vector store.

    Args:
        texts (list): List of text chunks to be added to the vector store.
    """
    PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)


def move_files(source_dir, destination_dir):
    """
    Moves PDF files from the source directory to the destination directory.

    Args:
        source_dir (str): The source directory path.
        destination_dir (str): The destination directory path.
    """
    for file_name in os.listdir(source_dir):
        if file_name.lower().endswith(".pdf"):
            source_file = os.path.join(source_dir, file_name)
            destination_file = os.path.join(destination_dir, file_name)

            # Move the file if it exists
            if os.path.isfile(source_file):
                shutil.move(source_file, destination_file)
                print(f"Moved: {file_name}")


def display(message):
    """
    Displays a success message on the Streamlit app and waits for 3 seconds.

    Args:
        message (str): The message to be displayed.
    """
    msg = st.success(message)
    time.sleep(3)
    msg.empty()


def ingest_pdf(uploaded_files):
    """
    Ingests uploaded PDF files by saving, renaming, and displaying a success message.

    Args:
        uploaded_files (list): List of uploaded PDF files.

    Returns:
        bool: True if the operation is successful.
    """
    try:
        save_file(uploaded_files)
    except Exception as e:
        print(f"Failed to save file: {e}")
        return False

    rename_files()
    display("Uploaded files saved locally")
    return True


def upsert_pdf():
    """
    Processes PDF files to text, adds them to the vector store,
    and moves files to the destination directory.
    """
    try:
        texts = to_text()
    except Exception as e:
        print(f"Failed to convert PDF to text: {e}")
        return

    try:
        add_to_vectordb(texts)
    except Exception as e:
        print(f"Failed to add texts to vector database: {e}")
        return

    try:
        move_files(source_dir, destination_dir)
    except Exception as e:
        print(f"Failed to move files: {e}")
        return

    display("Upsert PDF completed successfully")

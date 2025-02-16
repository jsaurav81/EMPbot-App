# Import necessary modules
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# Set up environment variables for OpenAI and Pinecone API keys
os.environ["OPENAI_API_KEY"] = ""  # Add your OpenAI API key here
PINECONE_ENV = "gcp-starter"  # Pinecone environment configuration

# Define Pinecone index and API key
index_name = "vectordb"
PINECONE_API_KEY = "d9b2f96e-d42e-47db-932d-dd5614ded0a6"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY  # Set Pinecone API key

# Initialize OpenAI embeddings and language model
embeddings = OpenAIEmbeddings()  # Embeddings for vector store
llm = ChatOpenAI(model="gpt-4o")  # Language model configuration

# Define source and destination directories for PDF files
source_dir = "uploaded_pdfs"
destination_dir = "pdf_database"

# Initialize Langfuse for logging and monitoring
langfuse = Langfuse(
    secret_key="sk-lf-3f878995-696a-474f-a270-de0e8ea5d4ed",  # Secret key for Langfuse
    public_key="pk-lf-9a493a30-8dd2-4856-ab80-f6ba6d765c2a",  # Public key for Langfuse
    host="https://cloud.langfuse.com",  # Langfuse cloud host URL
)

# Create a callback handler for Langfuse integration
langfuse_handler = CallbackHandler(
    secret_key="sk-lf-3f878995-696a-474f-a270-de0e8ea5d4ed",  # Secret key for Langfuse
    public_key="pk-lf-9a493a30-8dd2-4856-ab80-f6ba6d765c2a",  # Public key for Langfuse
    host="https://cloud.langfuse.com",  # Langfuse cloud host URL
)

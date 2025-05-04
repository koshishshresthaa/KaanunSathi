from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
import os


load_dotenv()

# Get the API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")


# Define the LLM model
llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=groq_api_key 
)

#Define the embedding model 

#(CHROMA DB BY DEFAULT USES THIS MODEL)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

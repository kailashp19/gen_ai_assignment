from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

class LLModels:
    def __init__(self):
        self.llm = ChatGroq(model="llama3-70b-8192", api_key="XXXX")
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
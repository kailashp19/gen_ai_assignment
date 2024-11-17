from pptx import Presentation  # For handling .pptx files
from PIL import Image  # Python Imaging Library for handling images
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
from docx.opc.exceptions import PackageNotFoundError
from nltk import sent_tokenize
import fitz  # PyMuPDF for PDFs
import pytesseract  # For image to text conversion
import google.generativeai as genai
import os
import re
import docx  # For handling .docx files
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import nltk

class RAGPRocessing():
    """
    A class which takes file(s) either in .pdf, .docx, .pptx, or image files in jpg or png format which contains the
    information on technical error codes. Once the documents from above format are provided, it then converts the above documents
    into text files. These text files are first broken down into chunks, tokenized it, and then later on embeddings are created out
    of it. After the embedding are created, these embeddings are stored into Vector Database (Qdrant). A user query is also passed 
    to search for the result from the vector database. the searched results are then passed to system prompt and LLM to generate the
    response of the uer query for the technical issues.
    """
    def __init__(self, file_dir: str, text_dir: str, client: str, model: str, collection_name: str, user_query: str, chunk_size: int, llm: str) -> None:
        self.file_dir = file_dir
        self.text_dir = text_dir
        self.client = client
        self.model = model
        self.collection_name = collection_name
        self.user_query = user_query
        self.chunk_size = chunk_size
        self.llm = llm
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text, removing extra spaces but preserving newlines.

        Parameters:
        text (str): A string to clean and normalize it.

        Return:
        text (str): A string which is cleaned and normalized

        """
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text

    def pdf_to_text(self, pdf_path: str, text_path: str) -> None:
        """
        Convert PDF to text, clean it, and save to a file.
        
        Parameters:
        pdf_path (str): A path to pdf file in string format.
        text_path (str): A path to save it as a text file in string format.

        Return:
        None
        """
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += self.clean_text(page_text) + "\n"

        with open(text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

    def image_to_text(self, image_path: str, text_path: str) -> None:
        """
        Convert image to text using pytesseract, clean it, and save to a file.

        Parameters:
        image_path (str): A path to image file in string format.
        text_path (str): A path to save it as a text file in string format.

        Return:
        None

        """
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        text = self.clean_text(text)

        with open(text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

    def docx_to_text(self, docx_path: str, text_path: str) -> None:
        """
        Convert DOCX to text, clean it, and save to a file.
        
        Parameters:
        docx_path (str): A path to docx file in string format.
        text_path (str): A path to save it as a text file in string format.

        Return:
        None
        """
        try:
            doc = docx.Document(docx_path)
            text = ""
            for para in doc.paragraphs:
                text += self.clean_text(para.text) + "\n"
            
            with open(text_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
        except PackageNotFoundError as e:
            print(f"PackageNotFoundError: {e} - The file {docx_path} is not valid or is corrupted.")
        except Exception as e:
            print(f"Error processing file {docx_path}: {e}")

    def pptx_to_text(self, pptx_path: str, text_path: str) -> None:
        """
        Convert PPTX to text, clean it, and save to a file.
        
        Parameters:
        pptx_path (str): A path to ppt file in string format.
        text_path (str): A path to save it as a text file in string format.

        Return:
        None
        """
        presentation = Presentation(pptx_path)
        text = ""

        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += self.clean_text(shape.text) + "\n"

        with open(text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

    def vector_db_creation(self) -> str:
        """
        Creates the embeddings, store the embeddings into a vector dataabse, search for the user query in the vector database and finally
        passed the search results to an LLM model for more contextualized results.

        Parameters:
        text_dir (str): A directory to text file in string format.
        client (object): A connection string to the Qdrant Client.
        model (object): An embedding model
        collection_name (str): A name of the collection where vector embeddings are stored.
        user_query (str): A user provided query.
        chunk_size (int): A size of the chunk, usually 128 characters
        llm (object): A Gemini 1.5 flash LLM Model

        Returns:
        extracted_text (str): A response from an LLM Model in the form of natural language.
        """
        with open(f"{self.text_dir}/Error Codes.txt", 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split the document into chunks (sentences in this case)
        sentences = sent_tokenize(text)
        chunks = [' '.join(sentences[i:i + self.chunk_size]) for i in range(0, len(sentences), self.chunk_size)]

        # Generate embeddings
        embeddings = self.model.encode(chunks, show_progress_bar=True)

        # Convert to a NumPy array
        embeddings = np.array(embeddings)

        # Save embeddings to a file
        np.save('sentence_embeddings.npy', embeddings)

        # Create a collection to store the embeddings
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=embeddings.shape[1], distance='Cosine')
        )

        # Add points to the collection
        points = [
            PointStruct(id=i, vector=embedding, payload={"document": doc})
            for i, (embedding, doc) in enumerate(zip(embeddings, chunks))
        ]

        client.upsert(collection_name=self.collection_name, points=points)

        # Encode the query into an embedding
        query_embedding = self.model.encode([user_query])[0]  # Flatten the query embedding

        # Perform the search
        search_results = client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=1  # Number of nearest neighbors to return
        )

        # Display the search results
        for idx, result in enumerate(search_results, start=1):
            text_chunk = result.payload.get("document", "No text found")
            score = result.score
            print(f"accuracy score: {score}")

        # Retrieve the top result text chunks for context
        context_texts = search_results

        context_prompt = f"""
        You are provided with a document in .txt format and your task is to provide a solution to a user coming with problem
        for tehcnical issue from a document {context_texts}.

        For Example:
        USER QUERY: I am getting an error as 1333
        OUTPUT: 
        Here is a recommended solution
        Verify the Joey is active on the customer account   
        Perform a front panel reset on the Joey and Hopper
        Ensure the Joey is linked to the Hopper in SETTINGS > WHOLE HOME
        Inspect the signal path starting at the Joey and working back toward the Hub/Node:

        User Query
        {self.user_query}\n\n"""

        # Call Gemini API to generate the response
        gemini_response = self.llm.generate_content(context_prompt)
        if gemini_response.candidates:
            extracted_text = gemini_response.text.strip('```json').strip('```').strip()
            return extracted_text
        else:
            extracted_text = ''
            return extracted_text

def main(file_dir, text_dir, client, model, collection_name, user_query, chunk_size, llm):
    pdoc = RAGPRocessing(file_dir, text_dir, client, model, collection_name, user_query, chunk_size, llm)
    # Process each file in the directory
    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        text_file_name = os.path.splitext(file_name)[0] + '.txt'
        text_path = os.path.join(text_dir, text_file_name)
        if file_name.lower().endswith('.pdf'):
            pdoc.pdf_to_text(file_path, text_path)
        elif file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            pdoc.image_to_text(file_path, text_path)
        elif file_name.lower().endswith('.docx'):
            pdoc.docx_to_text(file_path, text_path)
        elif file_name.lower().endswith('.pptx'):
            pdoc.pptx_to_text(file_path, text_path)
        else:
            print(f"Unsupported file format for {file_path}")
    result = pdoc.vector_db_creation()
    return result

if __name__=="__main__":
    # Directories
    file_dir = "Capstone data sets"  # Directory containing files of various formats
    text_dir = "Converted text files"  # Directory to save text files
    os.makedirs(text_dir, exist_ok=True)
    # Connect to the Qdrant service
    client = QdrantClient("http://localhost:6333")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    # defining the name of the collection
    collection_name = "Technical_Support_Agent"
    # Configure Google Gemini API
    genai.configure(api_key='AIzaSyAt8gpOAHgwzOGOhpJATz88vxMeeM1q2Lg')
    llm = genai.GenerativeModel("gemini-1.5-flash")
    user_query = "What does error code 001 means?"
    chunk_size = 128
    main(file_dir, text_dir, client, model, collection_name, user_query, chunk_size, llm)
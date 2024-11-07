import fitz  # PyMuPDF for PDFs
import pytesseract  # For image to text conversion
from PIL import Image  # Python Imaging Library for handling images
import os
import re
import docx  # For handling .docx files
from pptx import Presentation  # For handling .pptx files
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import warnings
from embedding_using_bert import Embedding, TextDatasets
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import VectorParams, PointStruct, CreateCollection

# Directories
file_dir = "Capstone data sets/Capstone data sets"  # Directory containing files of various formats
text_dir = "Capstone data sets/Converted text files"  # Directory to save text files
# defining the name of the collection
collection_name = "Technical_Support_Agent"
# Load model for generating embeddings
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
os.makedirs(text_dir, exist_ok=True)

def clean_text(text):
    """Clean and normalize extracted text, removing extra spaces but preserving newlines."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

def pdf_to_text(pdf_path, text_path):
    """Convert PDF to text, clean it, and save to a file."""
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        text += clean_text(page_text) + "\n"

    with open(text_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

def image_to_text(image_path, text_path):
    """Convert image to text using pytesseract, clean it, and save to a file."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    text = clean_text(text)

    with open(text_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

def docx_to_text(docx_path, text_path):
    """Convert DOCX to text, clean it, and save to a file."""
    doc = docx.Document(docx_path)
    text = ""

    for para in doc.paragraphs:
        text += clean_text(para.text) + "\n"

    with open(text_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

def pptx_to_text(pptx_path, text_path):
    """Convert PPTX to text, clean it, and save to a file."""
    presentation = Presentation(pptx_path)
    text = ""

    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += clean_text(shape.text) + "\n"

    with open(text_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

def vector_db_creation(text_path, collection_name, model):
        
    # Load model for generating embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load text from a file
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # getting the phrases as a list
    text_list = text.split()

    # Define chunk size (number of words per chunk)
    chunk_size = 100  # Example: 100 words per chunk

    # Split words into chunks and join each chunk into a single string
    chunks = [' '.join(text_list[i:i + chunk_size]) for i in range(0, len(text_list), chunk_size)]

    # Now `chunks` contains the words from the file split into chunks
    print(f"Number of chunks: {len(chunks)}")
    print(chunks)
    print(chunks[:2])  # Print the first two chunks as an example

    # Generate embeddings for each chunk
    doc_embeddings = np.load('embeddings_from_bert_transformers.npy')
    print("Embeddings shape: ", doc_embeddings.shape[0])

    # passing the query
    query = "Who are reserved athletes"

    # tokenize the text from the text file
    tokens = tokenizer.encode(query, add_special_tokens=True)
    max_length = min(len(tokens), 512)
    print(max_length)
    tokens = tokens[:max_length]
    print("Tokens: ", tokens)

    fde = None

    # converting 2D array to 1D
    fde = doc_embeddings.flatten()

    # Create a collection to store the embeddings
    # client.recreate_collection(
    #     collection_name=collection_name,
    #     vectors_config=VectorParams(size=fde.shape[0], distance='Cosine')
    # )

    # # Add points to the collection
    # points = [
    #     PointStruct(id=i, vector=embedding, payload={"document": doc})
    #     for i, (embedding, doc) in enumerate(zip(fde, chunks))
    # ]

def main():
    # Process each file in the directory
    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        text_file_name = os.path.splitext(file_name)[0] + '.txt'
        text_path = os.path.join(text_dir, text_file_name)

        if file_name.lower().endswith('.pdf'):
            print(f"Converting {file_path} to {text_path}")
            pdf_to_text(file_path, text_path)
        elif file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Converting {file_path} to {text_path}")
            image_to_text(file_path, text_path)
        elif file_name.lower().endswith('.docx'):
            print(f"Converting {file_path} to {text_path}")
            docx_to_text(file_path, text_path)
        elif file_name.lower().endswith('.pptx'):
            print(f"Converting {file_path} to {text_path}")
            pptx_to_text(file_path, text_path)
        else:
            print(f"Unsupported file format for {file_path}")

    print("All files have been converted to text files with preserved newlines and no extra spaces.")
    vector_db_creation(text_path, collection_name, model)
if __name__=="__main__":
    main()
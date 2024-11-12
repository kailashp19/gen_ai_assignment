import fitz  # PyMuPDF for PDFs
import pytesseract  # For image to text conversion
from PIL import Image  # Python Imaging Library for handling images
import google.generativeai as genai
import os
import re
import docx  # For handling .docx files
from pptx import Presentation  # For handling .pptx files
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct, CreateCollection
from docx.opc.exceptions import PackageNotFoundError
import nltk
from nltk import sent_tokenize

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
    try:
        doc = docx.Document(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += clean_text(para.text) + "\n"
        
        with open(text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)
    except PackageNotFoundError as e:
        print(f"PackageNotFoundError: {e} - The file {docx_path} is not valid or is corrupted.")
    except Exception as e:
        print(f"Error processing file {docx_path}: {e}")

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

def vector_db_creation(text_dir, collection_name):    

    with open(f"{text_dir}/Error Codes.txt", 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Split the document into chunks (sentences in this case)
    sentences = sent_tokenize(text)
    chunk_size = 128  # Adjust chunk size as needed
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

    # Generate embeddings
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Convert to a NumPy array
    embeddings = np.array(embeddings)

    # Save embeddings to a file
    np.save('sentence_embeddings.npy', embeddings)

    print(embeddings.shape[1])

    # Create a collection to store the embeddings
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance='Cosine')
    )

    # Add points to the collection
    points = [
        PointStruct(id=i, vector=embedding, payload={"document": doc})
        for i, (embedding, doc) in enumerate(zip(embeddings, chunks))
    ]

    client.upsert(collection_name=collection_name, points=points)

    # Encode the query into an embedding
    query = "Could you please provide me what is the solution for error code 021?"
    query_embedding = model.encode([query])[0]  # Flatten the query embedding
    print(query_embedding)

    # Perform the search
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=1  # Number of nearest neighbors to return
    )

    # Display the search results
    print("\nSearch Results:")
    for idx, result in enumerate(search_results, start=1):
        text_chunk = result.payload.get("document", "No text found")
        score = result.score
        print(f"{idx}. Response: {text_chunk}\n   Score: {score:.4f}\n")

    # Retrieve the top result text chunks for context
    context_texts = search_results

    context_prompt = f"""
    You are provided with a document in .txt format and your task is to provide a solution to a user coming with problem
    for tehcnical issue from a document {search_results}.
    For Example:
    User query: I am getting an error as 1333
    output: 
    Here is a recommended solution
    Verify the Joey is active on the customer account   
    Perform a front panel reset on the Joey and Hopper
    Ensure the Joey is linked to the Hopper in SETTINGS > WHOLE HOME
    Inspect the signal path starting at the Joey and working back toward the Hub/Node:
    {query}\n\n"""

    for i, text in enumerate(context_texts, start=1):
        context_prompt += f"Context {i}: {text}\n\n"""

    context_prompt += f"Question: {query}\nAnswer:"

    # Call Gemini API to generate the response
    gemini_response = llm.generate_content(context_prompt)

    # Display the generated response
    print("\nGemini Response:")
    # print(gemini_response["text"])

    if gemini_response.candidates:
        extracted_text = gemini_response.text.strip('```json').strip('```').strip()
        print(f"Extracted text: {extracted_text}")  # Debugging purpose
    else:
        extracted_text = ''

def main():
    # Process each file in the directory
    for file_name in os.listdir(file_dir):
        print(file_dir)
        file_path = os.path.join(file_dir, file_name)
        text_file_name = os.path.splitext(file_name)[0] + '.txt'
        text_path = os.path.join(text_dir, text_file_name)
        print(file_path)
        print(file_name)
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
    vector_db_creation(text_dir, collection_name)

if __name__=="__main__":
    nltk.download('punkt_tab')

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

    main()
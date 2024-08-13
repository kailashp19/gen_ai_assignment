import os
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex

# Directory containing PDF files
pdf_dir = "PDFs"

# Directory to save text files
text_dir = "text_files"
os.makedirs(text_dir, exist_ok=True)

# Function to load and index PDF files
def process_pdfs(pdf_dir, text_dir):
    """Convert PDF to text and index the content using LlamaIndex."""
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text_file_name = os.path.splitext(pdf_file)[0] + '.txt'
            text_path = os.path.join(text_dir, text_file_name)

            print(f"Processing {pdf_path}")

            # Use LlamaIndex to read and index the PDF file
            document_reader = SimpleDirectoryReader(input_dir=pdf_dir)
            documents = document_reader.load_data()
            print(documents)

            # Save the text to a file
            with open(text_path, 'w', encoding='utf-8') as text_file:
                for document in documents:
                    text_file.write(document.get_text())
            
            print(f"Converted {pdf_path} to {text_path}")

    print("All PDFs have been converted to text files.")

# Call the function to process PDFs
process_pdfs(pdf_dir, text_dir)

# Optional: Create an index of the documents
# index = GPTVectorStoreIndex.from_documents(documents)

# Save the index for later use
# index.save_to_disk("index.json")

print("Index has been created and saved.")

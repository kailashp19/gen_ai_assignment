import fitz  # PyMuPDF
import os

# Directory containing PDF files
pdf_dir = "PDFs"

# Directory to save text files
text_dir = "text_files"
os.makedirs(text_dir, exist_ok=True)

def pdf_to_text(pdf_path, text_path):
    """Convert PDF to text and save to a file."""
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()

    with open(text_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

# Process each PDF in the directory
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.lower().endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        text_file_name = os.path.splitext(pdf_file)[0] + '.txt'
        text_path = os.path.join(text_dir, text_file_name)

        print(f"Converting {pdf_path} to {text_path}")
        pdf_to_text(pdf_path, text_path)

print("All PDFs have been converted to text files.")
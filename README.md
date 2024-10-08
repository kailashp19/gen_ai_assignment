# 2024 Paris Olympics Gen AI Assignment - Team 6
A repository providing an overview of the 2024 Paris Olympics Gen AI question-answering solution.

## PDF Question-Answering System for Paris 2024 Athletics

### Problem Statement

Develop a Question-Answering solution using GenAI that can answer questions based on a multi-page searchable PDF (Paris2024Athletics.pdf). The system should provide answers along with references from the source document.

### Project Overview

This project implements a Question-Answering system for the Paris 2024 Athletics PDF document using GenAI techniques. The system can answer questions about the content of the PDF, providing relevant information and references to the source document.

### Expected Outcome

1. Working code in a Jupyter notebook or Python development interface.
2. The solution will be evaluated based on relevant questions, focusing on the approach, choice of tools and architecture, and underlying concepts.
3. The solution is implemented using:
   - Available live API endpoints online.
   - A downloaded model for local use.

### Project Structure

```plaintext
GenAIProject/
├── .venv/                      # Virtual environment (library root)
├── gen_ai_assignment/          # Main project directory
├── images/                     # Directory for image files (if any)
├── PDFs/                       # Directory containing the source PDF
│   └── Paris2024-QS-Athletics.pdf
├── text_files/                 # Directory for extracted text files
├── convert_pdf_to_text.py      # Script to convert PDF to text
├── dynamic_vector_search.py    # Script for dynamic vector search
├── generate_text_embeddings.py # Script to generate text embeddings
├── paris_olympics_Chat_PE.ipynb # Jupyter notebook for the main QA system
├── README.md                   # This file
├── sentence_embeddings.npy     # NumPy file containing sentence embeddings
└── sentences.txt               # Text file containing extracted sentences
```

### Setup and Installation

1. Clone the repository
2. Create and activate a virtual environment: python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
3. Install dependencies: pip install -r requirements.txt
4. Install docker
5. Run the docker command to activate the qdrant vector database: docker run -p 6333:6333 qdrant/qdrant

## Usage

1. Convert PDF to text: 
python convert_pdf_to_text.py
2. Generate embeddings:
python generate_text_embeddings.py 
3. Create vector database:
python dynamic_vector_search
4. Run the question-answering system:
- Open `paris_olympics_Chat_PE.ipynb` in Jupyter Notebook or JupyterLab
- Follow the instructions in the notebook to interact with the QA system

## Components

- `convert_pdf_to_text.py`: Extracts text from the PDF document
- `generate_text_embeddings.py`: Create embeddings from the text file by breaking the texts into sentences
- `dynamic_vector_search.py`: Builds Vector database and Implements dynamic search functionality
- `paris_olympics_Chat_PE.ipynb`: Main interface for the QA system

## Model and Techniques

The project uses the all-MiniLM-L6-v2 model from sentence-transformers to generate embeddings for sentences. These embeddings are used for vector search to find relevant documents or sentences in response to queries. The system supports both API-based and local models for question answering.

## Architecture

The system architecture includes:

PDF Processing: Extracts and structures text from PDFs.
Sentence Embedding: Generates embeddings for each sentence.
Document Retrieval: Uses vector search to find relevant sentences based on a query.
Answer Generation: Retrieves or generates answers from the most relevant text.

## Evaluation

The system's performance is evaluated using test questions, precision and recall metrics, and manual review to ensure the quality of retrieved answers.

## Submission Guidelines

1. Code files: All Python scripts and Jupyter notebooks are included in the project structure.
2. Documentation: This README file serves as the main documentation, explaining the solution and its components.

## Presentation

A 15-minute demo session (10 minutes presentation, 5 minutes Q&A) will be conducted to showcase the solution. The presentation will cover:
- Overview of the architecture and approach
- Demonstration of both API-based and local model solutions
- Key features and capabilities of the system
- Challenges faced and how they were addressed
- Potential improvements and future work

## Contributors

[Your Name]

## License

[License Reference Number]

# Level 2
## Dataframe creation from invoices
An updated repository for processing the invoices and converting it into a dataframe.

## Fine Tuning an LLM Model with prompting technique to convert the invoices to a dataframe.

### Problem Statement
The goal of this assignment is to leverage the capabilities of Large Language Models (LLMs) to develop data structurizer tool which is capable of extracting specific, relevant information from a variety of unstructured documents, including invoices, receipts, bills, medical prescriptions, and general documents.
 
The extracted data should be organized into a structured DataFrame format, providing a clean and usable dataset for further analysis or applications.
 
Expected Output
Working Notebook in any of the platform using python
The final output should be a well-structured DataFrame containing the extracted data.
Each DataFrame should represent a single document, and each column should correspond to a specific data element (e.g., invoice number, date, total amount, item descriptions).

Additional Considerations
Document Format: Consider the variety of document formats you'll be dealing with (e.g., PDF, Word, images) and choose appropriate tools or techniques for each.
Prompt Engineering: Craft effective prompts to guide the LLM in extracting the desired information.
Model Selection: Choose an LLM that is suitable for the task and the available resources.

Evaluation Criteria:
Data Quality and Accuracy:
Completeness: Were all relevant data elements extracted from the documents?
Accuracy: Is the extracted data correct and consistent with the original documents?
Consistency: Are there any inconsistencies or contradictions within the extracted data?

Data Structurization:
Data Organization: Is the data organized in a clear and understandable manner within the DataFrame?
Data Quality: Are there any missing values, inconsistencies, or errors in the structured data?

LLM Usage:
Prompt Engineering: Were the prompts used to guide the LLM effective in extracting the desired information?
Model Selection: Was the chosen LLM appropriate for the task?
Fine-Tuning: Was the LLM effectively fine-tuned on a relevant dataset?

Code Quality:
Readability: Is the code well-structured, commented, and easy to understand?
Efficiency: Is the code efficient in terms of computational resources and execution time?

### Project Overview

This project implements a dataframe conversion from invoices coming in varied formats like PDF, Word, Images and PPT.

### Expected Outcome

1. Working python code or Python development interface.
2. The solution is implemented using:
   - Open Sourced LLM like Gemini Flash 1.5 API
   - Prompting technique

### Setup and Installation

1. Clone the repository
2. Create and activate a virtual environment: python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
3. Install dependencies: pip install -r requirements.txt
4. Install necessary libraries

## Usage

1. Convert PDF, Images, DOCX and PPTX files to text files and making it in consistent format: 
python convert_pdf_to_text.py
2. Writing the detailed prompt to instruct Gemini 1.5 Flash API LLM to convert it to a jsoormat and further converting the json file
to a dataframe:
python create_outputs.py

## Components

- `convert_pdf_to_text.py`: Extracts text from the PDF document
- `generate_text_embeddings.py`: Create embeddings from the text file by breaking the texts into sentences
- `dynamic_vector_search.py`: Builds Vector database and Implements dynamic search functionality
- `paris_olympics_Chat_PE.ipynb`: Main interface for the QA system

## Model and Techniques

The project uses the gemini-1.5-flash model from Google Gemini. The system first converts any number of invoice documents in any format
like docx, pptx, pdf or images to text files and written a generic prompt for invoice handling to convert it to a valid json format.
Then all the json objects are converted to respective dataframes and finally saved as a csv file with respective file name.

## Evaluation

The system's performance is evaluated using test questions, precision and recall metrics, and manual review to ensure the quality of retrieved answers.

## Submission Guidelines

1. Code files: All Python scripts and Jupyter notebooks are included in the project structure.
2. Documentation: This README file serves as the main documentation, explaining the solution and its components.

## Presentation

A 15-minute demo session (10 minutes presentation, 5 minutes Q&A) will be conducted to showcase the solution. The presentation will cover:
- Overview of the architecture and approach
- Demonstration of both API-based and local model solutions
- Key features and capabilities of the system
- Challenges faced and how they were addressed
- Potential improvements and future work

## Contributors

[Your Name]

## License

[License Reference Number]

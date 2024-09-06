import os
import pandas as pd
import google.generativeai as genai

# Configure Google Gemini API
genai.configure(api_key="AIzaSyAt8gpOAHgwzOGOhpJATz88vxMeeM1q2Lg")

def generate_prompt_for_dataframe(invoice_text):
    prompt = (
        """CONTEXT: You are an AI assistant, and you'll be provided with documents in .txt format that contain billing invoice data.

        TASK: Extract key information from the document and convert it into a format suitable for creating a DataFrame. The key information in the document will serve as column names, and their corresponding values will be the row data. 

        FORMAT:
        - You should output the data in a table-like format where each key is a column header and each corresponding value is the data for that column. 
        - Use a clear structure, such as separating the columns and rows with `|` symbols, to ensure the data can be easily converted into a DataFrame.
        - Ensure that all column names and corresponding values are aligned.

        IMPORTANT:
        1. Not all data will be separated by colons. Sometimes, multiple spaces or lines might separate key-value pairs. Handle this by treating the first clear label as the column header and its corresponding data as the value.
        2. Ignore irrelevant text such as horizontal lines, titles, and descriptions that are not part of the invoice data.
        3. Extract all essential billing information from the document, such as transaction date, invoice number, amount, tax information, and other details.
        4. If a value is missing or not present, leave the cell empty in the row.
        
        EXAMPLE FORMAT:

        | Transaction Date | Amount | GST Number | Name              | Address       | OPD No      | OPD Date    | Bill No         | Department       | Doctor          | Total  | Amount Paid |
        |------------------|--------|------------|-------------------|---------------|-------------|-------------|-----------------|------------------|-----------------|--------|-------------|
        | 25/07/2024       | 2000   | XXYY27A    | MRS MANJU LATA SINGH | Sandeep Vihar | O01001888633| 12/08/2024  | MHW24OCS0192936 | INTERNAL MEDICINE | Dr Adithi Nagaraju | 0.00   | 0.00        |

        Please generate the output in a similar table format based on the following invoice data:

        INVOICE DATA:
        {}
        """
    ).format(invoice_text)
    return prompt
def extract_data_as_dataframe(invoice_text):
    prompt = generate_prompt_for_dataframe(invoice_text)
    
    # Initialize the Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Generate content
    response = model.generate_content(prompt)
    
    # Extract the structured tabular text from the response
    if response.candidates:
        extracted_text = response.candidates[0].content.parts[0].text
        print(f"Extracted table text: {extracted_text}")
    else:
        extracted_text = ''
    
    # Convert the extracted table text to a DataFrame
    try:
        rows = [row.split('|')[1:-1] for row in extracted_text.strip().split('\n') if '|' in row]
        columns = rows[0]  # First row as column names
        data = rows[1:]    # Subsequent rows as data

        # Ensure unique column names if duplicates exist
        columns = make_columns_unique(columns)

        # Create DataFrame with unique column headers
        df = pd.DataFrame(data, columns=[col.strip() for col in columns])

    except Exception as e:
        print(f"Error processing table: {e}")
        df = pd.DataFrame()

    return df

def make_columns_unique(columns):
    """Make column names unique by appending a counter to duplicates."""
    counts = {}
    unique_columns = []
    for col in columns:
        if col not in counts:
            counts[col] = 0
        else:
            counts[col] += 1
            col = f"{col}_{counts[col]}"
        unique_columns.append(col)
    return unique_columns

def read_invoice_texts(directory):
    invoice_texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                invoice_text = file.read()
                invoice_texts.append({"filename": filename, "text": invoice_text})
    return invoice_texts

# Read invoice texts
directory_path = 'text_files'
invoice_texts = read_invoice_texts(directory_path)

# Extract and store structured data as DataFrame
df_list = []

for invoice in invoice_texts:
    df = extract_data_as_dataframe(invoice['text'])
    df['filename'] = invoice['filename']
    df_list.append(df)
    # df_invoice = pd.DataFrame(df_list)
print(df_list)

# Concatenate all DataFrames
if df_list:
    final_df = pd.concat(df_list, ignore_index=True)
    # Save to CSV
    final_df.to_csv('extracted_invoices.csv', index=False)
else:
    print("No dataframes to concatenate.")
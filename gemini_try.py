import ast
import os
import json
import pandas as pd
import google.generativeai as genai

# Configure Google Gemini API
genai.configure(api_key="AIzaSyAt8gpOAHgwzOGOhpJATz88vxMeeM1q2Lg")

def generate_prompt(invoice_text):
    prompt = (
        f"""CONTEXT: You are an AI assistant and will be provided with billing invoice data in .txt format.
        
        TASK: Your job is to extract key information from the following invoice data and convert it into valid JSON format.

        INVOICE DATA:
        {invoice_text}

        INSTRUCTIONS:
        - Identify the key fields (e.g., "Transaction Date", "Amount", "GST Number") and use them as the JSON keys.
        - Extract the values and associate them with the corresponding keys.
        - Ignore irrelevant text or formatting (such as horizontal lines or empty lines).
        - Ensure the output is in valid JSON format.
        
        EXAMPLE:
        Input:
        Transaction Date: 25/07/2024
        Amount: 2000
        GST Number: XXYY27A

        Output:
        {{
          "Transaction Date": "25/07/2024",
          "Amount": 2000,
          "GST Number": "XXYY27A"
        }}

        Now, extract and convert the above invoice data to JSON:
        """
    )
    return prompt

def extract_structured_data_gemini(invoice_text):
    prompt = generate_prompt(invoice_text)
    
    # Initialize the Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Generate content
    response = model.generate_content(prompt)
    
    # Extract the text from the response
    if response.candidates:
        extracted_text = response.candidates[0].content.parts[0].text
        print(f"Extracted text: {extracted_text}")  # Debugging purpose
    else:
        extracted_text = ''
    
    # Extract JSON part from the response
    try:
        json_start = extracted_text.index('{')
        json_str = extracted_text[json_start:].strip()
        print(f"JSON str (before parsing): {json_str}")  # Print for debugging
        
        json_str = json_str.replace("Generated/Printed By", "Generated_or_Printed_By")
        # Check if the JSON string is empty
        if json_str == "":
            print("Empty or invalid JSON string!")
        
        # Attempt to parse using json.loads
        data_dict = json.loads(json_str)
        print(f"Parsed JSON: {data_dict}")
        
    except (ValueError, SyntaxError, IndexError, json.JSONDecodeError) as e:
        print(f"Failed to parse JSON with json.loads: {e}")
        
        # Try alternative parsing method
        try:
            data_dict = ast.literal_eval(json_str)
            print(f"Parsed with literal_eval: {data_dict}")
        except (ValueError, SyntaxError) as ast_e:
            print(f"Failed to parse with literal_eval: {ast_e}")
            data_dict = {}
    
    return data_dict

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

# Extract and store structured data
structured_data_list = []

for invoice in invoice_texts:
    structured_data = extract_structured_data_gemini(invoice['text'])
    print("Data: ", structured_data)
    structured_data['filename'] = invoice['filename']
    structured_data_list.append(structured_data)

# Convert to DataFrame
df = pd.DataFrame(structured_data_list)

# Save to CSV
df.to_csv('extracted_invoices.csv', index=False)

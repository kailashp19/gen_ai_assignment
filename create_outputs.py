import ast
import os
from pathlib import Path
import json
import pandas as pd
import google.generativeai as genai

api_key = os.getenv('API_KEY')

# Configure Google Gemini API
genai.configure(api_key='AIzaSyAt8gpOAHgwzOGOhpJATz88vxMeeM1q2Lg')

def generate_prompt(invoice_text):
    print(invoice_text)
    prompt = (
        """CONTEXT: You are an AI assistant and will be provided with billing invoice data in .txt format.
        
        TASK: Your job is to extract key information from the following invoice data and convert it into valid JSON format.

        REMEMBER:
        1. *Accuracy is crucial*. Double-check all extracted information.
        2. *Maintain consistency* in data representation across different invoices.
        3. *Handle edge cases* gracefully (e.g., missing information, unusual formats).

        INFORMATION TO EXTRACT:
        - Invoice number and date
        - Supplier and buyer details (including name, addresses, phone number, GST Number if applicable)
        - Line items (products/services, quantities, unit prices, total amount)
        - Tax information (e.g., GST, VAT, sales tax)
        - Total amount (before and after tax)
        - Payment terms and methods
        - Any additional fees or discounts

        ADDITIONAL INFORMATION:
        1. *Identify and extract key details* such as invoice number, date, supplier name, item details, amounts, and addresses.
        2. *Convert all information into English*, especially if the invoice is in another language. Specify the original language if translated.
        3. *Split addresses* into components: street, city, state, and zip code.
        4. *Clearly separate line items*, identifying products/services with quantities and unit prices.
        5. *Identify responsible parties* like buyers, suppliers, and authorized signatories.
        6. *Handle empty invoices* gracefully. If there are no line items, extract whatever information is available.
        7. *Adapt to different invoice types*, including but not limited to:
        - Standard commercial invoices
        - Medical invoices (e.g., including OPD information)
        - Service invoices
        - Utility bills
        8. *Convert amount to USD*. if amount or price is in different format then convert that specific value to USD.

        EXAMPLES:
        1. Standard Invoice:
        Input:
        Invoice Number: INV-001
        Transaction Date: 25/07/2024
        Amount: 2000
        GST Number: XXYY27A

        Output:
        {{
          "Invoice Number": "INV-001",
          "Transaction Date": "25/07/2024",
          "Amount": 2000,
          "GST Number": "XXYY27A"
        }}

        2. Medical Invoice:
        json:
        {{
            "invoice_number": "MED-001",
            "invoice_date": "2024-03-20",
            "hospital_name": "City General Hospital",
            "patient": {{
            "name": "John Doe",
            "id": "PATIENT123"
            }},
            "services": [
            {{
                "description": "OPD Consultation",
                "date": "2024-03-20",
                "doctor": "Dr. Smith",
                "charge": "100.00"
            }},
            {{
                "description": "Blood Test",
                "date": "2024-03-20",
                "charge": "50.00"
            }}
            ],
            "total_charge": "150.00",
            "insurance_coverage": "100.00",
            "patient_duty_charge": "50.00"
        }}

        OTPUT FORMAT:
        - Provide a single, valid JSON object.
        - Ensure all property names and string values are in double quotes.
        - Do not include any text or explanations outside the JSON object.
        - Do not use trailing commas in objects or arrays.

        INVOICE DATA:
        {}
        """
    ).format(invoice_text)
    return prompt

def extract_structured_data_gemini(invoice_text):
    """
    A function to extract relevant invoices data and convert it to a dataframe

    parameters: 
    invoice_texts (string): texts from invoices

    returns:
    df: a dataframe

    """
    prompt = generate_prompt(invoice_text)
    
    # Initialize the Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Generate content
    response = model.generate_content(prompt)
    # print(response)
    
    # Extract the text from the response
    if response.candidates:
        extracted_text = response.text.strip('```json').strip('```').strip()
        print(f"Extracted text: {extracted_text}")  # Debugging purpose
    else:
        extracted_text = ''
    
    # Extract JSON part from the response
    try:
        json_data = json.loads(extracted_text)
        main_df = pd.json_normalize(json_data)
        print(f"Text has been converted to a dataframe {main_df}")
        return main_df
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print("Ensure the input string is a valid JSON format without extra markdown formatting.")
        print("Response Text:", extracted_text)
        return pd.DataFrame()

def read_invoice_texts(directory):
    """
    
    A function to read the text files for further processing.
    
    parameters: 
    directory (string): A path to read the files

    returns:
    invoice_texts (string): An extracted text in string format

    """
    invoice_texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                invoice_text = file.read()
                invoice_texts.append({"filename": filename, "text": invoice_text})
    return invoice_texts

def main():

    invoice_texts = read_invoice_texts("text_files")

    saving_directory = "extracted_dfs"

    # Ensure saving_directory exists
    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)

    # Extract and save each DataFrame individually
    for invoice in invoice_texts:
        df = extract_structured_data_gemini(invoice['text'])
    
        if df is not None and not df.empty:
            # Save each DataFrame to a separate CSV file
            output_filename = os.path.splitext(invoice['filename'])[0] + '_extracted.csv'
            df.to_csv(f"{saving_directory}/{output_filename}", index=False)
            print(f"DataFrame saved as {output_filename}")
        else:
            print(f"Failed to extract data for {invoice['filename']}")

if __name__=="__main__":
    main()
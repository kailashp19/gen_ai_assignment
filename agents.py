import google.generativeai as genai
import os
from typing import TypedDict, Annotated, List, Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate
from langchain.tools import BaseTool, StructuredTool, Tool, tool
import json
import pandas as pd
import sqlite3
from langchain.llms.base import LLM

api_key = os.getenv('API_KEY')

# Configure Google Gemini API
genai.configure(api_key='AIzaSyAt8gpOAHgwzOGOhpJATz88vxMeeM1q2Lg')

# configure langchain api key
langchain_api_key = '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'

conn = sqlite3.connect('example.db')
c = conn.cursor()
print(c)

model = genai.GenerativeModel("gemini-1.5-flash")

@tool("call_gemini_flash", return_direct=True)
def call_gemini_flash(input: str) -> str:
    """
    Calls the Google Gemini Flash LLM with the given input and returns the response.
    Input should be a prompt or data that the LLM will process.
    """
    try:
        # Prepare the prompt for the LLM
        prompt = {"input_text": input}

        # Call the Gemini Flash LLM and get the response
        response = model.generate_content(prompt)

        # Return the LLM's output
        return response.get('output_text', 'No output generated')
    
    except Exception as e:
        return f"Error calling Gemini Flash LLM: {str(e)}"

@tool("read_csv_file", return_direct=True)
def read_csv_file(input:str) -> str:
  """Read the csv file and returns a dataframe"""
  try:
    csv_df = pd.read_csv(input)
    return csv_df
  except Exception as e:
     return f"Error reading the file {str(e)}"

@tool("calculate_profit", return_direct=True)
def calculate_profit(input: str) -> str:
    """
    Analyze the dataset and calculate total sales, total spend on ads, and profit.
    The input should be a DataFrame containing 'sales' and 'total_ad_spend' columns.
    """
    try:
        print(input)
        data = json.loads(input)
        df = pd.DataFrame(data)
        # Check if required columns exist
        if 'sales' not in input.columns or 'total_ad_spend' not in input.columns:
            return "Error: Dataset must contain 'sales' and 'total_ad_spend' columns."

        # Group by 'date' and calculate total sales, spend, and profit per date
        grouped_data = df.groupby('date').agg(
            total_revenue=('sales', 'sum'),
            total_spend=('spend', 'sum'),
            total_profit=lambda x: x['sales'].sum() - x['total_ad_spend'].sum()
        ).reset_index()

        return grouped_data
    except Exception as e:
        return f"Error analyzing the dataset: {str(e)}"

tools = {
  "read_csv_file": read_csv_file,
  "calculate_profit": calculate_profit,
  "call_gemini_flash": call_gemini_flash
}

# Define a function to call the tools and the Gemini Flash LLM
def call_tools_and_llm(tools: dict, input_data: str, llm, query: str):
    """
    Function to call tools and LLM (Gemini Flash) sequentially.
    Args:
        tools (dict): Dictionary of tools
        input_data (str): Input data for the tools
        llm: LLM (Gemini Flash)
        query (str): Query to send to the LLM after tools process data
    """
    try:
        # Call the 'calculate_profit' tool first
        profit_response = tools['calculate_profit'](input_data)

        # Assuming that the 'calculate_profit' tool returns a DataFrame
        if isinstance(profit_response, pd.DataFrame):
            print(profit_response.head())  # Print the first few rows of the profit data

            # Convert the DataFrame to JSON for LLM input
            profit_json = profit_response.to_json(orient="records")

            # Call the Gemini Flash LLM with the profit data for further insights
            llm_response = llm(query + "\n" + profit_json)
            return llm_response
        else:
            return f"Error in calculating profit: {profit_response}"
    
    except Exception as e:
        return f"Error: {str(e)}"


# Step to read the CSV and calculate profit
telecom_df = read_csv_file('D:/Users/kaila/Personal Projects/Team_6_Gen_AI/gen_ai_capstone/gen_ai_assignment/Capstone data sets/Capstone data sets/telecom.csv')

# Check if DataFrame is valid
if isinstance(telecom_df, pd.DataFrame):
    print(telecom_df.head())  # Show the first few rows of the data

    # Convert the DataFrame to JSON for tool input
    telecom_json = telecom_df.to_json(orient="records")

    # Call the tools and Gemini Flash LLM
    gemini_query = "Generate insights from the following telecom dataset profit calculations."
    response = call_tools_and_llm(tools, telecom_json, model, gemini_query)
    print(response)

else:
    print(telecom_df)  # Handle errors during file reading

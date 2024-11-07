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

class SQLiteTest():
    def __init__(self, file_name):
        self.file_name = file_name
        self.table_name = 'fact_marketing'

    def generate_table(self):
        conn = sqlite3.connect('marketing_data.db')
        marketing_df = pd.read_csv(self.file_name)
        marketing_df.to_sql(self.table_name, conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
        print("True")

    def system_prompt(self):
        system_prompt = """
                            You are a SQL query assistant. Your job is to:
                            1. Understand user input in natural language.
                            2. Generate an accurate SQL query based on the input and the given database schema.
                            3. Your response should only consist of SQL Query and no ther keywords accepted by SQL query.
                            4. Convert the above generated SQL query to a dataframe representing a data for the user provided sql query.

                            The database schema includes the following tables:
                            - `fact_marketing (
                            date date,
                            sales integer,
                            sales_from_finance integer,
                            total_ad_spend double,
                            corp_Google_DISCOVERY_spend double,
                            corp_Google_DISPLAY_spend double,
                            corp_Google_PERFORMANCE_MAX_spend double,
                            corp_Google_SEARCH_spend double,
                            corp_Google_SHOPPING_spend double,
                            corp_Google_VIDEO_spend double,
                            corp_Horizon_VIDEO_TIER_1_spend double,
                            corp_Horizon_VIDEO_TIER_2_spend double,
                            corp_Horizon_VIDEO_TIER_3_spend double,
                            corp_Horizon_VIDEO_TIER_BC_spend double,
                            corp_Horizon_VIDEO_TIER_HISP_spend double,
                            corp_Horizon_VIDEO_TIER_NA_spend double,
                            corp_Horizon_VIDEO_TIER_OTT_spend double,
                            corp_Horizon_VIDEO_TIER_SYND_spend double,
                            corp_Impact_AFFILIATE_spend double,
                            corp_Meta_SOCIAL_spend double,
                            corp_Microsoft_AUDIENCE_spend double,
                            corp_Microsoft_SEARCH_CONTENT_spend double,
                            corp_Microsoft_SHOPPING_spend double,
                            local_Google_DISPLAY_spend double,
                            local_Google_LOCAL_spend double,
                            local_Google_PERFORMANCE_MAX_spend double,
                            local_Google_SEARCH_spend double,
                            local_Google_SHOPPING_spend double,
                            local_Meta_SOCIAL_spend double,
                            local_Simpli_fi_GEO_OPTIMIZED_DISPLAY_spend double,
                            local_Simpli_fi_GEO_OPTIMIZED_VIDEO_spend double,
                            local_Simpli_fi_SEARCH_DISPLAY_spend double,
                            local_Simpli_fi_SEARCH_VIDEO_spend double,
                            local_Simpli_fi_SITE_RETARGETING_DISPLAY_spend double,
                            local_Simpli_fi_SITE_RETARGETING_VIDEO_spend double,
                            stock_market_index double,
                            dollar_to_pound double,
                            interest_rates double
                            )`

                            Examples:
                            - User input: "Show me the sales and total ad spend for January 2024."
                            - SQL query: "SELECT sales, total_ad_spend FROM sales_data WHERE date BETWEEN '2024-01-01' AND '2024-01-31';"
                            - Output: sales, total_ad_Spend
                                      1000, 500

                            Remember:
                            1. Your query response only contain SQL query which should be executable in the database.
                            2. Your response should not contains any other keywords apart from SQL query.
                            3. The query response should only contain the column names from fact_marketing table and sql clause
                            which is acceptable by most of the databases.
                            4. You should be able to handle SELECT queries, filtering conditions, and aggregations, and also explain the query output in a natural language response.
                        """
        return self.system_prompt
    
    # Function to send the user input to the LLM and get the SQL query
    def generate_sql_from_input(self, user_input, model):
        # Simulate LLM response here (in practice, this would be sent to an LLM like GPT or Gemini)
        # Assume `llm` is the LLM client interface
        prompt = f"{self.system_prompt}{user_input}"
        
        # Simulate LLM response (you would actually call an API here)
        llm_response = model.generate_content(prompt)

        print(llm_response)
        
        # Extract SQL from the response (or parse LLM output properly)
        sql_query = llm_response.text.strip('```')
        print(sql_query)
        return sql_query

    # Function to execute the SQL query
    def execute_sql_query(self, sql_query):
        # Establish SQLite connection (assume the database is set up and contains relevant data)
        conn = sqlite3.connect('marketing_data.db')
        cursor = conn.cursor()

        # Execute the SQL query
        cursor.execute(sql_query)
        result = cursor.fetchall()
        
        conn.close()
        return result

    # Function to generate a natural language response
    def generate_natural_language_response(self, result, user_input):
        # Generate a response based on the result (this can be customized based on input)
        if result:
            response = f"The query results are: {result}"
        else:
            response = "No matching data found for your query."
        
        return response

    # Main function to process user input and return a natural language response
    def process_user_query(self, model, user_input):
        # Step 1: Generate SQL from natural language input
        sql_query = self.generate_sql_from_input(user_input, model)
        
        if sql_query:
            # Step 2: Execute the SQL query
            result = self.execute_sql_query(sql_query)
            
            # Step 3: Generate a natural language response
            response = self.generate_natural_language_response(result, user_input)
            return response
        else:
            return "Sorry, I couldn't understand your query."


if __name__=="__main__":
    genai.configure(api_key='AIzaSyAt8gpOAHgwzOGOhpJATz88vxMeeM1q2Lg')
    model = genai.GenerativeModel('gemini-1.5-flash')
    sqliteobj = SQLiteTest('C:/Users/kailash.patel/Personal_Documents/GenAI_Assignment_1/gen_ai_assignment/Capstone data sets/Capstone data sets/telecom.csv')
    sqliteobj.generate_table()
    user_input = 'Can you tell me what is the total spend for the year 2024?'
    sqliteobj.process_user_query(model, user_input)
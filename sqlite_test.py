import google.generativeai as genai
import json
import pandas as pd
import sqlite3

class SQLiteTest():
    """
    A class which reads csv file(s) as an input and dumpt it into an SQLite Database. After saving the data into a table, when a
    user asks a query to fetch a particular data, we are providing a prompt which explicitly mentions to generate sql query.
    After a valid sql query is generated, it then executes the SQL query into a database and product an output. This output
    is in the form of database and returns as a result.

    """
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        self.table_name = 'fact_marketing'

    def generate_table(self):
        """
        A method to save the data to an sqlite database.

        Parameters: None

        Returns: None
        """
        conn = sqlite3.connect('marketing_data.db')
        marketing_df = pd.read_csv(self.file_name)
        marketing_df.to_sql(self.table_name, conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()

    def system_prompt(self, user_query: str) -> str:
        """
        A method which takes user query and returns the complete query.

        Parameters
        user_input (str): A user provided input query in natural language format.

        Returns
        system_prompt (str): A complete system query.
        """
        system_prompt = ("""
                            You are a SQL query assistant. Your job is to:
                            1. Understand user input in natural language.
                            2. Generate an accurate SQL query based on the input and the given database schema.
                            3. Your response should only consist of SQL Query and no ther keywords accepted by SQL query.

                            The database schema includes the following table with column name and data type:
                            - `fact_marketing (
                            date date, sales integer, sales_from_finance integer, total_ad_spend double, corp_Google_DISCOVERY_spend double,
                            corp_Google_DISPLAY_spend double, corp_Google_PERFORMANCE_MAX_spend double, corp_Google_SEARCH_spend double,
                            corp_Google_SHOPPING_spend double, corp_Google_VIDEO_spend double, corp_Horizon_VIDEO_TIER_1_spend double,
                            corp_Horizon_VIDEO_TIER_2_spend double, corp_Horizon_VIDEO_TIER_3_spend double,
                            corp_Horizon_VIDEO_TIER_BC_spend double, corp_Horizon_VIDEO_TIER_HISP_spend double,
                            corp_Horizon_VIDEO_TIER_NA_spend double, corp_Horizon_VIDEO_TIER_OTT_spend double,
                            corp_Horizon_VIDEO_TIER_SYND_spend double, corp_Impact_AFFILIATE_spend double,
                            corp_Meta_SOCIAL_spend double, corp_Microsoft_AUDIENCE_spend double,
                            corp_Microsoft_SEARCH_CONTENT_spend double, corp_Microsoft_SHOPPING_spend double,
                            local_Google_DISPLAY_spend double, local_Google_LOCAL_spend double,
                            local_Google_PERFORMANCE_MAX_spend double, local_Google_SEARCH_spend double,
                            local_Google_SHOPPING_spend double, local_Meta_SOCIAL_spend double,
                            local_Simpli_fi_GEO_OPTIMIZED_DISPLAY_spend double, local_Simpli_fi_GEO_OPTIMIZED_VIDEO_spend double,
                            local_Simpli_fi_SEARCH_DISPLAY_spend double, local_Simpli_fi_SEARCH_VIDEO_spend double,
                            local_Simpli_fi_SITE_RETARGETING_DISPLAY_spend double, local_Simpli_fi_SITE_RETARGETING_VIDEO_spend double,
                            stock_market_index double, dollar_to_pound double, interest_rates double
                            )`

                            EXAMPLES:
                            1. USER QUERY: "What are the sales and total ad spend for January 2024."
                            OUTPUT: SELECT sales, total_ad_spend FROM sales_data WHERE date BETWEEN '2024-01-01' AND '2024-01-31';
                         
                            2. USER QUERY: "What is the total profit for the year 2024."
                            OUTPUT: SELECT SUM(sales) - SUM(total_ad_spend) 
                            FROM sales_data WHERE date >= '2024-01-01';
                         
                            Remember:
                            1. Your query response only contain SQL query which should be executable in the database.
                            2. Your response should not contains any other keywords apart from SQL query.
                            3. The query response should only contain the column names from fact_marketing table and sql clause
                            which is acceptable by most of the databases.
                            4. You should be able to handle SELECT queries, filtering conditions, subqueries, various date function and aggregations.

                            user_query
                            {}
                        """).format(user_query)
        return system_prompt
    
    # Function to send the user input to the LLM and get the SQL query
    def generate_sql_from_input(self, user_input: str, model: object) -> str:
        """
        A method which generates SQL query from the user provided query.

        Parameters
        user_input (str): A user provided input query in natural language format.
        model (object): A gemini 1.5 flash model as an LLM

        Returns
        extracted_text (str): An LLM generated SQL query.
        """
        prompt = self.system_prompt(user_input)
        llm_response = model.generate_content(prompt)
        
        # Extract SQL from the response (or parse LLM output properly)
        if llm_response.candidates:
            extracted_text = llm_response.text.replace("```sql\n", "").replace("\n```", "").strip()
            # print(f"Extracted text: {extracted_text}")  # Debugging purpose
            return extracted_text
        else:
            extracted_text = ''
            return extracted_text

    # Function to execute the SQL query
    def execute_sql_query(self, sql_query: str) -> str:
        """
        A method which executes sql query to a database and returns the result in the form of a dataframe.

        Parameters
        sql_query (str): A user provided input query in natural language format.

        Returns
        result (dataframe): A dataframe generated from an sql query.
        """
        # Establish SQLite connection (assume the database is set up and contains relevant data)
        conn = sqlite3.connect('marketing_data.db')
        result = pd.read_sql_query(sql_query, conn)
        conn.close()
        return result

    # Main function to process user input and return a natural language response
    def process_user_query(self, model: object, user_input: str) -> str:
        """
        A method which executes sql query to a database based on the prompt and returns the result in the form of a dataframe.

        Parameters
        sql_query (str): A user provided input query in natural language format.
        model (object): A Gemini 1.5 Flash LLM to convert the natural language query to an SQL Query

        Returns
        result (dataframe): A dataframe generated from an sql query.
        """
        # Step 1: Generate SQL from natural language input
        sql_query = self.generate_sql_from_input(user_input, model)
        
        if sql_query:
            # Step 2: Execute the SQL query
            result = self.execute_sql_query(sql_query)
            print(result)

            # Step 3: Generate a natural language response
            # response = self.generate_natural_language_response(result, user_input)
            return result
        else:
            return "Sorry, I couldn't understand your query."
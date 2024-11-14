import os
import json
import pandas as pd
import google.generativeai as genai

api_key = os.getenv('API_KEY')

# Configure Google Gemini API
genai.configure(api_key='AIzaSyAt8gpOAHgwzOGOhpJATz88vxMeeM1q2Lg')

def generate_response(df, formula_sheet, user_query):
    prompt = (
        """
        CONTEXT: You are an AI assistant and will be provided with marketing data in a dataframe format.
        
        TASK: Your job is to interpret the query from the user related to the marketing data context only present in .csv or .xlsx format.
        Perform the mathematical calculation or retrieve column name/column names or retrieve particular value/values 
        from the data and respond it in the form of a dataframe, formatted as a table with each record on a separate line, based on the result you get from the mathematical calculations 
        or column names asked or values you get.

        REMEMBER:
        1. *Accuracy is crucial*. Double-check all extracted information based on the user query.
        2. *Respond with proper currency value* if needed.
        3. Sometimes user might ask to query to get the particular set of records, and you should be able to analyze the data
        and produce the result as a table with each record on a new line, including column names in the first line.
        4. If you are provided with multiple files representing different kinds of data, you should be able to interpret the user
        query, which might span across multiple files, analyze the data from multiple files, and produce the result.
        5. *Derived KPI values* like profit, turnover, etc., should use the formula.txt file.
        6. *Null/blank* values in the data should be replaced with 0.
        7. *Date* is provided in yyyy-mm-dd format.

        ADDITIONAL INFORMATION:
        1. *Identify and extract key details* like date, investments, returns, profits, discounts, interest rates, dollar to pound, pound to dollar, INR to dollar, Google ad spend, Facebook ad spend, offline marketing expenses, etc.
        2. *Clearly separate line items*, identifying different types of channels like Social Media, Search, Email Marketing, Offline Channel.
        3. *Adapt to different data types*, including but not limited to:
        - date
        - spend/profit/discount/revenue
        - currency conversion
        4. *Only convert amount to specific currency* if the user's request specifies a currency value and the data contains a column with the converted value; otherwise, respond with the non-converted value.
        5. Whenever asked total value\overall value, you should be able to aggregate the numbers and return a single value.
        6. If asked for a particular month\year\date and data is not present in the dataframe, then you should not generate the response.

        RESPONSE FORMAT:
        - Provide results in a *tabular format with each record on a separate line*, with column names in the first row.
        - *Use appropriate number formatting* (e.g., currency with commas for thousands).
        - Avoid wrapping lists in square brackets in the response, showing each record on a new line instead.
        
        EXAMPLES:
        DATA:
        date,sales,sales_from_finance,total_ad_spend,corp_Google_DISCOVERY_spend,corp_Google_DISPLAY_spend,corp_Google_PERFORMANCE_MAX_spend,corp_Google_SEARCH_spend,corp_Google_SHOPPING_spend,corp_Google_VIDEO_spend,corp_Horizon_VIDEO_TIER_1_spend,corp_Horizon_VIDEO_TIER_2_spend,corp_Horizon_VIDEO_TIER_3_spend,corp_Horizon_VIDEO_TIER_BC_spend,corp_Horizon_VIDEO_TIER_HISP_spend,corp_Horizon_VIDEO_TIER_NA_spend,corp_Horizon_VIDEO_TIER_OTT_spend,corp_Horizon_VIDEO_TIER_SYND_spend,corp_Impact_AFFILIATE_spend,corp_Meta_SOCIAL_spend,corp_Microsoft_AUDIENCE_spend,corp_Microsoft_SEARCH_CONTENT_spend,corp_Microsoft_SHOPPING_spend,local_Google_DISPLAY_spend,local_Google_LOCAL_spend,local_Google_PERFORMANCE_MAX_spend,local_Google_SEARCH_spend,local_Google_SHOPPING_spend,local_Meta_SOCIAL_spend,local_Simpli_fi_GEO_OPTIMIZED_DISPLAY_spend,local_Simpli_fi_GEO_OPTIMIZED_VIDEO_spend,local_Simpli_fi_SEARCH_DISPLAY_spend,local_Simpli_fi_SEARCH_VIDEO_spend,local_Simpli_fi_SITE_RETARGETING_DISPLAY_spend,local_Simpli_fi_SITE_RETARGETING_VIDEO_spend,stock_market_index,dollar_to_pound,interest_rates
        2024-10-14,6029,5707,334455.301008181,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1547.28,134731.73,0,15801.95,0,0,0,0,0,0,8754.73,0,0,0,0,0,0,43065.21875,0.766340017318726,4.08
        2024-10-13,4699,4394,310771.383758403,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1542.86,129217.67,0,9087.83,0,0,0,0,0,0,8849.86,0,0,0,0,0,0,42863.859375,0.765569984912872,4.08
        2024-10-12,7014,6595,299959.990704291,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1768.68,114375.55,0,11138.96,0,0,0,0,0,0,8632.99,0,0,0,0,0,0,42863.859375,0.765569984912872,4.08
        2024-10-11,7262,6855,320182.550824835,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2177.62,121933.97,0,15528.02,0,0,0,0,0,0,8318.1,0,0,0,0,0,0,42863.859375,0.765569984912872,4.08
        2024-10-10,6553,6148,336425.814900873,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2160.09,133346.53,0,15523.23,0,0,0,0,0,0,7820.29,0,0,0,0,0,0,42454.12109375,0.765209972858429,4.09
        2024-10-09,6288,5912,326055.870992673,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1812.82,129572.63,0,14749.18,0,0,0,0,0,0,5071.35,0,0,0,0,0,0,42512,0.76366001367569,4.06
        2024-10-08,6342,5952,357322.253690438,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1864.66,148274.9,0,15150.18,0,0,0,0,0,0,4994.39,0,0,0,0,0,0,42080.37109375,0.764249980449677,4.04
        2024-10-07,6773,6390,377007.902321535,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1730.86,160554.5,0,16697.6,0,0,0,0,0,0,5057.38,0,0,0,0,0,0,41954.23828125,0.761600017547607,4.03
        2024-10-06,5012,4692,399029.574233258,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1307.08,198468.8,0,12136.17,0,0,0,0,0,0,5186.96,0,0,0,0,0,0,42352.75,0.761600017547607,3.98
        2024-10-05,7440,7056,388140.422419238,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1731.77,178238.08,0,17851.88,0,0,0,0,0,0,5101.41,0,0,0,0,0,0,42352.75,0.761600017547607,3.98

        1. USER QUERY:
        What is total profit from 2024-10-14 to 2024-10-05
        RESPONSE:
        Total Profit
        12345

        1. USER QUERY:
        What were the dates where total ad spend exceeded 350,000, sales were above 6000, and dollar-to-pound exchange rate was below 0.765?
        RESPONSE:
        Date         Sales         Total Ad Spend         Dollar to Pound
        2024-10-08   6342          357,322.25             0.7642
        2024-10-07   6773          377,007.90             0.7616

        2. USER QUERY:
        List the dates and their corresponding sales and total ad spend for days when corp_Meta_SOCIAL_spend exceeded 150,000 and interest rates were exactly 4.03 or higher.
        RESPONSE:
        Date         Sales         Total Ad Spend         corp Meta SOCIAL spend         Interest Rates
        2024-10-07   6773          377,007.90             160,554.50                     4.03
        2024-10-06   5012          399,029.57             198,468.80                     3.98
        2024-10-05   7440          388,140.42             178,238.08                     3.98

        MARKETING DATA:
        {}
        FORMULA SHEET:
        {}
        USER QUERY:
        {}
        """
    ).format(df, formula_sheet, user_query)
    return prompt

def extract_structured_data_gemini(df, formula_sheet, user_query):
    
    """
    A function to extract relevant invoices data and convert it to a dataframe

    parameters: 
    invoice_texts (string): texts from invoices

    returns:
    df: a dataframe
    """
    
    prompt = generate_response(df, formula_sheet, user_query)
    
    # Initialize the Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Generate content
    response = model.generate_content(prompt)
    # print(response)
    
    # Extract the text from the response
    if response.candidates:
        extracted_text = response.text.strip('```json').strip('```').strip()
        # print(f"Extracted text: {extracted_text}")  # Debugging purpose
        return extracted_text
    else:
        extracted_text = ''
        return extracted_text

    # Extract JSON part from the response
    # try:
    #     json_data = json.loads(extracted_text)
    #     main_df = pd.json_normalize(json_data)
    #     # print(f"Text has been converted to a dataframe {main_df}")
    #     return main_df
    # except json.JSONDecodeError as e:
    #     print(f"JSON decoding error: {e}")
    #     print("Ensure the input string is a valid JSON format without extra markdown formatting.")
    #     print("Response Text:", extracted_text)
    #     return extracted_text
    
    # Extract JSON part from the response
    # return extracted_text

def read_data(directory):
    """
    A function to read the text files for further processing.
    
    parameters: 
    directory (string): A path to read the files

    returns:
    invoice_texts (string): An extracted text in string format
    """
    df = pd.DataFrame()
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".csv"):
            df = pd.read_csv(f"{directory}\{filename}")
        pd.concat([df])
    return df

def main():
    directory = 'Converted text files'
    df = read_data("Capstone data sets")

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                formula_sheet = file.read()

    # user_query = "What were the dates where total ad spend exceeded 400000, sales were above 100000, and dollar-to-pound exchange rate was below 0.75?"
    user_query = "What is the total sales in nov 2024?"
    response = extract_structured_data_gemini(df, formula_sheet, user_query)
    print(response)

if __name__=="__main__":
    main()
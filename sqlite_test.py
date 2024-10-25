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

if __name__=="__main__":
    sqliteobj = SQLiteTest('D:/Users/kaila/Personal Projects/Team_6_Gen_AI/gen_ai_capstone/gen_ai_assignment/Capstone data sets/Capstone data sets/telecom.csv')
    sqliteobj.generate_table()
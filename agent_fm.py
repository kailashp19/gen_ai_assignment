from sqlite_test import SQLiteTest
from rag_model import main as tech_main
from langchain.tools import Tool
from typing import Annotated, Literal, TypedDict, Sequence, Any, List
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import operator
import os

# Configure Google Gemini API
genai.configure(api_key='AIzaSyAt8gpOAHgwzOGOhpJATz88vxMeeM1q2Lg')
llm = genai.GenerativeModel('gemini-1.5-flash')

# Directories
file_dir = "Capstone data sets"  # Directory containing files of various formats
text_dir = "Converted text files"  # Directory to save text files

# Connect to the Qdrant service
client = QdrantClient("http://localhost:6333")

model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# defining the name of the collection
collection_name = "Technical_Support_Agent"

def sqlite_call():
    sqlite_obj = SQLiteTest('Capstone data sets/telecom.csv')
    sqlite_obj.generate_table()
    return sqlite_obj

marketing_tool = Tool(name="Marketing_Agent", func=sqlite_call, description="Handles Marketing related tasks")
technical_tool = Tool(name="Technical_Agent", func=tech_main, description="Handles Marketing related tasks")

marketing_tool = [marketing_tool]
technical_tool = [technical_tool]

marketing_tool_node = ToolNode(marketing_tool)
technical_tool_node = ToolNode(technical_tool)

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }

members = ["Marketing_Agent", "Technical_Agent"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members} which calls the marketing_main function from marketing_data_agent file or tech_main function from technical_agent_file."
    " Given the following user query,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members

class routeResponse(BaseModel):
    next: Literal[tuple(options)]

prompt = "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"

def supervisor_agent(state):
    # Combine system prompt, prompt, and human message
    system_prompt_combined = system_prompt.format(members=", ".join(members))
    prompt_combined = prompt.format(options=", ".join(options))
    human_message = state['messages'][-1].content
    full_message = f"{system_prompt_combined}\n\n{prompt_combined}\n\nUser Message: {human_message}"
    print(full_message)
    response = llm.generate_content(full_message)

    next_agent = "FINISH"  # Default to FINISH to prevent looping if no valid response is found

    if response.candidates:
        extracted_text = response.candidates[0].content.parts[0].text.strip('```json').strip('```').strip()
        print(f"Extracted text: {extracted_text}")

        # Set next_agent if extracted_text matches any valid agent name, else finish
        if extracted_text in members[0]:
           sqlite_obj = sqlite_call()
           result = sqlite_obj.process_user_query(llm, human_message)
           print(result)
        elif extracted_text in members[1]:
            result = tech_main(file_dir, text_dir, client, model, collection_name, human_message, llm)
            print(result)

        else:
            next_agent

    # Log the next agent choice
    print(f"Next agent selected: {next_agent}")

    # Return messages and the next agent to route
    return {
        "messages": [AIMessage(content=extracted_text)],
        "next": next_agent
    }


#    supervisor_chain = prompt | llm.generate_content(routeResponse(**state).dict())
#    return supervisor_chain.invoke(state)

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("Marketing_Agent", marketing_tool_node)
workflow.add_node("Technical_Agent", technical_tool_node)
workflow.add_node("supervisor", supervisor_agent)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")

# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="What is the sales for the year 2023?")
        ]
    }
):
    if "__end__" not in s:
        print("Inside the function")
        print(s)
        print("----")
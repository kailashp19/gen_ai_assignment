from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
import functools
import operator
import google.generativeai as genai
from langgraph.prebuilt import create_react_agent

genai.configure(api_key='AIzaSyAt8gpOAHgwzOGOhpJATz88vxMeeM1q2Lg')

@tool("call_marketing_data_agent", return_direct=True)
def marketing_data_agent(input: str):
    """
    A tool for marketing data agent.
    """
    response = llm.generate_content(input)
    return response

@tool("technical_support_agent", return_direct=True)
def technical_support_agent(input: str):
    """
    A tool for technical support agent.
    """
    response = llm.generate_content(input)
    return response

# Initialize the LLM (GenerativeModel from genai)
llm = genai.GenerativeModel("gemini-1.5-flash")

# Define the tools
tools = [marketing_data_agent, technical_support_agent]

# Define agent members
members = ["MarketingDataAgent", "TechnicalSupportAgent", "Communicate"]

# Custom function to route the user input to the correct tool or LLM
def invoke_custom_agent(input_data):
    # Check for specific keywords to decide which tool to invoke
    if "marketing" in input_data.lower():
        return marketing_data_agent(input_data)
    elif "support" in input_data.lower():
        return technical_support_agent(input_data)
    else:
        # Fallback to LLM for generic responses
        return llm.generate_content(input_data)

# Define the agent node
def agent_node(state, agent, name):
    # Invoke the correct agent (tool or LLM) based on `agent` provided
    result = agent(state)
    return {
        "messages": [HumanMessage(content=result, name=name)]
    }

# Define system prompt and options
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
options = ["FINISH"] + members

# Define route response model
class routeResponse(BaseModel):
    next: Literal[*options]

# Define ChatPromptTemplate for the supervisor
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

# Define supervisor agent function
def supervisor_agent(state):
    supervisor_chain = prompt | llm.generate_content(routeResponse)
    return supervisor_chain.invoke(state)

# Define the state class for message flow
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], operator.add]
    next: str

# Example usage of the custom agent
input_data = "I need help with marketing data analysis"
response = invoke_custom_agent(input_data)
print(response)

# Creating nodes with partial functions to act as graph nodes
data_analysis_node = functools.partial(agent_node, 
                                  agent=marketing_data_agent, 
                                  name="MarketingDataAgent")

technical_support_node = functools.partial(agent_node, 
                              agent=technical_support_agent, 
                              name="TechnicalSupportAgent")
# Store nodes in a dictionary for easy reference
nodes = {
    "MarketingDataAgent": data_analysis_node,
    "TechnicalSupportAgent": technical_support_node,
    "supervisor": supervisor_agent,
}

# Example to manually execute the flow of messages through the nodes
state = AgentState(messages=[HumanMessage(content="I need help with marketing data analysis")], next="MarketingDataAgent")

while state["next"] != "FINISH":
    current_node = nodes.get(state["next"])
    if current_node is None:
        raise ValueError(f"Node {state['next']} not found in the workflow.")
    
    # Process the current node
    result = current_node(state)
    state["messages"].extend(result["messages"])
    
    # Decide next action
    state["next"] = supervisor_agent(state)["next"]

# Final output
for message in state["messages"]:
    print(message.content)
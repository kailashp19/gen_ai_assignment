from marketing_data_agent import main as market_main
from rag_model import main as tech_main
from langchain.tools import Tool
from typing import Annotated, Literal, TypedDict
import google.generativeai as genai
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


model = genai.GenerativeModel('gemini-1.5-flash')

# Configure Google Gemini API
genai.configure(api_key='AIzaSyAt8gpOAHgwzOGOhpJATz88vxMeeM1q2Lg')

marketing_tool = Tool(
    name = "marketing_data_agent",
    func = market_main,
    description = "Generates response for marketing data analysis."
)

technical_tool = Tool(
    name = "technical_support_agent",
    func = tech_main,
    description = "Generates response for technical support."
)

combined_tool = [marketing_tool, technical_tool]

combined_tool_node = ToolNode(combined_tool)

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END

# Define the function that calls the model
def call_model(state: MessagesState):
    
    messages = "".join([message.content for message in state['messages']])
    print(messages)
    response = model.generate_content(messages)

    # We return a list, because this will get added to the existing list
    if response.candidates:
        extracted_text = response.text.strip('```json').strip('```').strip()
        print(f"Extracted text: {extracted_text}")  # Debugging purpose
    else:
        extracted_text = ''
    return {"messages": [AIMessage(content=extracted_text)]}

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", combined_tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="what does error code 002 means from the technical document?")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content
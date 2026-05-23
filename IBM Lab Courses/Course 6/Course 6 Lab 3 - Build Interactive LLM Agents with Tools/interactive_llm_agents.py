# INSTALLING  AND IMPORTING PACKAGES
# %pip install langchain===0.3.25 | tail -n 1
# %pip install langchain-openai===0.3.19 | tail -n 1

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider = "openai")

# Defining The Mathematical Functions
@tool
def add(a: int, b: int) -> int:
    """
    Add a and b.
    
    Args:
        a (int): first integer to be added
        b (int): second integer to be added

    Return:
        int: sum of a and b
    """
    return a + b

tools = [add]
llm_with_tools = llm.bind_tools(tools)

@tool
def subtract(a: int, b:int) -> int:
    """Subtract b from a."""
    return a - b

@tool
def multiply(a: int, b:int) -> int:
    """Multiply a and b."""
    return a * b

# Testing The Functions
tool_map = {"add": add,  "subtract": subtract, "multiply": multiply}
input_ = {"a": 1, "b": 2}
tool_map["add"].invoke(input_)

# Add New Tools To The LLM
tools = [add, subtract, multiply]
llm_with_tools = llm.bind_tools(tools)

# Interacting With The Model
query = "What is 3 + 2?"
chat_history = [HumanMessage(content = query)]

response_1 = llm_with_tools.invoke(chat_history)
chat_history.append(response_1)
print(response_1)

# Parsing Tool Calls
tool_calls_1 = response_1.tool_calls
tool_1_name = tool_calls_1[0]["name"]
tool_1_args = tool_calls_1[0]["args"]
tool_call_1_id = tool_calls_1[0]["id"]
print(f'tool name:\n{tool_1_name}')
print(f'tool args:\n{tool_1_args}')
print(f'tool call ID:\n{tool_call_1_id}')

# Invoking The Tool
tool_response = tool_map[tool_1_name].invoke(tool_1_args)
tool_message = ToolMessage(content = tool_response, tool_call_id = tool_call_1_id)
print(tool_message)
chat_history.append(tool_message)
answer = llm_with_tools.invoke(chat_history)
print(type(answer))
print(answer.content)

# Building The Agent
class ToolCallingAgent:
    def __init__(self, llm):
        self.llm_with_tools = llm.bind_tools(tools)
        self.tool_map = tool_map

    def run(self, query: str) -> str:
        # Step 1: Initial user message
        chat_history = [HumanMessage(content = query)]

        # Step 2: LLM chooses tool
        response = self.llm_with_tools.invoke(chat_history)
        if not response.tool_calls:
            return response.contet
          
        # Step 3: Handle first tool call
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        # Step 4: Call tool manually
        tool_result = self.tool_map[tool_name].invoke(tool_args)

        # Step 5: Send result back to LLM
        tool_message = ToolMessage(content = str(tool_result), tool_call_id = tool_call_id)

        chat_history.extend([response, tool_message])

        # Step 6: Final LLM result
        final_response = self.llm_with_tools.invoke(chat_history)
        return final_response.content


my_agent = ToolCallingAgent(llm)
print(my_agent.run("one plus 2"))
print(my_agent.run("one - 2"))
print(my_agent.run("three times two"))

# Exercise 1: Use the example tool format to create a new tool named calculate_tip that takes a total_bill and tip_percent, and returns the tip amount.
# Define and invoke the tool with sample inputs like total_bill=120, tip_percent=15. Create a tool_map with the calculate_tip tool.

@tool
def calculate_tip(total_bill: int, tip_percent: int) -> int:
    """Calculate tip"""
    return total_bill * tip_percent * 0.01
  
inputs = {"total_bill": 120, "tip_percent": 15}
calculate_tip.invoke(inputs)
tool_map = {"calculate_tip": calculate_tip}

# Example 2: Simulate a user query like "How much should I tip on $60 at 20%?". Bind the tool to the predefined llm and prompt the LLM with the query above. 
# Then parse the LLM response for the tool calling details and invoke the tool accordingly. Finally, take the entire chat history and prompt the LLM for a final output.

query = "How much should I tip on $60 at 20%?"
llm_with_tool = llm.bind_tools([calculate_tip])
chat_history = [HumanMessage(content = query)]
response = llm_with_tool.invoke(chat_history)
tool_calls = response.tool_calls
tool_name = tool_calls[0]["name"]
tool_args = tool_calls[0]["args"]
tool_call_id = tool_calls[0]["id"]
tool_response = tool_map[tool_name].invoke(tool_args)
tool_message = ToolMessage(content = tool_response, tool_call_id = tool_call_id)
chat_history.extend([response, tool_message])
result = llm_with_tool.invoke(chat_history)
print(result.content)

# Example 3: Create an agent to automate the entire process you previously completed

class TipAgent:
    def __init__(self, llm):
        self.llm_with_tool = llm.bind_tools([calculate_tip])
        self.tool_map = tool_map

    def run(self, query: str) -> str:
        chat_history = [HumanMessage(content = query)]
        response = llm_with_tool.invoke(chat_history)
        tool_calls = response.tool_calls
        tool_name = tool_calls[0]["name"]
        tool_args = tool_calls[0]["args"]
        tool_call_id = tool_calls[0]["id"]
        tool_response = tool_map[tool_name].invoke(tool_args)
        tool_message = ToolMessage(content = tool_response, tool_call_id = tool_call_id)
        chat_history.extend([response, tool_message])
        return llm_with_tool.invoke(chat_history).content
      
agent = TipAgent(llm)
agent.run("How much should I tip on $60 at 20%?")

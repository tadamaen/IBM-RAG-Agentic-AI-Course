# INSTALLING AND IMPORTING PACKAGES
# %pip install langchain==0.3.23 | tail -n 1 
# %pip install langchain-ibm==0.3.10 | tail -n 1 
# %pip install langchain-community==0.3.16 | tail -n 1 
# %pip install wikipedia==1.4.0 | tail -n 1
# %pip install openai==1.77.0 | tail -n 1
# %pip install langchain-openai==0.3.16 | tail -n 1
# %pip install langgraph==0.6.1 | tail -n 1

from langchain_ibm import ChatWatsonx
from langchain.agents import AgentType
from langchain.agents import Tool
import re

# Loading The LLM
llm = ChatWatsonx(model_id = "ibm/granite-4-h-small",
                  url = "https://us-south.ml.cloud.ibm.com",
                  project_id = "skills-network")

response = llm.invoke("What is tool calling in langchain?")
print("\nResponse Content: ", response.content)

# Function to add numbers
def add_numbers(inputs:str) -> dict:
    numbers = [int(x) for x in inputs.replace(",", "").split() if x.isdigit()]
    result = sum(numbers)
    return {"result": result}

add_numbers("1 2")

# Adding a Tool instance
add_tool = Tool(name = "AddTool", func = add_numbers, description = "Adds a list of numbers and returns the result.")
print("tool object", add_tool)
print("Tool Name:")
print(add_tool.name)
print("Tool Description:")
print(add_tool.description)
print("Tool Function:")
print(add_tool.invoke)

# Example of calling the tool function
print("Calling Tool Function:")
test_input = "10 20 30 a b" 
print(add_tool.invoke(test_input))

# Tool function with Decorator
from langchain_core.tools import tool
from typing import List
from typing import Dict, Union

@tool
def add_numbers(inputs:str) -> dict:
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    result = sum(numbers)
    return {"result": result}

print("Name: \n", add_numbers.name)
print("Description: \n", add_numbers.description) 
print("Args: \n", add_numbers.args) 

# Test Example
test_input = "what is the sum between 10, 20 and 30 " 
print(add_numbers.invoke(test_input))

# 2nd Example: Summing Absolute Values
@tool
def add_numbers_with_options(numbers: List[float], absolute: bool = False) -> float:
    if absolute:
        numbers = [abs(n) for n in numbers]
    return sum(numbers)

print(add_numbers_with_options.invoke({"numbers" : [-1.1,-2.1,-3.0], "absolute" : False}))
print(add_numbers_with_options.invoke({"numbers" : [-1.1,-2.1,-3.0], "absolute" : True}))

# Improved tool return types with Python typing
@tool
def sum_numbers_with_complex_output(inputs: str) -> Dict[str, Union[float, str]]:
    matches = re.findall(r'-?\d+(?:\.\d+)?', inputs)
    if not matches:
        return {"result": "No numbers found in input."}
    try:
        numbers = [float(num) for num in matches]
        total = sum(numbers)
        return {"result" : total}
    except Exception as e:
        return {"result" : f"Error during summation: {str(e)}"}

@tool
def sum_numbers_from_text(inputs: str) -> float:
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    result = sum(numbers)
    return result

# Initialize The Agent
from langchain.agents import initialize_agent
agent = initialize_agent([add_tool], llm, agent = "zero-shot-react-description", verbose = True, handle_parsing_errors = True)
response = agent.run("In 2023, the US GDP was approximately $27.72 trillion, while Canada's was around $2.14 trillion and Mexico's was about $1.79 trillion what is the total.")
agent.invoke({"input": "Add 10, 20, two and 30"})

# Using IBM GRANITE
agent_2 = initialize_agent([sum_numbers_from_text], llm, agent = "structured-chat-zero-shot-react-description", verbose = True, handle_parsing_errors = True)
response = agent_2.invoke({"input": "Add 10, 20 and 30"})
print(response)

# Trying out other models
from langchain_openai import ChatOpenAI
llm_ai = ChatOpenAI(model = "gpt-4.1-nano")
agent_3 = initialize_agent([sum_numbers_with_complex_output], llm_ai, agent = "openai-functions", verbose = True, handle_parsing_errors = True)
response = agent_3.invoke({"input": "Add 10, 20 and 30"})
print(response)

agent_2 = initialize_agent([add_numbers_with_options], llm, agent = "structured-chat-zero-shot-react-description", verbose = True)
response = agent_2.invoke({"input" : "Add -10, -20, and -30 using absolute values."})
print(response)

agent_openai = initialize_agent([add_numbers_with_options], llm_ai, agent = "openai-functions", verbose = True)
response = agent_openai.invoke({"input" : "Add -10, -20, and -30 using absolute values."})
print(response)

# Using create_react_agent in LangGraph Instead
from langgraph.prebuilt import create_react_agent

agent_exec = create_react_agent(model=llm_ai, tools = [sum_numbers_from_text])
msgs = agent_exec.invoke({"messages" : [("human", "Add the numbers -10, -20, -30")]})
print(msgs["messages"][-1].content)

# Subtraction Tool
@tool
def subtract_numbers(inputs: str) -> dict:
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    if not numbers:
        return {"result": 0}
    result = -1 * numbers[0]
    for num in numbers[1:]:
        result -= num
    return {"result" : result}

print("Name: \n", subtract_numbers.name)
print("Description: \n", subtract_numbers.description) 
print("Args: \n", subtract_numbers.args) 

# Test Example:
print("Calling Tool Function:")
test_input = "10 20 30 and four a b"
print(subtract_numbers.invoke(test_input))

# Multiplication And Division Tools
# Multiplication Tool
@tool
def multiply_numbers(inputs: str) -> dict:
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    if not numbers:
        return {"result": 1}
    result = 1
    for num in numbers:
        result *= num
        print(num)
    return {"result" : result}

@tool
def divide_numbers(inputs: str) -> dict:
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    if not numbers:
        return {"result": 0}
    result = numbers[0]
    for num in numbers[1:]:
        result /= num
    return {"result" : result}

# Testing multiply_tool
multiply_test_input = "2, 3, and four "
multiply_result = multiply_numbers.invoke(multiply_test_input)
print("--- Testing MultiplyTool ---")
print(f"Input: {multiply_test_input}")
print(f"Output: {multiply_result}")

# Testing divide_tool
divide_test_input = "100, 5, two"
divide_result = divide_numbers.invoke(divide_test_input)
print("--- Testing DivideTool ---")
print(f"Input: {divide_test_input}")
print(f"Output: {divide_result}")

# Building The Agent
tools = [add_numbers,subtract_numbers, multiply_numbers, divide_numbers]
math_agent = create_react_agent(model = llm_ai,
                                tools = tools,
                                prompt = "You are a helpful mathematical assistant that can perform various operations. Use the tools precisely and explain your reasoning clearly.")

response = math_agent.invoke({"messages": [("human", "What is 25 divided by 4?")]})
final_answer = response["messages"][-1].content
print(final_answer)

response_2 = math_agent.invoke({"messages": [("human", "Subtract 100, 20, and 10.")]})
final_answer_2 = response_2["messages"][-2].content
print(final_answer_2)

# Testing The Various Tools
print("\n--- Testing MultiplyTool ---")
response = math_agent.invoke({"messages": [("human", "Multiply 2, 3, and four.")]})
print("Agent Response:", response["messages"][-1].content)

print("\n--- Testing DivideTool ---")
response = math_agent.invoke({"messages": [("human", "Divide 100 by 5 and then by 2.")]})
print("Agent Response:", response["messages"][-1].content)

# Modify the subtract function to handle the edge cases
@tool
def new_subtract_numbers(inputs: str) -> dict:
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    if not numbers:
        return {"result": 0}
    result = numbers[0]
    for num in numbers[1:]:
        result -= num
    return {"result": result}

tools_updated = [add_numbers, new_subtract_numbers, multiply_numbers, divide_numbers]
math_agent_new = create_react_agent(model = llm, 
                                    tools = tools_updated,
                                    prompt = "You are a helpful mathematical assistant that can perform various operations. Use the tools precisely and explain your reasoning clearly.")
print("agent", math_agent_new)

# Test Cases(taking into account new subtract function)
test_cases = [
    {
        "query": "Subtract 100, 20, and 10.",
        "expected": {"result": 70},
        "description": "Testing subtraction tool with sequential subtraction."
    },
    {
        "query": "Multiply 2, 3, and 4.",
        "expected": {"result": 24},
        "description": "Testing multiplication tool for a list of numbers."
    },
    {
        "query": "Divide 100 by 5 and then by 2.",
        "expected": {"result": 10.0},
        "description": "Testing division tool with sequential division."
    },
    {
        "query": "Subtract 50 from 20.",
        "expected": {"result": -30},
        "description": "Testing subtraction tool with negative results."
    }
]

# Model Evaluation [Optional]:
correct_tasks = []
for index, test in enumerate(test_cases, start=1):
    query = test["query"]
    expected_result = test["expected"]["result"]
    print(f"\n--- Test Case {index}: {test['description']} ---")
    print(f"Query: {query}")

    response = math_agent_new.invoke({"messages": [("human", query)]})
    tool_message = None
    for msg in response["messages"]:
        if hasattr(msg, 'name') and msg.name in ['add_numbers', 'new_subtract_numbers', 'multiply_numbers', 'divide_numbers']:
            tool_message = msg
            break
    if tool_message:
        import json
        tool_result = json.loads(tool_message.content)["result"]
        print(f"Tool Result: {tool_result}")
        print(f"Expected Result: {expected_result}")
      
        if tool_result == expected_result:
            print(f"✅ Test Passed: {test['description']}")
            correct_tasks.append(test["description"])
        else:
            print(f"❌ Test Failed: {test['description']}")
    else:
        print("❌ No tool was called by the agent")
print("\nCorrectly passed tests:", correct_tasks)

# Exploring LangChain's built-in tools
from langchain_community.utilities import WikipediaAPIWrapper

# Function to serach in Wikipedia
@tool
def search_wikipedia(query: str) -> str:
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)

# Update your tools list to include the Wikipedia tool
tools_updated = [add_numbers, new_subtract_numbers, multiply_numbers, divide_numbers, search_wikipedia]
math_agent_updated = create_react_agent(model = llm_ai, 
                                        tools = tools_updated,
                                        prompt = """You are a helpful assistant that can perform various mathematical operations and look up information. 
                                                  Use the tools precisely and explain your reasoning clearly.""")

query = "What is the population of Canada? Multiply it by 0.75"
response = math_agent_updated.invoke({"messages": [("human", query)]})

print("\nMessage sequence:")
for i, msg in enumerate(response["messages"]):
    print(f"\n--- Message {i+1} ---")
    print(f"Type: {type(msg).__name__}")
    if hasattr(msg, 'content'):
        print(f"Content: {msg.content}")
    if hasattr(msg, 'name'):
        print(f"Name: {msg.name}")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"Tool calls: {msg.tool_calls}")


## **Exercise: Create a power tool to calculate exponents**

#### **Step 1: Create the power tool**

# 1. **Define the Tool Function**:
#    - Create a Python function named `calculate_power` that takes a string as input. The string will contain two numbers: the base (\( x \)) and the exponent (\( y \)).
#    - The function should extract the numbers, calculate \( x^y \), and return the result as a dictionary with the key `"result"`.

def calculate_power(input_text: str) -> dict:
    match = re.search(r"(\d+(?:\.\d+)?)\s*\^+\s*(\d+(?:\.\d+)?)", input_text)
    if match:
        base = float(match.group(1))
        exponent = float(match.group(2))
        return {"result": base ** exponent}
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to\s+the\s+power\s+of)\s*(\d+(?:\.\d+)?)", input_text, re.IGNORECASE)
    if match:
        base = float(match.group(1))
        exponent = float(match.group(2))
        return {"result": base ** exponent}
    try:
        numbers = [float(num) for num in input_text.replace(",", " ").split()]
        if len(numbers) != 2:
            return {"result": "Invalid input. Please provide exactly two numbers."}
        base, exponent = numbers
        return {"result": base ** exponent}
    except ValueError:
        return {"result": "Invalid input format. Provide input like '2 3', '2^3', or '2 to the power of 3'."}

# 2. Create the tool object:
# Use the Tool class from LangChain to create a tool object for the calculate_power function.
# Provide a name, description, and the function to the tool.

power_tool = Tool(name = "PowerTool", func = calculate_power,
                  description = "Calculates the power of a number (x^y). Input should be two numbers: base and exponent.")

# 3. Set up the agent:
# Use the initialize_agent function from LangChain to create an agent.
# Include the power_tool in the list of tools provided to the agent.
# Specify the agent type (e.g., zero-shot-react-description).
tools = [power_tool]
agent = initialize_agent(tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True, handle_parsing_errors = True)

# 4. Test the Agent Using the run Function:
# Use the run function of the agent to test its ability to calculate powers.
# Pass natural language queries to the agent and observe its responses.
agent.run("Calculate 5 to the power of 2.")
